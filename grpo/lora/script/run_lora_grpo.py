#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA (PEFT) RL training with a GRPO-style objective and LLM-as-judge reward.

This script trains *LoRA adapters only* (base model frozen).

vLLM integration (optional):
- Training side saves LoRA adapter checkpoints (PEFT format: adapter_config.json + adapter_model.*)
- vLLM side loads/reloads the adapter via OpenAI-compatible runtime LoRA endpoints:
    POST /v1/unload_lora_adapter
    POST /v1/load_lora_adapter
- Sampling for GRPO candidates can be done via vLLM OpenAI /v1/completions.

Notes:
- If you run multi-GPU with FSDP, LoRA params are marked as "ignored_states" (or "ignored_parameters" on older FSDP) so:
    * LoRA params are NOT sharded (easy to save as an adapter)
    * Their grads are NOT auto-synchronized by FSDP, so we all-reduce LoRA grads manually.

Reward shaping:
- Medical judge score: {0, 0.5, 1}
- SMART score: integer 0..10 (helpful/structured/clear, but NOT correctness)

Anti reward-hacking (SMART-only farming) defenses:
- SMART add is gated by medical score (soft/hard)
- SMART add is capped
- Optional hack penalty when (medical low) & (SMART high)
"""

import os
import re
import json
import random
import argparse
import hashlib
import time
import inspect
import threading
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Any, Dict
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry
    _URLLIB3_RETRY_AVAILABLE = True
except Exception:
    Retry = None  # type: ignore
    _URLLIB3_RETRY_AVAILABLE = False

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# LoRA (PEFT)
try:
    from peft import LoraConfig, get_peft_model
    _PEFT_AVAILABLE = True
except Exception:
    _PEFT_AVAILABLE = False

# FSDP
try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        MixedPrecision,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    _FSDP_AVAILABLE = True
except Exception:
    _FSDP_AVAILABLE = False


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_text_from_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [extract_text_from_content(x) for x in content]
        return "\n".join([p for p in parts if p])
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        inner = content.get("content")
        if isinstance(inner, str):
            return inner
        if isinstance(inner, (list, dict)):
            return extract_text_from_content(inner)
        parts = [v for v in content.values() if isinstance(v, str)]
        return "\n".join(parts)
    return ""


class ChatMLJsonl(IterableDataset):
    """
    Minimal ChatML jsonl reader:
      {"messages":[{"role":"user","content":...},{"role":"assistant","content":...}, ...]}

    Rank sharding: i % world_size == rank lines only.
    """
    def __init__(self, path: str, rank: int = 0, world_size: int = 1):
        self.path = path
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i % self.world_size != self.rank:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                msgs = rec.get("messages") or []
                umsgs = [m for m in msgs if m.get("role") == "user"]
                amsgs = [m for m in msgs if m.get("role") == "assistant"]
                if not umsgs:
                    continue
                user = extract_text_from_content(umsgs[0].get("content")).strip()
                ref = extract_text_from_content(amsgs[0].get("content")).strip() if amsgs else ""
                if not user:
                    continue
                yield {"prompt": user, "reference": ref}


def _split_csv(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    items = [x.strip() for x in s.split(",") if x.strip()]
    return items or None


# ===== LoRA無効化コンテキストマネージャー（KL計算用）=====
from contextlib import contextmanager

@contextmanager
def lora_disabled(model: nn.Module):
    """
    Temporarily disable LoRA adapters for reference forward pass (KL calculation).
    
    Supports:
      1) PEFT: PeftModel.disable_adapter() as context manager
      2) Transformers adapter integration: disable_adapters()/enable_adapters()
      3) Fallback: do nothing (KL ~ 0)
    """
    # 1) PEFT: PeftModel.disable_adapter() is a context manager
    if hasattr(model, "disable_adapter") and callable(getattr(model, "disable_adapter")):
        try:
            with model.disable_adapter():  # <- must use as context manager
                yield
            return
        except TypeError:
            # disable_adapter が with 不可の実装だった場合は次へ
            pass

    # 2) Transformers adapter integration: disable_adapters()/enable_adapters()
    if hasattr(model, "disable_adapters") and hasattr(model, "enable_adapters"):
        model.disable_adapters()
        try:
            yield
        finally:
            model.enable_adapters()
        return

    # 3) Fallback: do nothing (KL ~ 0)
    yield


@contextmanager
def dropout_disabled(model: nn.Module):
    """
    Temporarily disable all Dropout layers while keeping model in train() mode.
    
    This allows gradient checkpointing to work (which requires training=True)
    while preventing dropout noise from affecting the forward pass.
    """
    saved: List[Tuple[nn.Module, float]] = []
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            saved.append((m, m.p))
            m.p = 0.0
    try:
        yield
    finally:
        for m, p in saved:
            m.p = p


def apply_lora_peft(
    model: nn.Module,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Apply typical LoRA (PEFT) to a decoder LM.
    - Base params frozen
    - Only LoRA params trainable
    """
    if not _PEFT_AVAILABLE:
        raise RuntimeError("[LoRA] peft が必要です。`pip install peft` を実行してください。")

    for p in model.parameters():
        p.requires_grad = False

    if target_modules is None:
        linear_leaf_names = {name.split(".")[-1] for name, m in model.named_modules() if isinstance(m, nn.Linear)}
        if not linear_leaf_names:
            raise RuntimeError("[LoRA] nn.Linear が見つかりませんでした。")

        typical = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        alternatives = [
            "qkv_proj", "Wqkv", "wqkv",
            "c_attn", "c_proj",
            "fc1", "fc2",
            "w1", "w2", "w3",
        ]

        present_typical = [t for t in typical if t in linear_leaf_names]
        present_alt = [t for t in alternatives if t in linear_leaf_names]

        if present_typical:
            target_modules = present_typical
        elif present_alt:
            target_modules = present_alt
        else:
            target_modules = sorted(linear_leaf_names)
            print(
                "[WARN] Typical LoRA target module names were not found; "
                "falling back to ALL Linear leaf module names.",
                flush=True,
            )

    print(f"[INFO] LoRA target_modules={target_modules}", flush=True)

    lora_cfg = LoraConfig(
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)

    for n, p in model.named_parameters():
        p.requires_grad = ("lora_" in n)

    total = 0
    trainable = 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    print(
        f"[INFO] trainable(lora)={trainable:,} / total={total:,} "
        f"({trainable / max(1, total) * 100:.4f}%)",
        flush=True,
    )
    return model


# ===== ignored_* な LoRA params は自動でGPUに移動されないので in-place でGPUへ移動 =====
def _move_params_inplace(params: List[torch.nn.Parameter], device: torch.device) -> None:
    for p in params:
        if p is None:
            continue
        if p.device != device:
            p.data = p.data.to(device, non_blocking=True)
            if p.grad is not None:
                p.grad.data = p.grad.data.to(device, non_blocking=True)


# -------- Judge prompt (medical correctness) --------

JUDGE_SYSTEM = (
    "You are a strict medical grading assistant.\n"
    "You evaluate the candidate answer for medical correctness, completeness, safety (no hallucinations), and clarity.\n"
    "You MUST follow this procedure:\n"
    "1. First, think step by step in detail about the question and the candidate answer.\n"
    "   - Carefully compare it with the reference answer if provided.\n"
    "   - Identify correct points, missing points, and any unsafe or incorrect statements.\n"
    "2. After you finish your reasoning, on the FINAL line output exactly ONE of:\n"
    "   [1], [0.5], [0]\n"
    "   - The final line must contain ONLY the score token (no extra text).\n"
    "Scoring rule:\n"
    "- [1]   : completely correct and safe. No important errors or omissions.\n"
    "- [0.5] : partially correct. Some important parts are missing or slightly wrong, "
    "but parts of the answer are still useful.\n"
    "- [0]   : incorrect, unsafe, or mostly wrong / misleading.\n"
)

JUDGE_TEMPLATE = (
    "User question:\n{prompt}\n\n"
    "{ref_block}"
    "Candidate answer:\n{candidate}\n\n"
    "Think step by step according to the rubric above. "
    "After you finish your reasoning, output the final score on the LAST line as one of [1], [0.5], or [0] "
    "with no additional text on that line."
)


def build_judge_text(prompt: str, candidate: str, reference: Optional[str]) -> str:
    ref_block = ""
    if reference:
        ref_block = f"Ideal reference answer (for evaluator's eyes only):\n{reference}\n\n"
    return JUDGE_TEMPLATE.format(prompt=prompt, ref_block=ref_block, candidate=candidate)


# -------- SMART prompt (helpfulness/structure only; NOT correctness) --------

SMART_SYSTEM = (
    "You are a strict evaluator of answer quality and helpfulness (NOT correctness).\n"
    "You score how well the candidate answer is structured, actionable, and clear for the user question.\n"
    "IMPORTANT:\n"
    "- Do NOT give points for being generic, vague, or just saying 'consult a doctor'.\n"
    "- If the answer is mostly template/disclaimer without addressing specifics of the question, score low.\n"
    "- If it is overly long without adding useful content, score lower.\n"
    "You MUST follow this procedure:\n"
    "1. Think step by step about structure, specificity, readability, and whether it directly answers the question.\n"
    "2. On the FINAL line output exactly ONE of: [0] [1] [2] ... [10]\n"
    "   - The final line must contain ONLY the score token.\n"
    "Rubric:\n"
    "- [10]: Excellent structure, specific and actionable, concise, directly answers.\n"
    "- [7-9]: Good, minor issues.\n"
    "- [4-6]: Some helpful parts, but missing structure/specificity.\n"
    "- [1-3]: Mostly generic, vague, repetitive, or disclaimer-heavy.\n"
    "- [0]: Useless / non-answer / pure template.\n"
)

SMART_TEMPLATE = (
    "User question:\n{prompt}\n\n"
    "{ref_block}"
    "Candidate answer:\n{candidate}\n\n"
    "Evaluate helpfulness and structure (NOT correctness). "
    "After reasoning, output the final score on the LAST line as one of [0]..[10] with no extra text."
)


def build_smart_text(prompt: str, candidate: str, reference: Optional[str]) -> str:
    ref_block = ""
    if reference:
        ref_block = f"(Reference answer is provided for context only. Do NOT grade correctness.)\n{reference}\n\n"
    return SMART_TEMPLATE.format(prompt=prompt, ref_block=ref_block, candidate=candidate)


class ScoreCache:
    def __init__(self, cache_path: Optional[str]):
        self.path = cache_path
        self.mem = {}
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        self.mem[rec["key"]] = rec["score"]
            except Exception:
                pass

    def key(self, prompt: str, candidate: str, reference: Optional[str], tag: str = "") -> str:
        h = hashlib.sha1()
        h.update(tag.encode("utf-8"))
        h.update(b"\x00")
        h.update(prompt.encode("utf-8"))
        h.update(b"\x00")
        h.update(candidate.encode("utf-8"))
        h.update(b"\x00")
        h.update((reference or "").encode("utf-8"))
        return h.hexdigest()

    def get(self, k: str) -> Optional[float]:
        return self.mem.get(k)

    def put(self, k: str, score: float):
        self.mem[k] = float(score)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": k, "score": float(score)}, ensure_ascii=False) + "\n")


# -------- HTTP session helpers (OpenRouter改善: keep-alive + retry + backoff) --------

def _make_retrying_session(
    total_retries: int = 8,
    backoff_factor: float = 0.5,
    status_forcelist: Tuple[int, ...] = (408, 409, 425, 429, 500, 502, 503, 504),
    pool_connections: int = 128,
    pool_maxsize: int = 128,
) -> requests.Session:
    """
    - keep-alive で TLS/接続コストを減らす
    - 429/5xx 等は urllib3 Retry で軽く吸収（対応環境のみ）
    """
    s = requests.Session()
    if _URLLIB3_RETRY_AVAILABLE:
        try:
            retry = Retry(
                total=total_retries,
                connect=total_retries,
                read=total_retries,
                status=total_retries,
                backoff_factor=backoff_factor,
                status_forcelist=status_forcelist,
                allowed_methods=frozenset(["POST"]),
                raise_on_status=False,
                respect_retry_after_header=True,
            )
        except TypeError:
            retry = Retry(
                total=total_retries,
                connect=total_retries,
                read=total_retries,
                status=total_retries,
                backoff_factor=backoff_factor,
                status_forcelist=status_forcelist,
                method_whitelist=frozenset(["POST"]),  # type: ignore
                raise_on_status=False,
                respect_retry_after_header=True,
            )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=pool_connections, pool_maxsize=pool_maxsize)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
    return s


# -------- vLLM OpenAI client (sampling + runtime LoRA) --------

class VLLMOpenAIClient:
    """
    Minimal OpenAI-compatible client for vLLM API server.
    - Sampling: POST /v1/completions
    - Runtime LoRA: POST /v1/load_lora_adapter, /v1/unload_lora_adapter
    """
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 240):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = int(timeout)
        self._sess = _make_retrying_session(total_retries=3, backoff_factor=0.2)

    def _headers(self):
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "lora_grpo/1.0",
        }
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def post_json(self, path: str, payload: dict) -> dict:
        url = self.base_url + path
        resp = self._sess.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
        resp.raise_for_status()

        ct = (resp.headers.get("content-type") or "").lower()
        txt = resp.text or ""
        if not txt.strip():
            return {"ok": True, "status_code": resp.status_code}
        if "application/json" in ct:
            try:
                return resp.json()
            except Exception:
                return {"ok": True, "status_code": resp.status_code, "non_json_body": txt[:2000]}
        return {"ok": True, "status_code": resp.status_code, "non_json_body": txt[:2000]}

    def completions(self, model: str, prompt: str, n: int, max_tokens: int, temperature: float,
                    top_p: float, repetition_penalty: float) -> List[str]:
        """
        Generate completions via vLLM.
        Returns List[str] of generated texts.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "n": int(n),
            "repetition_penalty": float(repetition_penalty),
        }

        data = self.post_json("/v1/completions", payload)
        choices = data.get("choices", []) or []

        outs = []
        for c in choices:
            txt = c.get("text", "")
            outs.append(txt or "")
        return outs

    def unload_lora_adapter(self, lora_name: str) -> None:
        payload = {"lora_name": str(lora_name)}
        _ = self.post_json("/v1/unload_lora_adapter", payload)

    def load_lora_adapter(self, lora_name: str, lora_path: str) -> None:
        payload = {"lora_name": str(lora_name), "lora_path": str(lora_path)}
        _ = self.post_json("/v1/load_lora_adapter", payload)


# -------- Trainer --------

class LoraGRPOTrainer:
    def __init__(self, args):
        self.args = args
        self.wandb_run = None
        self._wandb_mod = None  # store wandb module if available

        if not _FSDP_AVAILABLE:
            raise RuntimeError("[FSDP] torch.distributed.fsdp が使えません。")
        if not torch.cuda.is_available():
            raise RuntimeError("[CUDA] GPU が必要です。")

        # smart judge settings fallback to judge settings
        if getattr(args, "smart_backend", None) is None:
            args.smart_backend = args.judge_backend
        if getattr(args, "smart_api_base", None) is None:
            args.smart_api_base = args.judge_api_base
        if getattr(args, "smart_api_key", None) is None:
            args.smart_api_key = args.judge_api_key
        if getattr(args, "smart_model_name", None) is None:
            args.smart_model_name = args.judge_model_name
        if getattr(args, "smart_max_new_tokens", None) is None:
            args.smart_max_new_tokens = args.judge_max_new_tokens
        if getattr(args, "smart_temperature", None) is None:
            args.smart_temperature = args.judge_temperature
        if getattr(args, "smart_max_retries", None) is None:
            args.smart_max_retries = args.judge_max_retries

        from accelerate import Accelerator
        self.accelerator = Accelerator(mixed_precision="bf16")

        self.rank = self.accelerator.process_index
        self.world_size = self.accelerator.num_processes

        local_rank = int(os.environ.get("LOCAL_RANK", str(self.accelerator.local_process_index)))
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)
        self.local_rank = local_rank

        # --- CPU comm group (prefer gloo) for long object collectives (avoid NCCL watchdog) ---
        self.cpu_group = None
        if dist.is_available() and dist.is_initialized() and self.world_size > 1:
            try:
                self.cpu_group = dist.new_group(backend="gloo")
                if self.accelerator.is_main_process:
                    print("[DIST] created cpu_group backend=gloo for object collectives", flush=True)
            except Exception as e:
                self.cpu_group = None
                if self.accelerator.is_main_process:
                    print(f"[WARN][DIST] failed to create cpu_group: {e}", flush=True)

        # keep-alive + retry
        self._judge_sess = _make_retrying_session(
            total_retries=int(getattr(self.args, "judge_http_retries", 8)),
            backoff_factor=float(getattr(self.args, "judge_http_backoff", 0.5)),
        )
        self._judge_tls = threading.local()
        self._judge_tls.sess = self._judge_sess  # for main thread

        if self.accelerator.is_main_process:
            print(f"[DIST] rank={self.rank} / world_size={self.world_size} local_rank={self.local_rank}", flush=True)

        # ===== samples jsonl path (single file, rank0 writes) =====
        self.samples_jsonl_path = str(getattr(self.args, "samples_jsonl_path", "samples.jsonl") or "samples.jsonl")
        if not os.path.isabs(self.samples_jsonl_path):
            self.samples_jsonl_path = os.path.join(str(self.args.save_dir), self.samples_jsonl_path)
        self.samples_jsonl_every = int(getattr(self.args, "samples_jsonl_every", 1) or 1)
        self.samples_jsonl_text_max_chars = int(getattr(self.args, "samples_jsonl_text_max_chars", 0) or 0)

        if self.accelerator.is_main_process:
            os.makedirs(os.path.dirname(self.samples_jsonl_path), exist_ok=True)
            if bool(getattr(self.args, "samples_jsonl_overwrite", False)):
                with open(self.samples_jsonl_path, "w", encoding="utf-8") as f:
                    f.write("")
            print(f"[SAMPLES] jsonl_path={self.samples_jsonl_path} every={self.samples_jsonl_every} overwrite={getattr(self.args,'samples_jsonl_overwrite',False)}", flush=True)

        if self.accelerator.is_main_process:
            print("[INIT] Detecting transformer layer class (meta init)...", flush=True)

        temp_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        with torch.device("meta"):
            temp_model = AutoModelForCausalLM.from_config(temp_config, trust_remote_code=True)

        if hasattr(temp_model, "model") and hasattr(temp_model.model, "layers"):
            layer_cls = temp_model.model.layers[0].__class__
        elif hasattr(temp_model, "layers"):
            layer_cls = temp_model.layers[0].__class__
        else:
            layer_cls = None
        del temp_model

        if self.accelerator.is_main_process:
            print(f"[INIT] Detected Transformer Layer Class: {layer_cls}", flush=True)

        auto_wrap_policy = None
        if layer_cls is not None:
            auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={layer_cls},
            )

        if self.accelerator.is_main_process:
            print("[LOAD] tokenizer/model:", args.model, flush=True)

        self.tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        config.torch_dtype = torch.bfloat16

        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = True
        if hasattr(base_model.config, "output_router_logits"):
            base_model.config.output_router_logits = False

        base_model.gradient_checkpointing_enable()
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        base_model.train()

        base_model = apply_lora_peft(
            base_model,
            r=int(args.lora_r),
            alpha=int(args.lora_alpha),
            dropout=float(args.lora_dropout),
            target_modules=_split_csv(args.lora_target_modules),
        )

        # Collect LoRA params
        self.lora_named_params: List[Tuple[str, torch.nn.Parameter]] = []
        lora_params: List[torch.nn.Parameter] = []
        for n, p in base_model.named_parameters():
            if p.requires_grad and ("lora_" in n):
                self.lora_named_params.append((n, p))
                lora_params.append(p)
        if not lora_params:
            raise RuntimeError("No trainable LoRA parameters found.")
        self.lora_params = lora_params

        if self.accelerator.is_main_process:
            print(f"[INFO] ignored_lora_params={len(lora_params)}", flush=True)

        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        if self.accelerator.is_main_process:
            print("[FSDP] wrapping model (FULL_SHARD, auto_wrap_policy, ignore LoRA params)...", flush=True)

        fsdp_kwargs = dict(
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=auto_wrap_policy,
            use_orig_params=True,
            sync_module_states=False,
            limit_all_gathers=True,
            mixed_precision=mp,
            device_id=self.local_rank,
        )

        sig = inspect.signature(FSDP.__init__)
        if "ignored_states" in sig.parameters:
            fsdp_kwargs["ignored_states"] = set(lora_params)
        elif "ignored_parameters" in sig.parameters:
            fsdp_kwargs["ignored_parameters"] = set(lora_params)
        else:
            if self.accelerator.is_main_process:
                print("[WARN][FSDP] No ignored_states/ignored_parameters in this FSDP; LoRA params may be sharded.", flush=True)

        self.model = FSDP(base_model, **fsdp_kwargs)

        self.model_device = self.device
        _move_params_inplace(self.lora_params, self.device)

        self.opt = torch.optim.AdamW(
            self.lora_params,
            lr=args.lr,
            betas=(0.9, 0.98),
            weight_decay=args.weight_decay,
        )

        self.ds = ChatMLJsonl(args.dataset, rank=self.rank, world_size=self.world_size)
        self.loader = DataLoader(
            self.ds,
            batch_size=args.batch_size,
            drop_last=True,
        )

        self.pad_token_id = (
            self.tok.eos_token_id
            if self.tok.eos_token_id is not None
            else self.tok.pad_token_id
        )

        # ---- caches ----
        if getattr(args, "reward_cache", None):
            base = args.reward_cache
            root, ext = os.path.splitext(base)
            ext = ext if ext else ".jsonl"
            args.reward_cache = f"{root}.rank{self.rank}{ext}"
        self.cache = ScoreCache(args.reward_cache)

        if getattr(args, "smart_reward_cache", None):
            base = args.smart_reward_cache
            root, ext = os.path.splitext(base)
            ext = ext if ext else ".jsonl"
            args.smart_reward_cache = f"{root}.rank{self.rank}{ext}"
        self.smart_cache = ScoreCache(args.smart_reward_cache)

        # smart enabled?
        self.use_smart_reward = (
            float(getattr(self.args, "smart_reward_weight", 0.0) or 0.0) != 0.0
            or float(getattr(self.args, "smart_hack_penalty", 0.0) or 0.0) != 0.0
        )

        # vLLM
        self.use_vllm = bool(getattr(args, "use_vllm", False))
        self.vllm_clients: List[VLLMOpenAIClient] = []
        self.vllm_enabled: List[bool] = []
        self.vllm_sampling_model: str = ""
        if self.use_vllm:
            self._init_vllm_clients()

        self.global_step = 0
        self.baseline: Optional[float] = None

        # ---- NaN/Inf tracking for early abort ----
        self.consecutive_nan_count: int = 0
        self.max_consecutive_nan: int = int(getattr(self.args, "max_consecutive_nan", 10) or 10)

        # ---- progress / ETA tracking ----
        self.total_steps: int = int(getattr(self.args, "steps", 0) or 0)
        self.train_wall_start: Optional[float] = None
        self.step_time_ema: Optional[float] = None
        self.step_time_ema_alpha: float = 0.1  # ETA smoothing

        # ---- epoch progress tracking ----
        self.epoch_total: int = int(getattr(self.args, "epochs", 0) or 0)
        self.epoch_idx: int = 0
        self.steps_per_epoch: int = 0
        self.step_in_epoch: int = 0

        if self.args.wandb and self.accelerator.is_main_process:
            # Auto-generate wandb run name if not specified
            if self.args.wandb_run_name is None:
                reward_tag = str(getattr(self.args, "reward_mode", "grpo") or "grpo").lower()
                self.args.wandb_run_name = (
                    f"{reward_tag}_lr{self.args.lr}_r{self.args.lora_r}_gs{self.args.grpo_group_size}"
                )
            try:
                import wandb
                self._wandb_mod = wandb
                self.wandb_run = wandb.init(
                    project=self.args.wandb_project,
                    entity=self.args.wandb_entity or None,
                    name=self.args.wandb_run_name,
                    config=vars(self.args),
                )
            except Exception as e:
                print(f"[WARN] wandb init failed: {e}", flush=True)
                self.wandb_run = None
                self._wandb_mod = None

        if self.accelerator.is_main_process:
            print("========== GRPO w/ LLM-as-judge (LoRA/PEFT) ==========")
            print(f"actor base model        : {self.args.model}")
            print(f"dataset                 : {self.args.dataset}")
            print(f"grpo_group_size         : {self.args.grpo_group_size}")
            print(f"use_vllm (sampling)     : {self.args.use_vllm}")
            print(f"vllm_base_urls          : {self.args.vllm_base_urls or self.args.vllm_base_url}")
            print(f"vllm_model_name(base)   : {self.args.vllm_model_name}")
            print(f"vllm_lora_name          : {self.args.vllm_lora_name}")
            print(f"vllm_reload_max_retries : {self.args.vllm_reload_max_retries}")
            print(f"save_dir                : {self.args.save_dir}")
            print(f"lora (r/alpha/dropout)  : {self.args.lora_r}/{self.args.lora_alpha}/{self.args.lora_dropout}")
            print(f"lora_target_modules     : {self.args.lora_target_modules}")
            print(f"debug_print             : {self.args.debug_print}")
            print(f"debug_barrier           : {self.args.debug_barrier}")
            print(f"centralize_actor_judge  : {getattr(self.args, 'centralize_actor_judge', False)}")
            print(f"judge_parallel_workers  : {getattr(self.args, 'judge_parallel_workers', 0)}")
            print(f"wandb_table_rows        : {getattr(self.args, 'wandb_table_rows', 4)}")
            print(f"wandb_table_text_max    : {getattr(self.args, 'wandb_table_text_max_chars', 2000)}")
            print(f"use_smart_reward        : {self.use_smart_reward}")
            print(f"smart_reward_weight     : {getattr(self.args,'smart_reward_weight',0.0)}")
            print(f"smart_gate_mode         : {getattr(self.args,'smart_gate_mode','soft')}")
            print(f"smart_gate_threshold    : {getattr(self.args,'smart_gate_threshold',0.5)}")
            print(f"smart_reward_cap        : {getattr(self.args,'smart_reward_cap',0.2)}")
            print(f"smart_hack_penalty      : {getattr(self.args,'smart_hack_penalty',0.0)}")
            print(f"reward_mode             : {getattr(self.args,'reward_mode','grpo')}")
            print(f"grpo_clip_eps           : {getattr(self.args,'grpo_clip_eps',0.2)}")
            print(f"grpo_logratio_clip      : {getattr(self.args,'grpo_logratio_clip',20.0)}")
            print(f"grpo_epochs             : {getattr(self.args,'grpo_epochs',1)}")
            print(f"gdpo_eps                : {getattr(self.args,'gdpo_eps',1e-6)}")
            print(f"gdpo_whiten_global      : {getattr(self.args,'gdpo_whiten_global',True)}")
            print(f"gspo_clip_eps_low       : {getattr(self.args,'gspo_clip_eps_low',3e-4)}")
            print(f"gspo_clip_eps_high      : {getattr(self.args,'gspo_clip_eps_high',4e-4)}")
            print(f"gspo_logratio_clip      : {getattr(self.args,'gspo_logratio_clip',20.0)}")
            print(f"gspo_epochs             : {getattr(self.args,'gspo_epochs',4)}")
            print(f"kl_beta                 : {getattr(self.args,'kl_beta',0.0)}")
            print(f"kl_clamp_max            : {getattr(self.args,'kl_clamp_max',10.0)}")
            print(f"kl_x_clip               : {getattr(self.args,'kl_x_clip',20.0)}")
            print(f"kl_clamp_min0           : {getattr(self.args,'kl_clamp_min0',True)}")
            print("=====================================================", flush=True)

        # Always reload at start when using vLLM (on-policy requirement)
        if self.use_vllm:
            self._maybe_save_and_reload_vllm(current_step=0, force=True)

    @staticmethod
    def _fmt_hms(seconds: float) -> str:
        s = int(max(0, round(seconds)))
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:02d}"

    # ===== 全ランク集計 mean/std =====
    def _global_mean_std(self, x: torch.Tensor) -> Tuple[float, float, int]:
        x = x.detach()
        n_local = int(x.numel())
        if self.world_size <= 1 or (not dist.is_available()) or (not dist.is_initialized()):
            if n_local <= 0:
                return 0.0, 0.0, 0
            mean = float(x.mean().item())
            std = float(x.std().item())
            return mean, std, n_local

        x64 = x.to(dtype=torch.float64)
        s_local = x64.sum() if n_local > 0 else torch.zeros((), device=x.device, dtype=torch.float64)
        ss_local = (x64 * x64).sum() if n_local > 0 else torch.zeros((), device=x.device, dtype=torch.float64)
        n_local_t = torch.tensor(float(n_local), device=x.device, dtype=torch.float64)

        stats = torch.stack([s_local, ss_local, n_local_t], dim=0)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        s = float(stats[0].item())
        ss = float(stats[1].item())
        n = float(stats[2].item())
        if n <= 0.0:
            return 0.0, 0.0, 0

        mean = s / n
        if n > 1.0:
            var = (ss - (s * s) / n) / (n - 1.0)
            if var < 0.0:
                var = 0.0
            std = var ** 0.5
        else:
            std = 0.0
        return float(mean), float(std), int(n)

    def _clip_text(self, s: Any) -> str:
        if s is None:
            return ""
        txt = str(s)
        mx = int(getattr(self.args, "wandb_table_text_max_chars", 2000) or 2000)
        if mx > 0 and len(txt) > mx:
            return txt[: max(0, mx - 3)] + "..."
        return txt

    def _clip_text_jsonl(self, s: Any) -> str:
        if s is None:
            return ""
        txt = str(s)
        mx = int(self.samples_jsonl_text_max_chars or 0)
        if mx > 0 and len(txt) > mx:
            return txt[: max(0, mx - 3)] + "..."
        return txt

    def _wandb_log_samples_table(
        self,
        step: int,
        prompts: List[str],
        refs: List[str],
        candidate_groups: List[List[str]],
        final_scores_group: torch.Tensor,
        med_scores_group: torch.Tensor,
        smart_scores_group: torch.Tensor,
        med_judge_outputs_group: List[List[str]],
        smart_judge_outputs_group: List[List[str]],
    ) -> None:
        if self.wandb_run is None or self._wandb_mod is None:
            return
        if not self.accelerator.is_main_process:
            return

        wandb = self._wandb_mod
        group_size = int(self.args.grpo_group_size)

        cols = ["step", "問題(prompt)", "答え(reference)"]
        for k in range(group_size):
            cols += [
                f"アクター生成_{k}",
                f"正解スコア_{k}",
                f"SMARTスコア_{k}",
                f"最終スコア_{k}",
                f"正解評価生成_{k}",
                f"SMART評価生成_{k}",
            ]

        table = wandb.Table(columns=cols)
        n_rows = min(len(prompts), int(getattr(self.args, "wandb_table_rows", 4) or 4))

        for i in range(n_rows):
            row: List[Any] = [int(step), self._clip_text(prompts[i]), self._clip_text(refs[i])]
            for k in range(group_size):
                cand = ""
                med_jo = ""
                smart_jo = ""
                med_sc = 0.0
                smart_sc = 0.0
                final_sc = 0.0
                try:
                    cand = candidate_groups[i][k]
                except Exception:
                    cand = ""
                try:
                    med_jo = med_judge_outputs_group[i][k]
                except Exception:
                    med_jo = ""
                try:
                    smart_jo = smart_judge_outputs_group[i][k]
                except Exception:
                    smart_jo = ""
                try:
                    med_sc = float(med_scores_group[i, k].item())
                except Exception:
                    med_sc = 0.0
                try:
                    smart_sc = float(smart_scores_group[i, k].item())
                except Exception:
                    smart_sc = 0.0
                try:
                    final_sc = float(final_scores_group[i, k].item())
                except Exception:
                    final_sc = 0.0

                row += [
                    self._clip_text(cand),
                    med_sc,
                    smart_sc,
                    final_sc,
                    self._clip_text(med_jo),
                    self._clip_text(smart_jo),
                ]
            table.add_data(*row)

        self.wandb_run.log({"samples/table": table}, step=int(step))

    def _comm_group(self):
        return self.cpu_group if self.cpu_group is not None else None

    def _write_samples_jsonl(
        self,
        step: int,
        epoch: int,
        prompts: List[str],
        refs: List[str],
        candidate_groups: List[List[str]],
        med_judge_outputs_group: List[List[str]],
        smart_judge_outputs_group: List[List[str]],
        final_scores_group: torch.Tensor,
        med_scores_group: torch.Tensor,
        smart_scores_group: torch.Tensor,
    ) -> None:
        if self.samples_jsonl_every <= 0:
            return
        if step % self.samples_jsonl_every != 0:
            return

        group_size = int(self.args.grpo_group_size)
        B = len(prompts)

        # ===== centralize_actor_judge 時は rank0 が _centralized_generate_and_score 内で全サンプルを書くので skip =====
        # この判定は all_gather_object より前に置く必要がある（全 rank で一貫して return）
        if bool(getattr(self.args, "centralize_actor_judge", False)):
            return

        # B <= 0 の場合も全 rank で一貫して処理（all_gather_object を呼ぶため）
        local_recs: List[Dict[str, Any]] = []
        now_iso = datetime.now().isoformat(timespec="seconds")

        # local_recs 構築中の例外は catch して空リストで続行（all_gather_object デッドロック防止）
        try:
            for i in range(B):
                p = prompts[i]
                r = refs[i] if i < len(refs) else ""

                h = hashlib.sha1()
                h.update(str(step).encode("utf-8"))
                h.update(b"\x00")
                h.update(str(self.rank).encode("utf-8"))
                h.update(b"\x00")
                h.update(str(i).encode("utf-8"))
                h.update(b"\x00")
                h.update(p.encode("utf-8", errors="ignore"))
                group_id = h.hexdigest()[:16]

                g_list: List[Dict[str, Any]] = []
                for k in range(group_size):
                    cand = ""
                    med_jo = ""
                    smart_jo = ""
                    med_sc = 0.0
                    smart_sc = 0.0
                    final_sc = 0.0
                    try:
                        cand = candidate_groups[i][k]
                    except Exception:
                        cand = ""
                    try:
                        med_jo = med_judge_outputs_group[i][k]
                    except Exception:
                        med_jo = ""
                    try:
                        smart_jo = smart_judge_outputs_group[i][k]
                    except Exception:
                        smart_jo = ""
                    try:
                        med_sc = float(med_scores_group[i, k].item())
                    except Exception:
                        med_sc = 0.0
                    try:
                        smart_sc = float(smart_scores_group[i, k].item())
                    except Exception:
                        smart_sc = 0.0
                    try:
                        final_sc = float(final_scores_group[i, k].item())
                    except Exception:
                        final_sc = 0.0

                    g_list.append(
                        {
                            "k": int(k),
                            "actor": self._clip_text_jsonl(cand),
                            "score": float(final_sc),          # backward compatible: "score" = final
                            "med_score": float(med_sc),
                            "smart_score": float(smart_sc),
                            "judge_med": self._clip_text_jsonl(med_jo),
                            "judge_smart": self._clip_text_jsonl(smart_jo),
                        }
                    )

                local_recs.append(
                    {
                        "ts": now_iso,
                        "epoch": int(epoch),
                        "step": int(step),
                        "rank": int(self.rank),
                        "local_batch_index": int(i),
                        "group_size": int(group_size),
                        "group_id": group_id,
                        "prompt": self._clip_text_jsonl(p),
                        "reference": self._clip_text_jsonl(r),
                        "group": g_list,
                    }
                )
        except Exception as e:
            print(f"[rank{self.rank}][WARN][SAMPLES] error building local_recs: {e}", flush=True)
            local_recs = []  # 空リストで all_gather_object に参加

        if self.world_size <= 1 or (not dist.is_available()) or (not dist.is_initialized()):
            if self.accelerator.is_main_process:
                os.makedirs(os.path.dirname(self.samples_jsonl_path), exist_ok=True)
                with open(self.samples_jsonl_path, "a", encoding="utf-8") as f:
                    for rec in local_recs:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return

        # ===== centralize_actor_judge 時は rank0 が _centralized_generate_and_score 内で全サンプルを書くので skip =====
        if bool(getattr(self.args, "centralize_actor_judge", False)):
            return

        comm = self._comm_group()
        if comm is None:
            # gloo group がない場合は rank0 だけでローカル書き込み
            if self.accelerator.is_main_process:
                print(f"[WARN][SAMPLES] cpu_group is None, writing only rank0 local records", flush=True)
                os.makedirs(os.path.dirname(self.samples_jsonl_path), exist_ok=True)
                with open(self.samples_jsonl_path, "a", encoding="utf-8") as f:
                    for rec in local_recs:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return

        gathered: List[Any] = [None] * int(self.world_size)
        dist.all_gather_object(gathered, local_recs, group=comm)

        if self.rank == 0:
            os.makedirs(os.path.dirname(self.samples_jsonl_path), exist_ok=True)
            with open(self.samples_jsonl_path, "a", encoding="utf-8") as f:
                for part in gathered:
                    if not part:
                        continue
                    if isinstance(part, list):
                        for rec in part:
                            if isinstance(rec, dict):
                                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _write_samples_jsonl_centralized(
        self,
        step: int,
        epoch: int,
        prompts_all: List[str],
        refs_all: List[str],
        candidate_groups_all: List[List[str]],
        med_judge_outputs_all: List[List[str]],
        smart_judge_outputs_all: List[List[str]],
        med_scores_all: torch.Tensor,
        smart_scores_all: torch.Tensor,
        per_rank_B: List[int],
    ) -> None:
        """
        centralize モード専用: rank0 が scatter 前に全サンプルを書く（通信ゼロ）
        """
        if self.samples_jsonl_every <= 0:
            return
        if step % self.samples_jsonl_every != 0:
            return
        if self.rank != 0:
            return

        group_size = int(self.args.grpo_group_size)
        total_B = len(prompts_all)
        if total_B <= 0:
            return

        # final scores を計算
        med_flat = med_scores_all.view(-1).to(dtype=torch.float32)
        smart_flat = smart_scores_all.view(-1).to(dtype=torch.float32)
        final_flat, _, _ = self._combine_med_smart(med_flat, smart_flat)
        final_scores_all = final_flat.view(total_B, group_size)

        now_iso = datetime.now().isoformat(timespec="seconds")
        os.makedirs(os.path.dirname(self.samples_jsonl_path), exist_ok=True)

        ptr = 0
        with open(self.samples_jsonl_path, "a", encoding="utf-8") as f:
            for rr, b in enumerate(per_rank_B):
                for local_i in range(b):
                    global_i = ptr + local_i
                    p = prompts_all[global_i]
                    r = refs_all[global_i] if global_i < len(refs_all) else ""

                    h = hashlib.sha1()
                    h.update(str(step).encode("utf-8"))
                    h.update(b"\x00")
                    h.update(str(rr).encode("utf-8"))  # original rank
                    h.update(b"\x00")
                    h.update(str(local_i).encode("utf-8"))
                    h.update(b"\x00")
                    h.update(p.encode("utf-8", errors="ignore"))
                    group_id = h.hexdigest()[:16]

                    g_list: List[Dict[str, Any]] = []
                    for k in range(group_size):
                        cand = ""
                        med_jo = ""
                        smart_jo = ""
                        med_sc = 0.0
                        smart_sc = 0.0
                        final_sc = 0.0
                        try:
                            cand = candidate_groups_all[global_i][k]
                        except Exception:
                            cand = ""
                        try:
                            med_jo = med_judge_outputs_all[global_i][k]
                        except Exception:
                            med_jo = ""
                        try:
                            smart_jo = smart_judge_outputs_all[global_i][k]
                        except Exception:
                            smart_jo = ""
                        try:
                            med_sc = float(med_scores_all[global_i, k].item())
                        except Exception:
                            med_sc = 0.0
                        try:
                            smart_sc = float(smart_scores_all[global_i, k].item())
                        except Exception:
                            smart_sc = 0.0
                        try:
                            final_sc = float(final_scores_all[global_i, k].item())
                        except Exception:
                            final_sc = 0.0

                        g_list.append(
                            {
                                "k": int(k),
                                "actor": self._clip_text_jsonl(cand),
                                "score": float(final_sc),
                                "med_score": float(med_sc),
                                "smart_score": float(smart_sc),
                                "judge_med": self._clip_text_jsonl(med_jo),
                                "judge_smart": self._clip_text_jsonl(smart_jo),
                            }
                        )

                    rec = {
                        "ts": now_iso,
                        "epoch": int(epoch),
                        "step": int(step),
                        "rank": int(rr),  # original rank
                        "local_batch_index": int(local_i),
                        "group_size": int(group_size),
                        "group_id": group_id,
                        "prompt": self._clip_text_jsonl(p),
                        "reference": self._clip_text_jsonl(r),
                        "group": g_list,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                ptr += b

    # ===== debug phase logs =====
    def _dbg_enabled(self) -> bool:
        return bool(getattr(self.args, "debug_print", False))

    def _dbg_sync(self) -> bool:
        return bool(getattr(self.args, "sync_debug", False))

    def _phase_start(self, name: str) -> float:
        if self._dbg_enabled():
            print(f"[rank{self.rank}] {name} start", flush=True)
        return time.time()

    def _phase_end(self, name: str, t0: float):
        if not (self._dbg_enabled() or self._dbg_sync()):
            return
        if self._dbg_sync():
            torch.cuda.synchronize(self.device)
        dt = time.time() - t0
        if self._dbg_enabled():
            print(f"[rank{self.rank}] {name} done in {dt:.3f}s", flush=True)

    def _pre_fwd_barrier(self):
        if not bool(getattr(self.args, "debug_barrier", False)):
            return
        t0 = time.time()
        print(f"[rank{self.rank}] pre_fwd_barrier start", flush=True)
        self.accelerator.wait_for_everyone()
        dt = time.time() - t0
        print(f"[rank{self.rank}] pre_fwd_barrier done in {dt:.3f}s", flush=True)

    def _init_vllm_clients(self):
        timeout = int(self.args.vllm_connection_timeout)
        api_key = self.args.vllm_api_key or os.getenv("VLLM_API_KEY")

        urls: List[str] = []
        if getattr(self.args, "vllm_base_urls", None):
            urls = [u.strip() for u in self.args.vllm_base_urls.split(",") if u.strip()]
        elif self.args.vllm_base_url:
            urls = [self.args.vllm_base_url.strip()]
        else:
            urls = ["http://localhost:8000"]

        self.vllm_clients = [VLLMOpenAIClient(u, api_key=api_key, timeout=timeout) for u in urls]
        self.vllm_enabled = [True for _ in self.vllm_clients]
        self.vllm_sampling_model = str(self.args.vllm_lora_name) if self.args.vllm_lora_name else str(self.args.vllm_model_name)

        if self.accelerator.is_main_process:
            print(f"[vLLM] ready clients={len(self.vllm_clients)} sampling_model='{self.vllm_sampling_model}'", flush=True)

    def _adapter_dir_for_step(self, step: int) -> str:
        root = self.args.vllm_lora_save_root or os.path.join(self.args.save_dir, "adapters")
        name = str(self.args.vllm_lora_name)
        return os.path.join(root, name, f"step_{step:08d}")

    @torch.no_grad()
    def save_lora_adapter_pretrained(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)

        peft_model = self.model.module if isinstance(self.model, FSDP) else self.model
        if not hasattr(peft_model, "save_pretrained"):
            raise RuntimeError("PEFT model does not have save_pretrained(). Is peft installed and applied?")

        total_elems = 0
        for _, p in self.lora_named_params:
            total_elems += p.numel()
        if self.accelerator.is_main_process:
            approx_gb = (total_elems * 2) / (1024**3)
            print(f"[SAVE][LoRA] tensors={len(self.lora_named_params)} elems={total_elems:,} (~{approx_gb:.2f} GiB bf16)", flush=True)

        adapter_sd = {}
        for name, p in self.lora_named_params:
            adapter_sd[name] = p.detach().to("cpu", non_blocking=True)

        peft_model.save_pretrained(
            out_dir,
            state_dict=adapter_sd,
            safe_serialization=True,
        )

        latest_path = os.path.join(os.path.dirname(out_dir), "LATEST_STEP.txt")
        with open(latest_path, "w", encoding="utf-8") as f:
            f.write(str(out_dir) + "\n")

    # ===== vLLM reload helpers =====
    def _sleep_backoff(self, attempt: int):
        base = float(getattr(self.args, "vllm_reload_retry_backoff", 2.0))
        jitter = float(getattr(self.args, "vllm_reload_retry_jitter", 0.2))
        if attempt <= 1:
            return
        wait = base * (2 ** (attempt - 2))
        wait = wait * (1.0 + random.random() * jitter)
        time.sleep(wait)

    def _reload_one_server_with_retry(self, idx: int, lora_name: str, lora_path: str) -> bool:
        c = self.vllm_clients[idx]
        max_tries = int(getattr(self.args, "vllm_reload_max_retries", 5))

        last_err: Optional[Exception] = None
        for attempt in range(1, max_tries + 1):
            self._sleep_backoff(attempt)
            try:
                try:
                    c.unload_lora_adapter(lora_name)
                except Exception:
                    pass

                c.load_lora_adapter(lora_name, lora_path)
                if self.accelerator.is_main_process:
                    print(f"[vLLM][RELOAD][OK] server#{idx} base_url={c.base_url} attempt={attempt}", flush=True)
                return True
            except Exception as e:
                last_err = e
                if self.accelerator.is_main_process:
                    print(
                        f"[vLLM][RELOAD][RETRY] server#{idx} base_url={c.base_url} attempt={attempt}/{max_tries} err={e}",
                        flush=True,
                    )

        if self.accelerator.is_main_process:
            print(f"[vLLM][RELOAD][FAIL] server#{idx} base_url={c.base_url} err={last_err}", flush=True)
        return False

    def _reload_vllm_adapter_all_servers(self, lora_name: str, lora_path: str):
        enabled_idxs = [i for i, en in enumerate(self.vllm_enabled) if en]
        if not enabled_idxs:
            return

        failed: List[int] = []
        ok_count = 0

        max_workers = min(32, len(enabled_idxs))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_map = {
                ex.submit(self._reload_one_server_with_retry, i, lora_name, lora_path): i
                for i in enabled_idxs
            }
            for fut in as_completed(fut_map):
                i = fut_map[fut]
                try:
                    ok = bool(fut.result())
                except Exception:
                    ok = False
                if ok:
                    ok_count += 1
                else:
                    failed.append(i)

        if failed:
            for i in failed:
                self.vllm_enabled[i] = False
            if self.accelerator.is_main_process:
                bad_urls = [self.vllm_clients[i].base_url for i in failed]
                print(f"[WARN][vLLM] disabled {len(failed)} servers due to reload failure: {bad_urls}", flush=True)

        if self.accelerator.is_main_process:
            enabled_cnt = sum(1 for x in self.vllm_enabled if x)
            print(f"[vLLM][RELOAD] done ok={ok_count}/{len(enabled_idxs)} enabled_servers={enabled_cnt}/{len(self.vllm_clients)}", flush=True)

    def _maybe_save_and_reload_vllm(self, current_step: int, force: bool = False):
        """Save and reload vLLM adapter every step (required for on-policy training)."""
        if not self.use_vllm:
            return

        # Always reload every step (on-policy requirement)
        # force=True is used for initial reload at step=0
        if not force and current_step == 0:
            return

        if self.accelerator.is_main_process:
            print(f"[vLLM][RELOAD] enter barrier(0) step={current_step}", flush=True)
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            print(f"[vLLM][RELOAD] passed barrier(0) step={current_step}", flush=True)

        if self.accelerator.is_main_process:
            out_dir = self._adapter_dir_for_step(current_step)
            t0 = time.time()
            print(f"[vLLM][RELOAD] saving adapter -> {out_dir}", flush=True)
            self.save_lora_adapter_pretrained(out_dir)
            t1 = time.time()
            print(f"[vLLM][RELOAD] adapter saved ({t1 - t0:.1f}s). now reloading vLLM...", flush=True)

            self._reload_vllm_adapter_all_servers(str(self.args.vllm_lora_name), out_dir)

            enabled_cnt = sum(1 for x in self.vllm_enabled if x)
            if enabled_cnt == 0:
                raise RuntimeError("[vLLM] All servers disabled (reload failed everywhere). Abort.")

            t2 = time.time()
            print(f"[vLLM][RELOAD] vLLM reload done ({t2 - t1:.1f}s). enabled_servers={enabled_cnt}/{len(self.vllm_clients)}", flush=True)

        if self.accelerator.is_main_process:
            print(f"[vLLM][RELOAD] enter barrier(1) step={current_step}", flush=True)
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            print(f"[vLLM][RELOAD] passed barrier(1) step={current_step}", flush=True)

    def _sync_lora_grads(self, trainable_params: List[torch.nn.Parameter]):
        if self.world_size <= 1:
            return
        if not dist.is_available() or not dist.is_initialized():
            return
        none_grad_count = 0
        for p in trainable_params:
            if p.grad is None:
                # grad が None の場合、ゼロで初期化して all_reduce に参加
                # （一部 rank だけ None だとデッドロックするため）
                p.grad = torch.zeros_like(p.data)
                none_grad_count += 1
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(float(self.world_size))
        if none_grad_count > 0:
            print(f"[rank{self.rank}][WARN][SYNC_GRAD] {none_grad_count}/{len(trainable_params)} params had None grad (zero-initialized)", flush=True)

    def _get_thread_judge_session(self) -> requests.Session:
        s = getattr(self._judge_tls, "sess", None)
        if s is None:
            s = _make_retrying_session(
                total_retries=int(getattr(self.args, "judge_http_retries", 8)),
                backoff_factor=float(getattr(self.args, "judge_http_backoff", 0.5)),
            )
            self._judge_tls.sess = s
        return s

    # ===== External judge call (generic) =====
    def _call_external_chat(self, system_text: str, user_text: str, *, backend: str, api_base: Optional[str],
                           model_name: Optional[str], api_key: Optional[str],
                           max_tokens: int, temperature: float, max_attempts: int,
                           fallback_model_name: Optional[str] = None) -> str:
        """
        Wrapper with fallback support.
        backend can be:
          - "openrouter", "openai", "cerebras" (single)
          - "cerebras+openrouter" (try cerebras first, fallback to openrouter)
          - "openrouter+cerebras" (try openrouter first, fallback to cerebras)
        """
        # Parse fallback chain
        if "+" in backend:
            backends = [b.strip() for b in backend.split("+") if b.strip()]
        else:
            backends = [backend]

        last_result = ""
        for i, be in enumerate(backends):
            # Determine api_base and api_key for this backend
            be_api_base = api_base  # user override takes precedence
            be_api_key = api_key
            be_model = model_name

            # If using fallback, get fallback-specific settings
            if i > 0:
                # For fallback backend, use fallback model name if provided
                be_api_base = None  # use default for fallback backend
                be_api_key = None   # will be resolved in _call_external_chat_single
                if fallback_model_name:
                    be_model = fallback_model_name

            result = self._call_external_chat_single(
                system_text, user_text,
                backend=be,
                api_base=be_api_base,
                model_name=be_model,
                api_key=be_api_key,
                max_tokens=max_tokens,
                temperature=temperature,
                max_attempts=max_attempts if i == 0 else max(1, max_attempts // 2),  # fewer retries for fallback
            )
            if result:
                if i > 0 and self.accelerator.is_main_process:
                    print(f"[INFO] Fallback to {be} (model={be_model}) succeeded", flush=True)
                return result
            last_result = result

            if i < len(backends) - 1 and self.accelerator.is_main_process:
                print(f"[WARN] Primary backend {be} failed, trying fallback {backends[i+1]}", flush=True)

        return last_result

    def _call_external_chat_single(self, system_text: str, user_text: str, *, backend: str, api_base: Optional[str],
                                   model_name: Optional[str], api_key: Optional[str],
                                   max_tokens: int, temperature: float, max_attempts: int) -> str:
        """Single backend call without fallback."""
        if backend not in ("openrouter", "openai", "cerebras"):
            raise ValueError(f"Unknown external backend: {backend}")
        if api_base is None:
            if backend == "openrouter":
                api_base = "https://openrouter.ai/api"
            elif backend == "cerebras":
                api_base = "https://api.cerebras.ai"
            else:
                api_base = "http://localhost:8000"

        if api_key is None:
            if backend == "openrouter":
                api_key = os.getenv("OPENROUTER_API_KEY")
            elif backend == "cerebras":
                api_key = os.getenv("CEREBRAS_API_KEY")
            if api_key is None:
                api_key = os.getenv("JUDGE_API_KEY")

        if api_key is None or not model_name:
            if self.accelerator.is_main_process:
                print(f"[WARN] external API key or model name missing for {backend}; skipping.", flush=True)
            return ""

        url = api_base.rstrip("/") + "/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "lora_grpo/1.0",
        }
        if backend == "openrouter":
            headers.setdefault("HTTP-Referer", os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"))
            headers.setdefault("X-Title", os.getenv("OPENROUTER_X_TITLE", "lora_grpo"))

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            "temperature": float(temperature),
        }
        # Cerebras uses max_completion_tokens; others use max_tokens
        if backend == "cerebras":
            payload["max_completion_tokens"] = int(max_tokens)
        else:
            payload["max_tokens"] = int(max_tokens)

        connect_to = float(getattr(self.args, "judge_connect_timeout", 10.0))
        read_to = float(getattr(self.args, "judge_read_timeout", 60.0))
        timeout = (connect_to, read_to)

        base_sleep = float(getattr(self.args, "judge_retry_sleep", 1.0))
        max_sleep = float(getattr(self.args, "judge_retry_sleep_max", 60.0))

        last_err: Optional[str] = None
        for attempt in range(1, int(max_attempts) + 1):
            try:
                sess = self._get_thread_judge_session()
                resp = sess.post(url, headers=headers, json=payload, timeout=timeout)

                # 400/401/403/404: client error, no retry, immediate fail (fallback)
                if resp.status_code in (400, 401, 403, 404):
                    body_snip = (resp.text or "")[:400].replace("\n", "\\n")
                    if self.accelerator.is_main_process:
                        print(f"[WARN] external HTTP {resp.status_code} (no retry, fail to fallback): '{body_snip}'", flush=True)
                    return ""  # immediate fail, trigger fallback

                # 429/5xx: transient error, retry with backoff
                if resp.status_code in (429, 500, 502, 503, 504):
                    ra = resp.headers.get("Retry-After")
                    if ra is not None:
                        try:
                            wait = float(ra)
                        except Exception:
                            wait = None
                    else:
                        wait = None

                    if wait is None:
                        wait = min(max_sleep, base_sleep * (2 ** max(0, attempt - 1)))
                        wait = wait * (1.0 + random.random() * 0.2)

                    if self.accelerator.is_main_process:
                        body_snip = (resp.text or "")[:400].replace("\n", "\\n")
                        print(f"[WARN] external HTTP {resp.status_code} attempt {attempt}/{max_attempts} wait={wait:.1f}s body='{body_snip}'", flush=True)
                    time.sleep(wait)
                    continue

                resp.raise_for_status()

                try:
                    data = resp.json()
                except Exception:
                    body_snip = (resp.text or "")[:800].replace("\n", "\\n")
                    raise RuntimeError(f"Non-JSON response: '{body_snip}'")

                try:
                    return data["choices"][0]["message"]["content"]
                except Exception:
                    if isinstance(data, dict) and "error" in data:
                        raise RuntimeError(f"API error: {data['error']}")
                    raise RuntimeError(f"Unexpected response keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")

            except Exception as e:
                last_err = str(e)
                if self.accelerator.is_main_process:
                    user_snip = (user_text or "")[:150].replace("\n", "\\n")
                    print(f"[WARN][JUDGE_API] attempt {attempt}/{max_attempts} failed: {e}, model={model_name}, user='{user_snip}...'", flush=True)

                wait = min(max_sleep, base_sleep * (2 ** max(0, attempt - 1)))
                wait = wait * (1.0 + random.random() * 0.2)
                time.sleep(wait)

        if self.accelerator.is_main_process and last_err:
            print(f"[ERROR][JUDGE_API] giving up after {max_attempts} attempts: {last_err}", flush=True)
        return ""

    # ===== Medical judge =====
    @torch.no_grad()
    def _score_single(self, prompt: str, candidate: str, reference: Optional[str]) -> Tuple[float, str]:
        text = build_judge_text(prompt, candidate, reference)

        if self.args.judge_backend == "local":
            if hasattr(self.tok, "apply_chat_template") and getattr(self.tok, "chat_template", None):
                msgs = [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": text},
                ]
                enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
                if isinstance(enc, dict):
                    enc = {k: v.to(self.model_device) for k, v in enc.items()}
                else:
                    enc = {"input_ids": enc.to(self.model_device)}
            else:
                sys_head = f"[SYSTEM]\n{JUDGE_SYSTEM}\n\n"
                enc = self.tok(sys_head + text, return_tensors="pt")
                enc = {k: v.to(self.model_device) for k, v in enc.items()}

            self.model.eval()
            gen = self.model.generate(
                **enc,
                max_new_tokens=self.args.judge_max_new_tokens,
                do_sample=(self.args.judge_temperature > 0),
                temperature=self.args.judge_temperature if self.args.judge_temperature > 0 else None,
                top_p=0.95 if self.args.judge_temperature > 0 else None,
                use_cache=True,
                pad_token_id=self.pad_token_id,
                output_router_logits=False,
                synced_gpus=True,
            )
            self.model.train()
            gen_txt = self.tok.decode(gen[0], skip_special_tokens=True)
        else:
            gen_txt = self._call_external_chat(
                JUDGE_SYSTEM,
                text,
                backend=str(self.args.judge_backend),
                api_base=self.args.judge_api_base,
                model_name=self.args.judge_model_name,
                api_key=self.args.judge_api_key,
                max_tokens=int(self.args.judge_max_new_tokens),
                temperature=float(self.args.judge_temperature),
                max_attempts=int(self.args.judge_max_retries),
                fallback_model_name=getattr(self.args, "judge_fallback_model_name", None),
            )

        matches = re.findall(r"\[(0|0\.5|1)\]", gen_txt)
        if matches:
            raw_str = matches[-1]
            try:
                raw_val = float(raw_str)
            except Exception:
                raw_val = 0.0
        else:
            matches = re.findall(r"\b(0(?:\.5)?|1)\b", gen_txt)
            if matches:
                try:
                    raw_val = float(matches[-1])
                except Exception:
                    raw_val = 0.0
            else:
                raw_val = 0.0
                # スコアが見つからなかった場合のログ
                if self._dbg_enabled():
                    snippet = (gen_txt or "")[:200].replace("\n", "\\n")
                    print(f"[WARN][MED_JUDGE] score pattern not found in output: '{snippet}...'", flush=True)

        candidates = [0.0, 0.5, 1.0]
        val = min(candidates, key=lambda x: abs(x - raw_val))
        return float(val), gen_txt

    @torch.no_grad()
    def judge_scores_flat(
        self,
        prompts_flat: List[str],
        cands_flat: List[str],
        refs_flat: List[str],
    ) -> Tuple[torch.Tensor, List[str]]:
        backend = str(self.args.judge_backend)
        # Parallel for any external backend (including fallback like "cerebras+openrouter")
        want_parallel = backend != "local"

        workers = int(getattr(self.args, "judge_parallel_workers", 0) or 0)
        if workers <= 0 and want_parallel:
            workers = min(32, max(1, int(self.world_size) * 2))

        N = len(prompts_flat)
        t0 = time.time()
        if self.accelerator.is_main_process:
            print(f"[MED_JUDGE] start: {N} samples, backend={backend}, parallel={want_parallel}, workers={workers}", flush=True)

        error_count = 0

        if (not want_parallel) or workers <= 1 or len(prompts_flat) <= 1:
            scores: List[float] = []
            judge_outputs: List[str] = []
            for idx, (p, c, r) in enumerate(zip(prompts_flat, cands_flat, refs_flat)):
                k = self.cache.key(p, c, r, tag="med")
                cached_score = self.cache.get(k)
                if cached_score is not None:
                    s = float(cached_score)
                    jo = ""
                else:
                    try:
                        s, jo = self._score_single(p, c, r)
                        self.cache.put(k, s)
                    except Exception as e:
                        error_count += 1
                        if self.accelerator.is_main_process:
                            prompt_snip = (p or "")[:100].replace("\n", "\\n")
                            print(f"[ERROR][MED_JUDGE] idx={idx} failed: {e}, prompt='{prompt_snip}...'", flush=True)
                        s, jo = 0.0, ""
                scores.append(s)
                judge_outputs.append(jo)

            dt = time.time() - t0
            if self.accelerator.is_main_process:
                print(f"[MED_JUDGE] done: {N} samples in {dt:.2f}s ({N/max(dt,0.001):.1f} samples/s), errors={error_count}", flush=True)
            return torch.tensor(scores, dtype=torch.float32, device=self.model_device), judge_outputs

        scores: List[float] = [0.0] * N
        judge_outputs: List[str] = [""] * N

        tasks: List[Tuple[int, str, str, str, str]] = []
        cache_hits = 0
        for i, (p, c, r) in enumerate(zip(prompts_flat, cands_flat, refs_flat)):
            k = self.cache.key(p, c, r, tag="med")
            cached_score = self.cache.get(k)
            if cached_score is not None:
                scores[i] = float(cached_score)
                judge_outputs[i] = ""
                cache_hits += 1
            else:
                tasks.append((i, p, c, r, k))

        if self.accelerator.is_main_process and cache_hits > 0:
            print(f"[MED_JUDGE] cache hits: {cache_hits}/{N}", flush=True)

        def _worker(t: Tuple[int, str, str, str, str]):
            i, p, c, r, k = t
            try:
                s, jo = self._score_single(p, c, r)
                return i, k, float(s), jo, None
            except Exception as e:
                return i, k, 0.0, "", str(e)

        if tasks:
            max_workers = min(int(workers), max(1, len(tasks)))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_worker, t) for t in tasks]
                for fut in as_completed(futures):
                    i, k, s, jo, err = fut.result()
                    if err is not None:
                        error_count += 1
                        if self.accelerator.is_main_process:
                            print(f"[ERROR][MED_JUDGE] idx={i} failed: {err}", flush=True)
                    scores[i] = float(s)
                    judge_outputs[i] = jo
                    if err is None:
                        self.cache.put(k, float(s))

        dt = time.time() - t0
        if self.accelerator.is_main_process:
            print(f"[MED_JUDGE] done: {N} samples in {dt:.2f}s ({N/max(dt,0.001):.1f} samples/s), errors={error_count}", flush=True)

        return torch.tensor(scores, dtype=torch.float32, device=self.model_device), judge_outputs

    # ===== SMART judge =====
    @torch.no_grad()
    def _smart_score_single(self, prompt: str, candidate: str, reference: Optional[str]) -> Tuple[float, str]:
        text = build_smart_text(prompt, candidate, reference)
        backend = str(self.args.smart_backend)

        if backend == "local":
            if hasattr(self.tok, "apply_chat_template") and getattr(self.tok, "chat_template", None):
                msgs = [
                    {"role": "system", "content": SMART_SYSTEM},
                    {"role": "user", "content": text},
                ]
                enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
                if isinstance(enc, dict):
                    enc = {k: v.to(self.model_device) for k, v in enc.items()}
                else:
                    enc = {"input_ids": enc.to(self.model_device)}
            else:
                sys_head = f"[SYSTEM]\n{SMART_SYSTEM}\n\n"
                enc = self.tok(sys_head + text, return_tensors="pt")
                enc = {k: v.to(self.model_device) for k, v in enc.items()}

            self.model.eval()
            gen = self.model.generate(
                **enc,
                max_new_tokens=int(self.args.smart_max_new_tokens),
                do_sample=(float(self.args.smart_temperature) > 0),
                temperature=float(self.args.smart_temperature) if float(self.args.smart_temperature) > 0 else None,
                top_p=0.95 if float(self.args.smart_temperature) > 0 else None,
                use_cache=True,
                pad_token_id=self.pad_token_id,
                output_router_logits=False,
                synced_gpus=True,
            )
            self.model.train()
            gen_txt = self.tok.decode(gen[0], skip_special_tokens=True)
        else:
            gen_txt = self._call_external_chat(
                SMART_SYSTEM,
                text,
                backend=backend,
                api_base=self.args.smart_api_base,
                model_name=self.args.smart_model_name,
                api_key=self.args.smart_api_key,
                max_tokens=int(self.args.smart_max_new_tokens),
                temperature=float(self.args.smart_temperature),
                max_attempts=int(self.args.smart_max_retries),
                fallback_model_name=getattr(self.args, "smart_fallback_model_name", None),
            )

        # parse [0]..[10]
        m = re.findall(r"\[(10|[0-9])\]", gen_txt)
        if m:
            raw = m[-1]
            try:
                val = float(int(raw))
            except Exception:
                val = 0.0
        else:
            # fallback: last integer 0..10
            m2 = re.findall(r"\b(10|[0-9])\b", gen_txt)
            if m2:
                try:
                    val = float(int(m2[-1]))
                except Exception:
                    val = 0.0
            else:
                val = 0.0
                # スコアが見つからなかった場合のログ
                if self._dbg_enabled():
                    snippet = (gen_txt or "")[:200].replace("\n", "\\n")
                    print(f"[WARN][SMART_JUDGE] score pattern not found in output: '{snippet}...'", flush=True)

        val = float(max(0, min(10, int(round(val)))))
        return val, gen_txt

    @torch.no_grad()
    def smart_scores_flat(
        self,
        prompts_flat: List[str],
        cands_flat: List[str],
        refs_flat: List[str],
    ) -> Tuple[torch.Tensor, List[str]]:
        # if smart is disabled entirely, skip calls
        if not self.use_smart_reward:
            zeros = torch.zeros((len(prompts_flat),), dtype=torch.float32, device=self.model_device)
            return zeros, [""] * len(prompts_flat)

        backend = str(self.args.smart_backend)
        # Parallel for any external backend (including fallback like "cerebras+openrouter")
        want_parallel = backend != "local"

        workers = int(getattr(self.args, "smart_parallel_workers", 0) or 0)
        if workers <= 0 and want_parallel:
            workers = min(32, max(1, int(self.world_size) * 2))

        N = len(prompts_flat)
        t0 = time.time()
        if self.accelerator.is_main_process:
            print(f"[SMART_JUDGE] start: {N} samples, backend={backend}, parallel={want_parallel}, workers={workers}", flush=True)

        error_count = 0

        if (not want_parallel) or workers <= 1 or len(prompts_flat) <= 1:
            scores: List[float] = []
            judge_outputs: List[str] = []
            for idx, (p, c, r) in enumerate(zip(prompts_flat, cands_flat, refs_flat)):
                k = self.smart_cache.key(p, c, r, tag="smart")
                cached_score = self.smart_cache.get(k)
                if cached_score is not None:
                    s = float(cached_score)
                    jo = ""
                else:
                    try:
                        s, jo = self._smart_score_single(p, c, r)
                        self.smart_cache.put(k, s)
                    except Exception as e:
                        error_count += 1
                        if self.accelerator.is_main_process:
                            prompt_snip = (p or "")[:100].replace("\n", "\\n")
                            print(f"[ERROR][SMART_JUDGE] idx={idx} failed: {e}, prompt='{prompt_snip}...'", flush=True)
                        s, jo = 0.0, ""
                scores.append(s)
                judge_outputs.append(jo)

            dt = time.time() - t0
            if self.accelerator.is_main_process:
                print(f"[SMART_JUDGE] done: {N} samples in {dt:.2f}s ({N/max(dt,0.001):.1f} samples/s), errors={error_count}", flush=True)
            return torch.tensor(scores, dtype=torch.float32, device=self.model_device), judge_outputs

        scores: List[float] = [0.0] * N
        judge_outputs: List[str] = [""] * N

        tasks: List[Tuple[int, str, str, str, str]] = []
        cache_hits = 0
        for i, (p, c, r) in enumerate(zip(prompts_flat, cands_flat, refs_flat)):
            k = self.smart_cache.key(p, c, r, tag="smart")
            cached_score = self.smart_cache.get(k)
            if cached_score is not None:
                scores[i] = float(cached_score)
                judge_outputs[i] = ""
                cache_hits += 1
            else:
                tasks.append((i, p, c, r, k))

        if self.accelerator.is_main_process and cache_hits > 0:
            print(f"[SMART_JUDGE] cache hits: {cache_hits}/{N}", flush=True)

        def _worker(t: Tuple[int, str, str, str, str]):
            i, p, c, r, k = t
            try:
                s, jo = self._smart_score_single(p, c, r)
                return i, k, float(s), jo, None
            except Exception as e:
                return i, k, 0.0, "", str(e)

        if tasks:
            max_workers = min(int(workers), max(1, len(tasks)))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_worker, t) for t in tasks]
                for fut in as_completed(futures):
                    i, k, s, jo, err = fut.result()
                    if err is not None:
                        error_count += 1
                        if self.accelerator.is_main_process:
                            print(f"[ERROR][SMART_JUDGE] idx={i} failed: {err}", flush=True)
                    scores[i] = float(s)
                    judge_outputs[i] = jo
                    if err is None:
                        self.smart_cache.put(k, float(s))

        dt = time.time() - t0
        if self.accelerator.is_main_process:
            print(f"[SMART_JUDGE] done: {N} samples in {dt:.2f}s ({N/max(dt,0.001):.1f} samples/s), errors={error_count}", flush=True)

        return torch.tensor(scores, dtype=torch.float32, device=self.model_device), judge_outputs

    # ===== reward combine (anti-hack) =====
    def _combine_med_smart(
        self,
        med_scores_flat: torch.Tensor,   # (N,) in {0,0.5,1}
        smart_scores_flat: torch.Tensor, # (N,) in [0..10]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returns:
          final_scores_flat, smart_add_flat, hack_mask_flat
        """
        med = med_scores_flat.to(dtype=torch.float32)
        smart = smart_scores_flat.to(dtype=torch.float32)

        smart_max = float(getattr(self.args, "smart_max_score", 10.0) or 10.0)
        smart_norm = smart / max(1e-6, smart_max)

        smart_w = float(getattr(self.args, "smart_reward_weight", 0.0) or 0.0)
        smart_cap = float(getattr(self.args, "smart_reward_cap", 0.2) or 0.0)

        gate_mode = str(getattr(self.args, "smart_gate_mode", "soft") or "soft").lower()
        gate_thr = float(getattr(self.args, "smart_gate_threshold", 0.5) or 0.5)

        if smart_w == 0.0 and float(getattr(self.args, "smart_hack_penalty", 0.0) or 0.0) == 0.0:
            zeros = torch.zeros_like(med)
            return med, zeros, zeros

        if gate_mode == "hard":
            gate = (med >= gate_thr).float()
        elif gate_mode == "none":
            gate = torch.ones_like(med)
        else:
            # soft gate (recommended): proportional to medical score
            gate = med.detach()

        smart_add = smart_w * smart_norm * gate
        if smart_cap > 0:
            smart_add = smart_add.clamp(max=smart_cap)

        hack_pen = float(getattr(self.args, "smart_hack_penalty", 0.0) or 0.0)
        hack_med_le = float(getattr(self.args, "smart_hack_med_le", 0.0) or 0.0)
        hack_smart_ge = float(getattr(self.args, "smart_hack_smart_ge", 8.0) or 8.0)
        hack_mask = ((med <= hack_med_le) & (smart >= hack_smart_ge)).float()

        final = med + smart_add
        if hack_pen != 0.0:
            final = final - hack_pen * hack_mask

        return final, smart_add, hack_mask

    def _enabled_client_indices(self) -> List[int]:
        return [i for i, en in enumerate(self.vllm_enabled) if en]

    @torch.no_grad()
    def _generate_with_vllm_grpo(
        self,
        prompts: List[str],
        group_size: int,
    ) -> List[List[str]]:
        """
        Generate candidates via vLLM.
        Returns list of candidate groups (one group per prompt).
        """
        assert self.vllm_clients, "vLLM clients not initialized"
        enabled_idxs = self._enabled_client_indices()
        if not enabled_idxs:
            raise RuntimeError("[vLLM] No enabled clients available for sampling.")

        clients = self.vllm_clients
        K = len(enabled_idxs)

        orig_B = len(prompts)
        force_n1 = bool(getattr(self.args, "vllm_force_n1", False))

        call_prompts = prompts
        call_n = int(group_size)

        if force_n1:
            expanded: List[str] = []
            for p in prompts:
                expanded.extend([p] * int(group_size))
            call_prompts = expanded
            call_n = 1

        if hasattr(self.tok, "apply_chat_template") and getattr(self.tok, "chat_template", None):
            call_prompts = [
                self.tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for p in call_prompts
            ]

        max_tokens = int(self.args.gen_max_new_tokens)
        temperature = float(self.args.gen_temperature)
        top_p = float(self.args.gen_top_p)
        rep_pen = float(self.args.gen_repetition_penalty)
        model_name = str(self.vllm_sampling_model)

        def _one_request(i: int, prompt: str):
            c = clients[enabled_idxs[i % K]]
            try:
                outs = c.completions(
                    model=model_name,
                    prompt=prompt,
                    n=int(call_n),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=rep_pen,
                )
            except Exception as e:
                if self.accelerator.is_main_process:
                    prompt_snip = (prompt or "")[:100].replace("\n", "\\n")
                    print(f"[ERROR][ACTOR_GEN][vLLM] idx={i} base_url={c.base_url} failed: {e}, prompt='{prompt_snip}...'", flush=True)
                outs = []
            
            # Pad if needed
            while len(outs) < int(call_n):
                outs.append("")
            return i, outs[:int(call_n)]

        per_prompt_out: List[Optional[List[str]]] = [None] * len(call_prompts)
        with ThreadPoolExecutor(max_workers=min(64, max(1, K * 8))) as ex:
            for i, outs in ex.map(lambda t: _one_request(t[0], t[1]), enumerate(call_prompts)):
                per_prompt_out[i] = outs

        for i in range(len(per_prompt_out)):
            if per_prompt_out[i] is None:
                per_prompt_out[i] = [""] * int(call_n)

        if not force_n1:
            return per_prompt_out  # type: ignore

        # force_n1 mode: reassemble into groups
        out_texts: List[List[str]] = []
        ptr = 0
        for _ in range(orig_B):
            g_texts: List[str] = []
            for _ in range(int(group_size)):
                g_texts.append(per_prompt_out[ptr][0] if per_prompt_out[ptr] else "")
                ptr += 1
            out_texts.append(g_texts)

        return out_texts

    @torch.no_grad()
    def _generate_local_grpo(self, prompts: List[str], group_size: int) -> List[List[str]]:
        outs_grouped: List[List[str]] = []
        self.model.eval()

        for p in prompts:
            group: List[str] = []
            if hasattr(self.tok, "apply_chat_template") and getattr(self.tok, "chat_template", None):
                msgs = [{"role": "user", "content": p}]
                enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
                if isinstance(enc, dict):
                    input_ids = enc["input_ids"].to(self.model_device)
                    attention_mask = enc.get("attention_mask", None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.model_device)
                else:
                    input_ids = enc.to(self.model_device)
                    attention_mask = None
            else:
                enc = self.tok(p, return_tensors="pt")
                input_ids = enc["input_ids"].to(self.model_device)
                attention_mask = enc.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.model_device)

            gen_kwargs = dict(
                input_ids=input_ids.repeat(group_size, 1),
                max_new_tokens=self.args.gen_max_new_tokens,
                do_sample=True,
                temperature=self.args.gen_temperature,
                top_p=self.args.gen_top_p,
                repetition_penalty=self.args.gen_repetition_penalty,
                use_cache=True,
                pad_token_id=self.pad_token_id,
                output_router_logits=False,
                synced_gpus=True,
            )
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask.repeat(group_size, 1)

            gen_ids = self.model.generate(**gen_kwargs)
            for k in range(group_size):
                g = gen_ids[k]
                generated = g[input_ids.shape[-1]:]
                txt = self.tok.decode(generated, skip_special_tokens=True)
                group.append(txt)
            outs_grouped.append(group)

        self.model.train()
        return outs_grouped

    @torch.no_grad()
    def generate_candidates_grpo(
        self,
        prompts: List[str],
    ) -> List[List[str]]:
        """
        Generate GRPO candidate groups.
        Returns list of candidate groups (one group per prompt).
        """
        group_size = int(self.args.grpo_group_size)
        if group_size <= 0:
            raise ValueError("--grpo_group_size must be >= 1")

        N = len(prompts)
        t0 = time.time()
        if self.accelerator.is_main_process:
            mode = "vLLM" if self.use_vllm else "local"
            print(f"[ACTOR_GEN] start: {N} prompts x {group_size} candidates = {N * group_size} total, mode={mode}", flush=True)

        if self.use_vllm:
            result = self._generate_with_vllm_grpo(prompts, group_size)
        else:
            result = self._generate_local_grpo(prompts, group_size)

        dt = time.time() - t0
        if self.accelerator.is_main_process:
            total_cands = N * group_size
            print(f"[ACTOR_GEN] done: {total_cands} candidates in {dt:.2f}s ({total_cands/max(dt,0.001):.1f} cands/s)", flush=True)

        return result

    @torch.no_grad()
    def _centralized_generate_and_score(
        self,
        local_prompts: List[str],
        local_refs: List[str],
    ) -> Tuple[List[List[str]], torch.Tensor, torch.Tensor, List[List[str]], List[List[str]]]:
        """
        Rank0 does (generate candidates + judge) for ALL ranks' batches, then distributes
        each rank's slice. Reduces rank skew.

        Returns (local slice):
          candidate_groups, med_scores_group, smart_scores_group, med_judge_groups, smart_judge_groups
        """
        group_size = int(self.args.grpo_group_size)

        if self.world_size <= 1 or (not dist.is_available()) or (not dist.is_initialized()):
            cand_groups = self.generate_candidates_grpo(local_prompts)

            prompts_flat, refs_flat, cands_flat = [], [], []
            for p, r, group_c in zip(local_prompts, local_refs, cand_groups):
                for c in group_c:
                    prompts_flat.append(p)
                    refs_flat.append(r)
                    cands_flat.append(c)

            med_flat, med_j_flat = self.judge_scores_flat(prompts_flat, cands_flat, refs_flat)
            smart_flat, smart_j_flat = self.smart_scores_flat(prompts_flat, cands_flat, refs_flat)

            med_group = med_flat.view(len(local_prompts), group_size)
            smart_group = smart_flat.view(len(local_prompts), group_size)

            med_judge_groups: List[List[str]] = []
            smart_judge_groups: List[List[str]] = []
            for i in range(len(local_prompts)):
                med_judge_groups.append(med_j_flat[i * group_size:(i + 1) * group_size])
                smart_judge_groups.append(smart_j_flat[i * group_size:(i + 1) * group_size])

            return cand_groups, med_group, smart_group, med_judge_groups, smart_judge_groups

        comm = self._comm_group()
        if comm is None:
            raise RuntimeError(
                "[CENTRALIZE] cpu_group (gloo) is None; cannot run all_gather_object safely. "
                "Ensure distributed is initialized with gloo backend available."
            )

        payload: Dict[str, Any] = {"prompt": local_prompts, "reference": local_refs}
        gathered: List[Any] = [None] * int(self.world_size)

        # 全 rank でログを出力（ハング診断用）
        print(f"[rank{self.rank}][CENTRALIZE] entering all_gather_object with {len(local_prompts)} prompts", flush=True)

        t_gather = time.time()
        dist.all_gather_object(gathered, payload, group=comm)
        dt_gather = time.time() - t_gather

        if self.accelerator.is_main_process:
            print(f"[CENTRALIZE] all_gather_object done in {dt_gather:.2f}s", flush=True)
        if self._dbg_enabled():
            print(f"[rank{self.rank}][CENTRALIZE] all_gather_object done in {dt_gather:.2f}s", flush=True)

        if self.rank == 0:
            prompts_all: List[str] = []
            refs_all: List[str] = []
            per_rank_B: List[int] = []

            for g in gathered:
                gg = g if isinstance(g, dict) else {}
                ps = gg.get("prompt") or []
                rs = gg.get("reference") or []
                ps = list(ps) if isinstance(ps, list) else []
                rs = list(rs) if isinstance(rs, list) else []
                per_rank_B.append(len(ps))
                prompts_all.extend(ps)
                refs_all.extend(rs)

            total_prompts = len(prompts_all)
            print(f"[CENTRALIZE] rank0 gathered {total_prompts} prompts from {self.world_size} ranks: {per_rank_B}", flush=True)

            cand_groups_all = self.generate_candidates_grpo(prompts_all)

            prompts_flat, refs_flat, cands_flat = [], [], []
            for p, r, group_c in zip(prompts_all, refs_all, cand_groups_all):
                for c in group_c:
                    prompts_flat.append(p)
                    refs_flat.append(r)
                    cands_flat.append(c)

            med_flat_all, med_j_flat_all = self.judge_scores_flat(prompts_flat, cands_flat, refs_flat)
            smart_flat_all, smart_j_flat_all = self.smart_scores_flat(prompts_flat, cands_flat, refs_flat)

            med_group_all_cpu = med_flat_all.view(len(prompts_all), group_size).to("cpu")
            smart_group_all_cpu = smart_flat_all.view(len(prompts_all), group_size).to("cpu")

            med_judge_groups_all: List[List[str]] = []
            smart_judge_groups_all: List[List[str]] = []
            for i in range(len(prompts_all)):
                med_judge_groups_all.append(med_j_flat_all[i * group_size:(i + 1) * group_size])
                smart_judge_groups_all.append(smart_j_flat_all[i * group_size:(i + 1) * group_size])

            per_rank_pack: List[Dict[str, Any]] = []
            ptr = 0
            # rank0 は自分のローカル分だけ judge 出力を保持（他rankへは配らない）
            rank0_med_judge_slice: List[List[str]] = []
            rank0_smart_judge_slice: List[List[str]] = []
            for rr in range(int(self.world_size)):
                b = int(per_rank_B[rr])
                if rr == 0:
                    rank0_med_judge_slice = med_judge_groups_all[ptr:ptr + b]
                    rank0_smart_judge_slice = smart_judge_groups_all[ptr:ptr + b]
                pack_item: Dict[str, Any] = {
                    "candidate_groups": cand_groups_all[ptr:ptr + b],
                    "med_scores_group": med_group_all_cpu[ptr:ptr + b].tolist(),
                    "smart_scores_group": smart_group_all_cpu[ptr:ptr + b].tolist(),
                }
                per_rank_pack.append(pack_item)
                ptr += b

            # ===== rank0 が scatter 前に全サンプルを samples.jsonl に書く（通信ゼロ）=====
            t_jsonl = time.time()
            self._write_samples_jsonl_centralized(
                step=int(self.global_step + 1),
                epoch=int(self.epoch_idx),
                prompts_all=prompts_all,
                refs_all=refs_all,
                candidate_groups_all=cand_groups_all,
                med_judge_outputs_all=med_judge_groups_all,
                smart_judge_outputs_all=smart_judge_groups_all,
                med_scores_all=med_group_all_cpu,
                smart_scores_all=smart_group_all_cpu,
                per_rank_B=per_rank_B,
            )
            dt_jsonl = time.time() - t_jsonl
            print(f"[CENTRALIZE] samples.jsonl written: {total_prompts} samples in {dt_jsonl:.2f}s", flush=True)
        else:
            per_rank_pack = []  # type: ignore

        recv: List[Any] = [None]
        if self.accelerator.is_main_process:
            print(f"[CENTRALIZE] rank0 entering scatter_object_list", flush=True)
        if self._dbg_enabled():
            print(f"[rank{self.rank}][CENTRALIZE] entering scatter_object_list", flush=True)

        t_scatter = time.time()
        if hasattr(dist, "scatter_object_list"):
            if self.rank == 0:
                dist.scatter_object_list(recv, per_rank_pack, src=0, group=comm)  # type: ignore[arg-type]
            else:
                dist.scatter_object_list(recv, None, src=0, group=comm)  # type: ignore[arg-type]
            pack = recv[0]
        else:
            objlist: List[Any] = [per_rank_pack if self.rank == 0 else None]
            dist.broadcast_object_list(objlist, src=0, group=comm)
            pack = objlist[0][int(self.rank)]
        dt_scatter = time.time() - t_scatter

        if self.accelerator.is_main_process:
            print(f"[CENTRALIZE] scatter_object_list done in {dt_scatter:.2f}s", flush=True)

        if not isinstance(pack, dict):
            raise RuntimeError(f"[centralize] received non-dict pack: {type(pack)}")

        cand_groups = pack.get("candidate_groups") or []
        med_scores_group_list = pack.get("med_scores_group") or []
        smart_scores_group_list = pack.get("smart_scores_group") or []

        if self._dbg_enabled():
            print(f"[rank{self.rank}][CENTRALIZE] received {len(cand_groups)} candidate groups", flush=True)

        med_scores_group = torch.tensor(med_scores_group_list, dtype=torch.float32, device=self.model_device)
        smart_scores_group = torch.tensor(smart_scores_group_list, dtype=torch.float32, device=self.model_device)

        # judge出力は配らない：rank0だけ保持、それ以外は空文字で埋める（shape維持）
        if self.rank == 0:
            med_judge_groups = rank0_med_judge_slice
            smart_judge_groups = rank0_smart_judge_slice
        else:
            b = len(cand_groups)
            med_judge_groups = [[""] * group_size for _ in range(b)]
            smart_judge_groups = [[""] * group_size for _ in range(b)]

        return cand_groups, med_scores_group, smart_scores_group, med_judge_groups, smart_judge_groups

    def _build_policy_batch(self, prompts_flat: List[str], cands_flat: List[str]):
        input_ids_list = []
        labels_list = []

        if hasattr(self.tok, "apply_chat_template") and getattr(self.tok, "chat_template", None):
            for p, c in zip(prompts_flat, cands_flat):
                msgs_user = [{"role": "user", "content": p}]
                msgs_full = [{"role": "user", "content": p}, {"role": "assistant", "content": c}]
                enc_user = self.tok.apply_chat_template(
                    msgs_user, add_generation_prompt=True, tokenize=True, return_tensors="pt"
                )
                enc_full = self.tok.apply_chat_template(
                    msgs_full, add_generation_prompt=False, tokenize=True, return_tensors="pt"
                )

                if isinstance(enc_user, dict):
                    ids_user = enc_user["input_ids"][0]
                else:
                    ids_user = enc_user[0]
                if isinstance(enc_full, dict):
                    ids_full = enc_full["input_ids"][0]
                else:
                    ids_full = enc_full[0]

                start = ids_user.shape[0]
                ids = ids_full
                labels = torch.full_like(ids, -100)
                labels[start:] = ids[start:]
                input_ids_list.append(ids)
                labels_list.append(labels)
        else:
            for p, c in zip(prompts_flat, cands_flat):
                full_text = p + "\n" + c
                enc = self.tok(full_text, return_tensors="pt")
                ids = enc["input_ids"][0]
                enc_p = self.tok(p, return_tensors="pt", add_special_tokens=False)
                p_len = enc_p["input_ids"].shape[1]
                labels = torch.full_like(ids, -100)
                if p_len < ids.shape[0]:
                    labels[p_len:] = ids[p_len:]
                else:
                    labels[:] = ids
                input_ids_list.append(ids)
                labels_list.append(labels)

        max_len = min(max(x.shape[0] for x in input_ids_list), self.args.seq_len)
        batch_input_ids = []
        batch_labels = []
        batch_attn = []

        for ids, lbl in zip(input_ids_list, labels_list):
            ids = ids[:max_len]
            lbl = lbl[:max_len]
            real_len = ids.shape[0]
            pad_len = max_len - real_len
            if pad_len > 0:
                ids = torch.nn.functional.pad(ids, (0, pad_len), value=self.pad_token_id)
                lbl = torch.nn.functional.pad(lbl, (0, pad_len), value=-100)
            # パディング長ベースで attention_mask を構築する。
            # (ids != pad_token_id) 方式だと pad_token_id == eos_token_id の場合に
            # 応答末尾の本物の EOS までマスクされてしまう。
            attn = torch.zeros(max_len, dtype=torch.long)
            attn[:real_len] = 1
            batch_input_ids.append(ids)
            batch_labels.append(lbl)
            batch_attn.append(attn)

        return (
            torch.stack(batch_input_ids, dim=0).to(self.model_device),
            torch.stack(batch_attn, dim=0).to(self.model_device),
            torch.stack(batch_labels, dim=0).to(self.model_device),
        )

    def _logp_tokens_from_logits(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns token-level log probabilities for GRPO (PPO-clip on tokens).
        
        logits: (B, seq_len, vocab_size)
        labels: (B, seq_len) with -100 for ignored positions
        Returns:
            token_logp: (B, T-1) log-prob at each position (0 for ignored)
            valid_mask: (B, T-1) 1.0 for generated tokens, 0.0 otherwise
            lengths: (B,) number of valid tokens per sample (>=1)
        """
        logits_shifted = logits[:, :-1, :]  # (B, seq_len-1, vocab)
        labels_shifted = labels[:, 1:]       # (B, seq_len-1)

        # Use fp32 for log_softmax to avoid -inf with bf16 (low-prob tokens)
        log_probs = torch.log_softmax(logits_shifted.float(), dim=-1)
        valid_mask = (labels_shifted != -100).float()

        safe_labels = labels_shifted.clone()
        safe_labels[safe_labels == -100] = 0

        token_logp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        token_logp = token_logp * valid_mask

        lengths = valid_mask.sum(dim=-1).clamp_min(1.0)
        return token_logp, valid_mask, lengths

    def _kl_pos_estimate(self, logp_token: torch.Tensor, ref_logp_token: torch.Tensor, x_clip: Optional[float] = None) -> torch.Tensor:
        """
        Unbiased non-negative KL estimator.
        KL(p||q) ≈ E_p[log(p/q)] = E_p[logp - logq]
        
        Using: expm1(-x) + x = exp(-x) - 1 + x >= 0 for all x
        where x = logp - ref_logp
        
        fp32 + clamp for numerical stability (bf16 can produce -inf logprobs).
        """
        if x_clip is None:
            x_clip = float(getattr(self.args, "kl_x_clip", 20.0) or 20.0)
        x = (logp_token - ref_logp_token).float().clamp(-x_clip, x_clip)
        kl = torch.expm1(-x) + x
        # Safety: replace NaN/Inf with 0 (can occur if logprobs were -inf)
        return torch.nan_to_num(kl, nan=0.0, posinf=0.0, neginf=0.0)

    @torch.no_grad()
    def _compute_ref_token_logp_no_lora(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute token-level log probabilities with LoRA disabled (reference policy).
        Returns: (B, T-1) token log probs
        """
        peft_model = self.model.module if isinstance(self.model, FSDP) else self.model
        was_training = self.model.training
        self.model.eval()

        with lora_disabled(peft_model):
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_router_logits=False,
                return_dict=True,
            )

        if was_training:
            self.model.train()

        ref_token_logp, _, _ = self._logp_tokens_from_logits(out.logits, labels)
        return ref_token_logp.detach()

    def _logp_mean_from_logits(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        logits: (B, seq_len, vocab_size)
        labels: (B, seq_len) with -100 for ignored positions
        Returns:
            logp_mean: (B,) average log-prob over valid tokens
            lengths: (B,) number of valid tokens per sample
        """
        # Shift for next-token prediction
        logits_shifted = logits[:, :-1, :]  # (B, seq_len-1, vocab)
        labels_shifted = labels[:, 1:]       # (B, seq_len-1)

        # Use fp32 for log_softmax to avoid -inf with bf16 (low-prob tokens)
        log_probs = torch.log_softmax(logits_shifted.float(), dim=-1)
        valid_mask = (labels_shifted != -100).float()

        # Replace -100 with 0 for gather (won't affect result due to masking)
        safe_labels = labels_shifted.clone()
        safe_labels[safe_labels == -100] = 0

        # Gather log-probs of target tokens
        token_logp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        token_logp = token_logp * valid_mask

        lengths = valid_mask.sum(dim=-1).clamp_min(1.0)
        logp_mean = token_logp.sum(dim=-1) / lengths

        return logp_mean, lengths

    @torch.no_grad()
    def _compute_ref_logp_mean_no_lora(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reference log-probs with LoRA disabled (base model only).
        Used for KL penalty calculation.
        """
        # Get underlying PEFT model
        peft_model = self.model.module if isinstance(self.model, FSDP) else self.model

        # Check if PEFT disable_adapter API is available
        # Must match lora_disabled() logic: disable_adapter() OR disable_adapters()/enable_adapters()
        has_disable_api = (
            hasattr(peft_model, "disable_adapter") or
            (hasattr(peft_model, "disable_adapters") and hasattr(peft_model, "enable_adapters"))
        )

        if not has_disable_api:
            if self.accelerator.is_main_process and self.global_step == 0:
                print("[WARN][KL] PEFT disable_adapter/disable_adapters API not found. KL penalty will be ~0.", flush=True)

        # Save training state and switch to eval (avoid dropout noise in ref)
        was_training = self.model.training
        self.model.eval()

        with lora_disabled(peft_model):
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_router_logits=False,
                return_dict=True,
            )

        # Restore training state
        if was_training:
            self.model.train()

        ref_logp_mean, _ = self._logp_mean_from_logits(out.logits, labels)
        return ref_logp_mean

    def step(self, batch):
        t_step0 = time.time()

        current_step_num = self.global_step + 1
        if self.accelerator.is_main_process:
            print(f"[STEP] ===== start step={current_step_num} =====", flush=True)

        prompts: List[str] = batch["prompt"]
        refs: List[str] = batch["reference"]
        B = len(prompts)

        # ===== 空バッチチェック（全rank同期で判定）=====
        # 一部rankだけB==0だとデッドロックするので、全rankで同期して判定
        b_local = torch.tensor(B, device=self.model_device, dtype=torch.int32)
        b_min = b_local.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(b_min, op=dist.ReduceOp.MIN)

        if b_min.item() == 0:
            # 少なくとも1つのrankが空バッチ → 全rankでエラー（デッドロック防止）
            raise RuntimeError(
                f"[rank{self.rank}] Empty batch detected (local B={B}, global min B={b_min.item()}). "
                "This indicates data distribution issue. Use --epochs mode or ensure dataset size is divisible by world_size * batch_size."
            )

        group_size = int(self.args.grpo_group_size)

        current_step_preview = self.global_step + 1
        if self.accelerator.is_main_process:
            print(f"[STEP] ===== step {current_step_preview} start: B={B}, group_size={group_size} =====", flush=True)

        # Determine reward mode
        reward_mode = str(getattr(self.args, "reward_mode", "grpo") or "grpo").lower()

        # candidate + judge
        if bool(getattr(self.args, "centralize_actor_judge", False)):
            candidate_groups, med_scores_group, smart_scores_group, med_judge_outputs_group, smart_judge_outputs_group = \
                self._centralized_generate_and_score(prompts, refs)
        else:
            candidate_groups = self.generate_candidates_grpo(prompts)

            prompts_flat_tmp, refs_flat_tmp, cands_flat_tmp = [], [], []
            for p, r, group_c in zip(prompts, refs, candidate_groups):
                for c in group_c:
                    prompts_flat_tmp.append(p)
                    refs_flat_tmp.append(r)
                    cands_flat_tmp.append(c)

            med_scores_flat, med_judge_outputs_flat = self.judge_scores_flat(prompts_flat_tmp, cands_flat_tmp, refs_flat_tmp)
            smart_scores_flat, smart_judge_outputs_flat = self.smart_scores_flat(prompts_flat_tmp, cands_flat_tmp, refs_flat_tmp)

            med_scores_group = med_scores_flat.view(B, group_size)
            smart_scores_group = smart_scores_flat.view(B, group_size)

            med_judge_outputs_group = []
            smart_judge_outputs_group = []
            for i in range(B):
                med_judge_outputs_group.append(med_judge_outputs_flat[i * group_size:(i + 1) * group_size])
                smart_judge_outputs_group.append(smart_judge_outputs_flat[i * group_size:(i + 1) * group_size])

        # flatten for policy batch
        prompts_flat, cands_flat = [], []
        for p, group_c in zip(prompts, candidate_groups):
            for c in group_c:
                prompts_flat.append(p)
                cands_flat.append(c)

        med_scores_flat2 = med_scores_group.contiguous().view(-1)
        smart_scores_flat2 = smart_scores_group.contiguous().view(-1)

        # _combine_med_smart is always called for logging/baseline stats
        base_final_scores_flat, smart_add_flat, hack_mask_flat = self._combine_med_smart(med_scores_flat2, smart_scores_flat2)

        input_ids, attention_mask, labels = self._build_policy_batch(prompts_flat, cands_flat)

        # ===== KL penalty setup =====
        kl_beta = float(getattr(self.args, "kl_beta", 0.0) or 0.0)
        kl_clamp_min0 = bool(getattr(self.args, "kl_clamp_min0", True))
        # For logging (will be populated later for GRPO/GDPO)
        kl_raw_flat = torch.zeros_like(base_final_scores_flat)
        kl_pen_flat = torch.zeros_like(base_final_scores_flat)
        kl_sat_flat = torch.zeros_like(base_final_scores_flat)  # saturation rate tracking
        ref_logp_mean: Optional[torch.Tensor] = None  # for GSPO sequence-level KL

        # ref_logp_mean is only needed for GSPO (sequence-level KL)
        # GRPO/GDPO use token-level KL computed later
        need_ref_logp = (kl_beta > 0 and reward_mode == "gspo")

        if need_ref_logp:
            t_kl = self._phase_start("kl_ref_forward")
            ref_logp_mean = self._compute_ref_logp_mean_no_lora(input_ids, attention_mask, labels)
            self._phase_end("kl_ref_forward", t_kl)

        self._pre_fwd_barrier()

        # === forward (policy with LoRA) ===
        # Use train() mode to preserve gradient checkpointing
        # Use dropout_disabled() to ensure consistent logp (no dropout noise)
        t0 = self._phase_start("forward")
        self.model.train()
        with dropout_disabled(self.model):
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_router_logits=False,
                return_dict=True,
            )
        self._phase_end("forward", t0)

        logp_mean, lengths = self._logp_mean_from_logits(out.logits, labels)

        # GSPO: compute logp_old_mean (π_old) using mean (論文準拠: length-normalized)
        # s_i(θ) = exp(1/|y| * Σ log(π_θ/π_old)) = exp(mean_logp - mean_logp_old)
        logp_old_mean: Optional[torch.Tensor] = None
        if reward_mode == "gspo":
            logp_old_mean = logp_mean.detach().clone()

        # KL is handled in the loss for all modes (GRPO/GDPO token-level; GSPO sequence-level)
        # No longer subtracting KL from rewards
        final_scores_flat = base_final_scores_flat
        final_scores_group = final_scores_flat.view(B, group_size)

        # ===== 全ランク集計で baseline / 表示統計 =====
        global_mean_final, global_std_final, _ = self._global_mean_std(final_scores_flat)
        mean_final = float(global_mean_final)

        if self.baseline is None:
            self.baseline = mean_final
        else:
            self.baseline = (1 - self.args.ema_alpha) * self.baseline + self.args.ema_alpha * mean_final

        mean_group = final_scores_group.mean(dim=1, keepdim=True)
        std_group = final_scores_group.std(dim=1, keepdim=True, unbiased=False)

        gdpo_eps = float(getattr(self.args, "gdpo_eps", 1e-6) or 1e-6)

        if reward_mode == "gdpo":
            # ===== GDPO (論文 Eq.4-6): Decoupled normalization =====
            # 1) per-reward group-wise normalization
            med_g = med_scores_group.to(dtype=torch.float32)  # r_1^(i,j)
            smart_g = smart_scores_group.to(dtype=torch.float32)  # r_2^(i,j)
            
            # (optional) put SMART onto a 0..1 scale; normalization will remove scale anyway
            smart_max = float(getattr(self.args, "smart_max_score", 10.0) or 10.0)
            smart_g = smart_g / max(1e-6, smart_max)

            med_mean = med_g.mean(dim=1, keepdim=True)
            med_std = med_g.std(dim=1, keepdim=True, unbiased=False)
            z_med = (med_g - med_mean) / (med_std + gdpo_eps)  # \tilde A_1

            sm_mean = smart_g.mean(dim=1, keepdim=True)
            sm_std = smart_g.std(dim=1, keepdim=True, unbiased=False)
            z_smart = (smart_g - sm_mean) / (sm_std + gdpo_eps)  # \tilde A_2

            # 2) weighted sum (論文 Eq.5). Keep medical weight=1.0, SMART weight=smart_reward_weight
            w_smart = float(getattr(self.args, "smart_reward_weight", 0.0) or 0.0)
            adv_tilde = z_med + w_smart * z_smart

            # 3) batch-wise advantage normalization (論文 Eq.6)
            if bool(getattr(self.args, "gdpo_whiten_global", True)):
                flat = adv_tilde.view(-1)
                g_mean, g_std, _ = self._global_mean_std(flat)
                adv_group = (adv_tilde - g_mean) / (g_std + gdpo_eps)
            else:
                adv_group = adv_tilde
        else:
            # ===== Classic GRPO: group normalize on final reward =====
            adv_group = (final_scores_group - mean_group) / (std_group + 1e-6)

        if self.args.reward_scale is not None:
            adv_group = adv_group * self.args.reward_scale
        if self.args.reward_clip > 0:
            adv_group = adv_group.clamp(-self.args.reward_clip, self.args.reward_clip)
        advantage = adv_group.view(-1)

        # ===== Multi-epoch update =====
        # GRPO/GDPO: grpo_epochs (論文のμ、基本1)
        # GSPO: gspo_epochs
        if reward_mode in ("grpo", "gdpo"):
            inner_epochs = int(getattr(self.args, "grpo_epochs", 1) or 1)
        elif reward_mode == "gspo":
            inner_epochs = int(getattr(self.args, "gspo_epochs", 1) or 1)
        else:
            inner_epochs = 1

        # For GRPO/GDPO: compute token-level logp from existing out.logits (no extra forward)
        token_logp: Optional[torch.Tensor] = None
        token_mask: Optional[torch.Tensor] = None
        token_lengths: Optional[torch.Tensor] = None
        token_logp_old: Optional[torch.Tensor] = None
        ref_token_logp: Optional[torch.Tensor] = None

        if reward_mode in ("grpo", "gdpo"):
            # Reuse out.logits from the forward pass above
            token_logp, token_mask, token_lengths = self._logp_tokens_from_logits(out.logits, labels)
            # π_old = current policy (detached)
            token_logp_old = token_logp.detach().clone()

            # Compute ref token logp for KL (if needed)
            if kl_beta > 0:
                ref_token_logp = self._compute_ref_token_logp_no_lora(input_ids, attention_mask, labels)

        for inner_epoch in range(inner_epochs):
            # For GRPO/GDPO/GSPO epoch > 0, recompute logp with current (updated) policy
            if inner_epoch > 0:
                self.model.train()
                with dropout_disabled(self.model):
                    out = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_router_logits=False,
                        return_dict=True,
                    )
                if reward_mode in ("grpo", "gdpo"):
                    token_logp, token_mask, token_lengths = self._logp_tokens_from_logits(out.logits, labels)
                elif reward_mode == "gspo":
                    logp_mean, _ = self._logp_mean_from_logits(out.logits, labels)

            # ===== Loss calculation =====
            if reward_mode in ("grpo", "gdpo"):
                # ===== GRPO/GDPO: PPO-clip on token-level (論文 Eq.3 準拠) =====
                # ratio = exp(logp - logp_old) per token
                # advantage is broadcast to all tokens (outcome supervision)
                assert token_logp is not None and token_logp_old is not None
                assert token_mask is not None and token_lengths is not None

                grpo_clip_eps = float(getattr(self.args, "grpo_clip_eps", 0.2) or 0.2)
                grpo_logratio_clip = float(getattr(self.args, "grpo_logratio_clip", 20.0) or 20.0)

                log_ratio = (token_logp - token_logp_old).clamp(-grpo_logratio_clip, grpo_logratio_clip)
                ratio = torch.exp(log_ratio)
                clipped_ratio = ratio.clamp(1.0 - grpo_clip_eps, 1.0 + grpo_clip_eps)

                # Broadcast advantage to token dimension: (N,) -> (N,1)
                adv_tok = advantage.detach().unsqueeze(-1)

                surr1 = ratio * adv_tok
                surr2 = clipped_ratio * adv_tok
                surr = torch.min(surr1, surr2)

                # Token-average loss (only over valid tokens)
                loss_per_sample = -((surr * token_mask).sum(dim=-1) / token_lengths)
                loss = loss_per_sample.mean()

                # KL penalty as loss term (論文準拠: 報酬に混ぜない)
                kl_loss_val = 0.0
                if kl_beta > 0 and ref_token_logp is not None:
                    kl_clamp_max = float(getattr(self.args, "kl_clamp_max", 10.0) or 10.0)
                    kl_x_clip = float(getattr(self.args, "kl_x_clip", 20.0) or 20.0)
                    
                    # Debug: check for -inf in logprobs (first step only)
                    if inner_epoch == 0 and self.global_step == 0 and self.accelerator.is_main_process:
                        logp_finite = torch.isfinite(token_logp).float().mean().item()
                        ref_finite = torch.isfinite(ref_token_logp).float().mean().item()
                        print(f"[KL_DEBUG] token_logp finite: {logp_finite:.4f}, ref_logp finite: {ref_finite:.4f}", flush=True)
                    
                    kl_tok = self._kl_pos_estimate(token_logp, ref_token_logp, x_clip=kl_x_clip)
                    kl_mean_raw = (kl_tok * token_mask).sum(dim=-1) / token_lengths
                    # Clamp per-sample KL to prevent outliers from dominating loss
                    kl_mean_for_loss = kl_mean_raw.clamp(max=kl_clamp_max)
                    kl_loss = kl_beta * kl_mean_for_loss.mean()
                    kl_loss_val = kl_loss.item()
                    loss = loss + kl_loss
                    # Update for logging: use clamped KL (matches what's in loss)
                    kl_raw_flat = kl_mean_for_loss.detach()
                    kl_pen_flat = (kl_beta * kl_mean_for_loss).detach()
                    # Track saturation rate (how many samples hit the clamp)
                    kl_sat_flat = (kl_mean_raw > kl_clamp_max).float().detach()

                # Log
                if self.accelerator.is_main_process:
                    tag = "GDPO" if reward_mode == "gdpo" else "GRPO"
                    clip_frac = ((ratio < 1.0 - grpo_clip_eps) | (ratio > 1.0 + grpo_clip_eps)).float()
                    clip_frac = (clip_frac * token_mask).sum() / token_mask.sum()
                    kl_info = f", kl_loss={kl_loss_val:.4f}" if kl_beta > 0 else ""
                    print(f"[{tag}] epoch={inner_epoch}/{inner_epochs}, clip_frac={clip_frac.item():.3f}, ratio_mean={ratio.mean().item():.4f}{kl_info}", flush=True)

            elif reward_mode == "gspo":
                # ===== GSPO: PPO-clip on sequence-level (論文準拠) =====
                # s_i(θ) = exp(1/|y| * Σ log(π_θ/π_old)) = exp(logp_mean - logp_old_mean)
                assert logp_old_mean is not None, "GSPO requires logp_old_mean"

                gspo_logratio_clip = float(getattr(self.args, "gspo_logratio_clip", 20.0) or 20.0)
                # Use mean (length-normalized) for ratio (論文準拠)
                log_ratio = (logp_mean - logp_old_mean).clamp(-gspo_logratio_clip, gspo_logratio_clip)
                ratio = torch.exp(log_ratio)

                # Asymmetric clip (論文準拠: left=3e-4, right=4e-4)
                gspo_eps_low = float(getattr(self.args, "gspo_clip_eps_low", 3e-4) or 3e-4)
                gspo_eps_high = float(getattr(self.args, "gspo_clip_eps_high", 4e-4) or 4e-4)
                clipped_ratio = ratio.clamp(1.0 - gspo_eps_low, 1.0 + gspo_eps_high)

                surr1 = ratio * advantage.detach()
                surr2 = clipped_ratio * advantage.detach()
                loss = -torch.min(surr1, surr2).mean()

                # KL penalty as loss term (use mean for length-normalized KL)
                kl_loss_val = 0.0
                if kl_beta > 0 and ref_logp_mean is not None:
                    kl_clamp_max = float(getattr(self.args, "kl_clamp_max", 10.0) or 10.0)
                    kl_raw_val = logp_mean - ref_logp_mean.detach()
                    if kl_clamp_min0:
                        kl_raw_val = kl_raw_val.clamp_min(0.0)
                    # Clamp per-sample KL to prevent outliers from dominating loss
                    kl_for_loss = kl_raw_val.clamp(max=kl_clamp_max)
                    kl_loss = kl_beta * kl_for_loss.mean()
                    kl_loss_val = kl_loss.item()
                    loss = loss + kl_loss
                    # Update for logging: use clamped KL (matches what's in loss)
                    kl_raw_flat = kl_for_loss.detach()
                    kl_pen_flat = (kl_beta * kl_for_loss).detach()
                    # Track saturation rate (how many samples hit the clamp)
                    kl_sat_flat = (kl_raw_val > kl_clamp_max).float().detach()

                # Log
                if self.accelerator.is_main_process:
                    clip_frac = ((ratio < 1.0 - gspo_eps_low) | (ratio > 1.0 + gspo_eps_high)).float().mean().item()
                    kl_info = f", kl_loss={kl_loss_val:.4f}" if kl_beta > 0 else ""
                    print(f"[GSPO] epoch={inner_epoch}/{inner_epochs}, clip_frac={clip_frac:.3f}, ratio_mean={ratio.mean().item():.4f}{kl_info}", flush=True)

            else:
                raise RuntimeError(f"Unknown reward_mode: {reward_mode}. Must be one of: grpo, gdpo, gspo")

            # ===== NaN/Inf チェック（全rank同期で判定）=====
            loss_bad_local = torch.tensor(
                1 if (torch.isnan(loss) or torch.isinf(loss)) else 0,
                device=self.model_device,
                dtype=torch.int32,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(loss_bad_local, op=dist.ReduceOp.SUM)
            skip_update = (loss_bad_local.item() > 0)

            if skip_update:
                safe_logp_sum = torch.nan_to_num(
                    logp_mean, nan=0.0, posinf=0.0, neginf=0.0
                ).sum()
                loss = safe_logp_sum * 0.0
                self.consecutive_nan_count += 1
                print(f"[rank{self.rank}][WARN] NaN/Inf detected (consecutive={self.consecutive_nan_count}), skipping update", flush=True)
                if self.consecutive_nan_count >= self.max_consecutive_nan:
                    raise RuntimeError(
                        f"[rank{self.rank}] NaN/Inf loss detected {self.consecutive_nan_count} times consecutively. "
                        "Training aborted. Check data, learning rate, or model state."
                    )
            else:
                self.consecutive_nan_count = 0

            # === backward ===
            self.opt.zero_grad(set_to_none=True)
            t1 = self._phase_start("backward")
            self.accelerator.backward(loss)
            self._phase_end("backward", t1)

            # === all_reduce ===
            t2 = self._phase_start("all_reduce")
            self._sync_lora_grads(self.lora_params)
            self._phase_end("all_reduce", t2)

            if not skip_update:
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.lora_params, self.args.grad_clip)
                self.opt.step()
            else:
                if self.accelerator.is_main_process:
                    print(f"[WARN] optimizer step skipped due to NaN/Inf loss", flush=True)

        # End of gspo_epochs loop

        current_step = self.global_step + 1

        # ===== samples jsonl 保存 =====
        try:
            self._write_samples_jsonl(
                step=int(current_step),
                epoch=int(self.epoch_idx),
                prompts=prompts,
                refs=refs,
                candidate_groups=candidate_groups,
                med_judge_outputs_group=med_judge_outputs_group,
                smart_judge_outputs_group=smart_judge_outputs_group,
                final_scores_group=final_scores_group,
                med_scores_group=med_scores_group,
                smart_scores_group=smart_scores_group,
            )
        except Exception as e:
            if self.accelerator.is_main_process:
                print(f"[WARN][SAMPLES] failed to write jsonl: {e}", flush=True)

        self._maybe_save_and_reload_vllm(current_step=current_step, force=False)

        dt_step = time.time() - t_step0
        if self.step_time_ema is None:
            self.step_time_ema = dt_step
        else:
            a = self.step_time_ema_alpha
            self.step_time_ema = (1 - a) * self.step_time_ema + a * dt_step

        # ===== 全 rank で統計計算（all_reduce を使うため）=====
        should_log = (current_step % self.args.log_every == 0)
        if should_log:
            med_mean, med_std, _ = self._global_mean_std(med_scores_flat2)
            smart_mean, smart_std, _ = self._global_mean_std(smart_scores_flat2)
            smart_add_mean, smart_add_std, _ = self._global_mean_std(smart_add_flat)
            hack_frac, _, _ = self._global_mean_std(hack_mask_flat)
            # KL統計
            kl_raw_mean, kl_raw_std, _ = self._global_mean_std(kl_raw_flat)
            kl_pen_mean, kl_pen_std, _ = self._global_mean_std(kl_pen_flat)
            kl_sat_frac, _, _ = self._global_mean_std(kl_sat_flat)  # saturation rate

        if should_log and self.accelerator.is_main_process:
            total = int(self.total_steps) if int(self.total_steps) > 0 else int(self.args.steps)
            done = int(current_step)
            pct = (done / total * 100.0) if total > 0 else 0.0

            elapsed_s = (time.time() - self.train_wall_start) if self.train_wall_start else 0.0
            step_s = float(self.step_time_ema or dt_step)
            remaining_steps = max(0, total - done)
            eta_s = remaining_steps * step_s
            finish_dt = datetime.now() + timedelta(seconds=eta_s)

            if int(getattr(self.args, "epochs", 0)) > 0 and self.steps_per_epoch > 0:
                ep_str = f"[epoch {self.epoch_idx}/{self.epoch_total} | ep_step {self.step_in_epoch}/{self.steps_per_epoch}] "
            else:
                ep_str = ""

            # KL部分のログ文字列
            kl_str = ""
            if kl_beta > 0:
                kl_str = f"kl={float(kl_raw_mean):.4f}±{float(kl_raw_std):.4f} kl_sat={float(kl_sat_frac):.3f} "

            mode_tag = reward_mode.upper()  # GRPO, GDPO, or GSPO
            print(
                f"{ep_str}[step {done:6d}/{total} ({pct:5.1f}%)] {mode_tag} loss={loss.item():.4f} "
                f"final(mean±std)={float(global_mean_final):.3f}±{float(global_std_final):.3f} "
                f"med(mean±std)={float(med_mean):.3f}±{float(med_std):.3f} "
                f"smart(mean±std)={float(smart_mean):.3f}±{float(smart_std):.3f} "
                f"smart_add(mean)={float(smart_add_mean):.3f} hack_frac={float(hack_frac):.3f} "
                f"{kl_str}"
                f"baseline(global-EMA)={self.baseline:.3f} | "
                f"elapsed {self._fmt_hms(elapsed_s)} "
                f"eta {self._fmt_hms(eta_s)} "
                f"finish {finish_dt.strftime('%Y-%m-%d %H:%M:%S')}",
                flush=True,
            )

            if self.wandb_run is not None:
                log_dict = {
                        "train/loss": float(loss.item()),
                        "train/reward_mode": reward_mode,
                        "reward/final_mean": float(global_mean_final),
                        "reward/final_std": float(global_std_final),
                        "reward/med_mean": float(med_mean),
                        "reward/med_std": float(med_std),
                        "reward/smart_mean": float(smart_mean),
                        "reward/smart_std": float(smart_std),
                        "reward/smart_add_mean": float(smart_add_mean),
                        "reward/smart_add_std": float(smart_add_std),
                        "reward/hack_frac": float(hack_frac),
                        "reward/baseline_ema": float(self.baseline),
                        "opt/lr": float(self.opt.param_groups[0]["lr"]),
                        "progress/step": done,
                        "progress/total_steps": total,
                        "progress/pct": float(pct),
                        "time/step_s_ema": float(step_s),
                        "time/elapsed_s": float(elapsed_s),
                        "time/eta_s": float(eta_s),
                        "epoch/idx": int(self.epoch_idx),
                        "epoch/total": int(self.epoch_total),
                        "epoch/step": int(self.step_in_epoch),
                        "epoch/steps_per_epoch": int(self.steps_per_epoch),
                }
                # KL統計を追加
                if kl_beta > 0:
                    log_dict["kl/mean"] = float(kl_raw_mean)
                    log_dict["kl/std"] = float(kl_raw_std)
                    log_dict["kl/sat_frac"] = float(kl_sat_frac)
                    log_dict["kl/beta"] = float(kl_beta)

                self.wandb_run.log(log_dict, step=int(done))

                self._wandb_log_samples_table(
                    step=done,
                    prompts=prompts,
                    refs=refs,
                    candidate_groups=candidate_groups,
                    final_scores_group=final_scores_group,
                    med_scores_group=med_scores_group,
                    smart_scores_group=smart_scores_group,
                    med_judge_outputs_group=med_judge_outputs_group,
                    smart_judge_outputs_group=smart_judge_outputs_group,
                )

        self.global_step += 1

        if self.accelerator.is_main_process:
            dt_total = time.time() - t_step0
            print(f"[STEP] ===== step {self.global_step} done in {dt_total:.2f}s =====", flush=True)

    def train(self):
        if self.accelerator.is_main_process:
            self.train_wall_start = time.time()

        if int(getattr(self.args, "epochs", 0)) > 0:
            epochs = int(self.args.epochs)
            global_batch = int(self.world_size) * int(self.args.batch_size)

            try:
                with open(self.args.dataset, "r", encoding="utf-8") as f:
                    dataset_size = sum(1 for _ in f)
            except Exception:
                dataset_size = 0
            if dataset_size <= 0:
                raise RuntimeError(f"Failed to count dataset lines: {self.args.dataset}")

            steps_per_epoch = (int(dataset_size) + int(global_batch) - 1) // int(global_batch)
            self.args.steps = int(epochs) * int(steps_per_epoch)
            self.total_steps = int(self.args.steps)

            self.epoch_total = int(epochs)
            self.steps_per_epoch = int(steps_per_epoch)

            step = 0
            it = iter(self.loader)
            if self.accelerator.is_main_process:
                print(f"[TRAIN] epochs mode: {epochs} epochs, {steps_per_epoch} steps/epoch, total_steps={self.total_steps}", flush=True)
            for ep in range(epochs):
                self.epoch_idx = int(ep) + 1
                if self.accelerator.is_main_process:
                    print(f"[TRAIN] starting epoch {self.epoch_idx}/{epochs}", flush=True)
                for ep_step in range(int(steps_per_epoch)):
                    self.step_in_epoch = int(ep_step) + 1
                    # 全 rank でログ
                    print(f"[rank{self.rank}][TRAIN] fetching batch for step {step + 1} (epoch {self.epoch_idx}, ep_step {self.step_in_epoch})", flush=True)
                    try:
                        batch = next(it)
                    except StopIteration:
                        print(f"[rank{self.rank}][TRAIN] iterator exhausted, resetting", flush=True)
                        it = iter(self.loader)
                        try:
                            batch = next(it)
                        except StopIteration:
                            # データセットが空の場合
                            print(f"[rank{self.rank}][ERROR][TRAIN] dataset is empty after reset", flush=True)
                            raise RuntimeError("Dataset is empty - cannot continue training")
                    batch_prompts = len(batch.get("prompt", []))
                    print(f"[rank{self.rank}][TRAIN] got batch with {batch_prompts} prompts, calling step()", flush=True)
                    self.step(batch)
                    step += 1
        else:
            # WARNING: steps モードで IterableDataset を使う場合、rank 間でイテレーション数が
            # 異なると all_gather_object でハングする可能性があります。
            # epochs モード (--epochs) の使用を推奨します。
            self.total_steps = int(self.args.steps)
            step = 0
            if self.accelerator.is_main_process:
                print(f"[TRAIN] steps mode: total_steps={self.total_steps}", flush=True)
                print(f"[WARN][TRAIN] steps mode with IterableDataset may hang if data is not evenly distributed. Consider using --epochs mode.", flush=True)
            while step < self.args.steps:
                if self.accelerator.is_main_process:
                    print(f"[TRAIN] entering loader iteration, current step={step}", flush=True)
                batch_count = 0
                for batch in self.loader:
                    batch_count += 1
                    batch_prompts = len(batch.get("prompt", []))
                    # 全 rank でログ（デバッグ用）
                    print(f"[rank{self.rank}][TRAIN] got batch #{batch_count} with {batch_prompts} prompts", flush=True)
                    self.step(batch)
                    step += 1
                    if step >= self.args.steps:
                        break
                # 全 rank でイテレーション終了をログ
                print(f"[rank{self.rank}][TRAIN] loader iteration ended after {batch_count} batches, step={step}", flush=True)

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            final_dir = os.path.join(self.args.save_dir, "final_adapter")
            print(f"[SAVE] final adapter -> {final_dir}", flush=True)
            self.save_lora_adapter_pretrained(final_dir)

        self.accelerator.wait_for_everyone()

        if self.wandb_run is not None and self.accelerator.is_main_process:
            try:
                self.wandb_run.finish()
            except Exception:
                pass


def build_argparser():
    ap = argparse.ArgumentParser(description="LoRA (PEFT) GRPO trainer (manual FSDP + optional vLLM runtime LoRA reload)")

    ap.add_argument("--model", type=str, default=os.environ.get(
        "BASE_MODEL",
        "/path/to/base_model",
    ))

    ap.add_argument("--dataset", type=str, default=os.environ.get(
        "SPECIALIST_CHATML_PATH",
        "/path/to/train.jsonl",
    ))
    ap.add_argument("--seq_len", type=int, default=2048)

    # LoRA knobs
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora_target_modules",
        type=str,
        default=None,
        help="Comma-separated LoRA target module leaf names. If omitted, auto-detect typical names.",
    )

    # GRPO / training
    ap.add_argument("--grpo_group_size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--max_consecutive_nan", type=int, default=10, help="Abort training after this many consecutive NaN/Inf losses")

    # generation
    ap.add_argument("--gen_max_new_tokens", type=int, default=192)
    ap.add_argument("--gen_temperature", type=float, default=0.7)
    ap.add_argument("--gen_top_p", type=float, default=0.9)
    ap.add_argument("--gen_repetition_penalty", type=float, default=1.05)

    # vLLM (OpenAI API server)
    ap.add_argument("--use_vllm", action="store_true", help="Use vLLM OpenAI server for sampling + runtime LoRA reload")
    ap.add_argument("--vllm_base_url", type=str, default=None, help="e.g. http://host:8000")
    ap.add_argument("--vllm_base_urls", type=str, default=None, help="comma-separated base URLs")
    ap.add_argument("--vllm_api_key", type=str, default=None, help="optional, if your vLLM is behind auth")
    ap.add_argument("--vllm_connection_timeout", type=int, default=240)

    ap.add_argument("--vllm_model_name", type=str, default=os.environ.get("VLLM_MODEL_NAME", None),
                    help="Base model name as served by vLLM (OpenAI 'model' field). If omitted, uses --model.")
    ap.add_argument("--vllm_lora_name", type=str, default=os.environ.get("VLLM_LORA_NAME", "train-lora"),
                    help="Adapter name (also used as OpenAI 'model' after loaded).")
    ap.add_argument("--vllm_lora_save_root", type=str, default=os.environ.get("VLLM_LORA_SAVE_ROOT", None),
                    help="Where to write adapters. Must be readable by vLLM server. Default: <save_dir>/adapters")

    ap.add_argument("--vllm_force_n1", action="store_true", help="Debug: emulate n>1 by repeated single-sample calls")

    ap.add_argument("--vllm_reload_max_retries", type=int, default=5, help="Max retries per vLLM server for unload/load")
    ap.add_argument("--vllm_reload_retry_backoff", type=float, default=2.0, help="Exponential backoff base seconds")
    ap.add_argument("--vllm_reload_retry_jitter", type=float, default=0.2, help="Jitter factor for backoff sleep")

    ap.add_argument("--save_updated_only", action="store_true", help="(compat) Ignored. Adapter-only saving is already used.")

    # medical judge
    ap.add_argument("--judge_max_new_tokens", type=int, default=2048)
    ap.add_argument("--judge_temperature", type=float, default=0.0)
    ap.add_argument("--judge_backend", type=str, default="local",
                    choices=["local", "openrouter", "openai", "cerebras",
                             "cerebras+openrouter", "openrouter+cerebras"],
                    help="Judge backend. Use 'X+Y' for fallback (e.g., cerebras+openrouter).")
    ap.add_argument("--judge_api_base", type=str, default=None)
    ap.add_argument("--judge_api_key", type=str, default=None)
    ap.add_argument("--judge_model_name", type=str, default=None)
    ap.add_argument("--judge_fallback_model_name", type=str, default=None,
                    help="Model name for fallback backend (e.g., OpenRouter model when primary is Cerebras).")
    ap.add_argument("--judge_max_retries", type=int, default=1000)

    ap.add_argument("--judge_connect_timeout", type=float, default=10.0, help="HTTP connect timeout (sec)")
    ap.add_argument("--judge_read_timeout", type=float, default=60.0, help="HTTP read timeout (sec)")
    ap.add_argument("--judge_retry_sleep", type=float, default=1.0, help="Base sleep for manual retry backoff (sec)")
    ap.add_argument("--judge_retry_sleep_max", type=float, default=60.0, help="Max sleep for manual retry backoff (sec)")
    ap.add_argument("--judge_http_retries", type=int, default=8, help="urllib3 Retry total (keep-alive session)")
    ap.add_argument("--judge_http_backoff", type=float, default=0.5, help="urllib3 Retry backoff_factor")
    ap.add_argument("--judge_parallel_workers", type=int, default=0,
                    help="External judge parallel workers (openrouter/openai/cerebras). 0=auto (~2*world_size, capped).")

    # SMART judge (optional overrides; default: reuse judge_*)
    ap.add_argument("--smart_backend", type=str, default=None,
                    choices=[None, "local", "openrouter", "openai", "cerebras",
                             "cerebras+openrouter", "openrouter+cerebras"],
                    help="SMART backend. Use 'X+Y' for fallback. Defaults to judge_backend.")
    ap.add_argument("--smart_api_base", type=str, default=None)
    ap.add_argument("--smart_api_key", type=str, default=None)
    ap.add_argument("--smart_model_name", type=str, default=None)
    ap.add_argument("--smart_fallback_model_name", type=str, default=None,
                    help="Model name for SMART fallback backend.")
    ap.add_argument("--smart_max_new_tokens", type=int, default=None)
    ap.add_argument("--smart_temperature", type=float, default=None)
    ap.add_argument("--smart_max_retries", type=int, default=None)
    ap.add_argument("--smart_parallel_workers", type=int, default=0,
                    help="SMART external judge parallel workers (openrouter/openai/cerebras). 0=auto (~2*world_size, capped).")

    # reward shaping
    ap.add_argument("--reward_mode", type=str, default="grpo", choices=["grpo", "gdpo", "gspo"],
                    help="grpo: token PPO-clip + group norm. gdpo: token PPO-clip + decoupled norm. gspo: seq PPO-clip.")
    # GRPO/GDPO (論文準拠: PPO-clip on tokens)
    ap.add_argument("--grpo_clip_eps", type=float, default=0.2,
                    help="(GRPO/GDPO) Clip epsilon for token-level ratio. Default: 0.2")
    ap.add_argument("--grpo_logratio_clip", type=float, default=20.0,
                    help="(GRPO/GDPO) Log-ratio clamp to prevent exp() overflow. Default: 20.0")
    ap.add_argument("--grpo_epochs", type=int, default=1,
                    help="(GRPO/GDPO) Inner epochs (論文のμ、実運用は1が基本). Default: 1")
    # GDPO (論文 Eq.4-6: Decoupled advantage + batch-wise normalization)
    ap.add_argument("--gdpo_eps", type=float, default=1e-6,
                    help="Epsilon for GDPO group normalization to avoid division by zero.")
    ap.add_argument("--gdpo_whiten_global", dest="gdpo_whiten_global", action="store_true",
                    help="(default: enabled) GDPO batch-wise advantage normalization (Eq.6).")
    ap.add_argument("--no_gdpo_whiten_global", dest="gdpo_whiten_global", action="store_false",
                    help="Disable GDPO batch-wise advantage normalization.")
    ap.set_defaults(gdpo_whiten_global=True)
    # GSPO (論文準拠: 左右非対称clip, eps_low=3e-4, eps_high=4e-4)
    ap.add_argument("--gspo_clip_eps_low", type=float, default=3e-4,
                    help="GSPO left clip epsilon (ratio >= 1-eps_low). Default: 3e-4 (論文準拠)")
    ap.add_argument("--gspo_clip_eps_high", type=float, default=4e-4,
                    help="GSPO right clip epsilon (ratio <= 1+eps_high). Default: 4e-4 (論文準拠)")
    ap.add_argument("--gspo_logratio_clip", type=float, default=20.0,
                    help="GSPO log-ratio clamp to prevent exp() overflow. Default: 20.0")
    ap.add_argument("--gspo_epochs", type=int, default=4,
                    help="GSPO inner epochs. 論文は4 mini-batches更新。Default: 4")
    ap.add_argument("--reward_scale", type=float, default=1.0)
    ap.add_argument("--reward_clip", type=float, default=1.0)
    ap.add_argument("--ema_alpha", type=float, default=0.05)
    ap.add_argument("--reward_cache", type=str, default="judge_score_cache.jsonl")

    # KL penalty (optional regularization to stay close to base model)
    ap.add_argument("--kl_beta", type=float, default=0.0,
                    help="KL penalty coefficient. 0 disables KL penalty. Typical values: 0.01-0.1")
    ap.add_argument("--kl_clamp_max", type=float, default=10.0,
                    help="Max per-sample KL to prevent outliers from dominating. Default: 10.0")
    ap.add_argument("--kl_x_clip", type=float, default=20.0,
                    help="Clamp range for log-ratio x in KL estimator. Default: 20.0 (veRL style)")
    ap.add_argument("--kl_clamp_min0", action="store_true", default=True,
                    help="(GSPO only) Clamp KL to min 0. GRPO/GDPO use non-negative KL estimator.")
    ap.add_argument("--no_kl_clamp_min0", dest="kl_clamp_min0", action="store_false",
                    help="(GSPO only) Disable KL clamping.")
    ap.set_defaults(kl_clamp_min0=True)

    # SMART reward settings (used in GRPO final_scores and logging; GDPO uses decoupled normalization)
    ap.add_argument("--smart_reward_cache", type=str, default="smart_score_cache.jsonl")
    ap.add_argument("--smart_max_score", type=float, default=10.0, help="SMART score max (for normalization).")
    ap.add_argument("--smart_reward_weight", type=float, default=0.2,
                    help="Weight for SMART reward. GRPO: used in final_scores. GDPO: used in Eq.5 weighted sum.")
    ap.add_argument("--smart_gate_mode", type=str, default="soft", choices=["soft", "hard", "none"],
                    help="(GRPO only) soft: smart_add *= medical. hard: add only if medical>=threshold. none: no gating.")
    ap.add_argument("--smart_gate_threshold", type=float, default=0.5,
                    help="(GRPO only) Threshold for hard gate.")
    ap.add_argument("--smart_reward_cap", type=float, default=0.2,
                    help="(GRPO only) Cap for SMART add (<=0 disables cap).")
    ap.add_argument("--smart_hack_penalty", type=float, default=0.2,
                    help="(GRPO only) Penalty when (medical<=med_le & smart>=smart_ge). 0 disables.")
    ap.add_argument("--smart_hack_med_le", type=float, default=0.0,
                    help="(GRPO only) Medical threshold for hack detection.")
    ap.add_argument("--smart_hack_smart_ge", type=float, default=8.0,
                    help="(GRPO only) SMART threshold for hack detection.")

    # misc
    ap.add_argument("--debug_print", action="store_true")
    ap.add_argument("--debug_barrier", action="store_true",
                    help="If set, do pre-forward barrier and log its wait time (rank desync diagnosis).")
    ap.add_argument("--sync_debug", action="store_true",
                    help="If set, torch.cuda.synchronize() at phase boundaries for accurate 'done' logs (slower).")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="lora_grpo")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_dir", type=str, default="lora_grpo_outputs")

    # centralize generation+judge
    ap.add_argument("--centralize_actor_judge", action="store_true",
                    help="Rank0 runs generation+judge for all ranks, then distributes results (reduces rank skew).")

    # wandb table
    ap.add_argument("--wandb_table_rows", type=int, default=4, help="How many rows from the current batch to log as a W&B table")
    ap.add_argument("--wandb_table_text_max_chars", type=int, default=2000, help="Max chars per text cell in W&B table (0=no clip)")

    # samples jsonl
    ap.add_argument("--samples_jsonl_path", type=str, default="samples.jsonl",
                    help="Save prompt/actor/judge/score/epoch/step as ONE jsonl (rank0 writes). Relative path => under save_dir.")
    ap.add_argument("--samples_jsonl_every", type=int, default=1, help="Write samples jsonl every N steps. 0 disables.")
    ap.add_argument("--samples_jsonl_overwrite", action="store_true", help="Overwrite samples jsonl at start (otherwise append).")
    ap.add_argument("--samples_jsonl_text_max_chars", type=int, default=0, help="Max chars per text field in samples jsonl (0=no clip).")

    return ap


def main():
    args = build_argparser().parse_args()

    if args.vllm_model_name is None:
        args.vllm_model_name = args.model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU が必要です。")

    set_seed(42)

    trainer = LoraGRPOTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
