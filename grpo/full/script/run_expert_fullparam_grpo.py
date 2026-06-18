#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expert-only full-parameter RL with GRPO/GDPO/GSPO objective and LLM-as-judge reward.

Features:
- Medical judge (correctness scoring)
- SMART judge (helpfulness/structure scoring)
- reward_mode: grpo, gdpo, gspo
- PPO-clip (token-level for GRPO/GDPO, sequence-level for GSPO)
- parallel judge execution
- judge backends: local, openrouter, openai, cerebras (with fallback: cerebras+openrouter)
- centralize_actor_judge: rank0 generates all batches to reduce rank skew
- samples jsonl output
- eval (temp=0) periodic evaluation
- vLLM parallel sync with retry
"""

import os
import re
import json
import random
import argparse
import hashlib
import time
import threading
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Any, Dict
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

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

# Optional: TRL vLLM client
try:
    from trl.extras.vllm_client import VLLMClient
    _TRL_VLLM_AVAILABLE = True
except Exception:
    VLLMClient = None  # type: ignore
    _TRL_VLLM_AVAILABLE = False

from accelerate import Accelerator, FullyShardedDataParallelPlugin

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        MixedPrecision,
        StateDictType,
        ShardedStateDictConfig,
    )
    from torch.distributed.fsdp.wrap import (
        transformer_auto_wrap_policy,
    )
    _FSDP_AVAILABLE = True
except ImportError:
    FSDP = None  # type: ignore
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
    def __init__(self, path: str, rank: int = 0, world_size: int = 1):
        self.path = path
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                # ランクごとの分割: 自分の担当以外の行はスキップ
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


_EXPERT_RE = re.compile(r"\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")


def parse_layer_expert_from_name(name: str) -> Optional[Tuple[int, int]]:
    m = _EXPERT_RE.search(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def load_mask(mask_json: Optional[str], n_layers: int, n_experts: int) -> Optional[List[torch.Tensor]]:
    if not mask_json:
        return None
    with open(mask_json, "r", encoding="utf-8") as f:
        obj = json.load(f)

    layer_masks = [torch.zeros(n_experts, dtype=torch.bool) for _ in range(n_layers)]

    sel = obj.get("selected_layer_experts")
    if sel:
        for it in sel:
            li = int(it["layer_idx"])
            ei = int(it["expert_id"])
            if 0 <= li < n_layers and 0 <= ei < n_experts:
                layer_masks[li][ei] = True
    else:
        ex = obj.get("selected_experts_unique") or obj.get("selected_experts") or obj.get("picked_experts")
        if not ex:
            raise RuntimeError("Mask JSON lacks selected_* fields.")
        for li in range(n_layers):
            for ei in ex:
                ei = int(ei)
                if 0 <= ei < n_experts:
                    layer_masks[li][ei] = True

    for li in range(n_layers):
        if layer_masks[li].sum().item() == 0:
            layer_masks[li] = torch.ones(n_experts, dtype=torch.bool)

    return layer_masks


def is_router_param_name(name: str) -> bool:
    lname = name.lower()
    if lname.endswith("mlp.gate.weight") or ".mlp.gate.weight" in lname:
        return True
    if "router" in lname and "router_logits" not in lname:
        return True
    if "routing" in lname:
        return True
    return False


def is_selected_expert_param(name: str, layer_masks: Optional[List[torch.Tensor]]) -> bool:
    if layer_masks is None:
        return False
    parsed = parse_layer_expert_from_name(name)
    if parsed is None:
        return False
    li, ei = parsed
    if li < 0 or li >= len(layer_masks):
        return False
    mask = layer_masks[li]
    if ei < 0 or ei >= mask.numel():
        return False
    return bool(mask[ei].item())


def mark_trainable_experts_only(model: nn.Module, layer_masks: Optional[List[torch.Tensor]]):
    total = 0
    trainable = 0
    for name, p in model.named_parameters():
        total += p.numel()
        if is_router_param_name(name):
            p.requires_grad = False
        elif is_selected_expert_param(name, layer_masks):
            p.requires_grad = True
            trainable += p.numel()
        else:
            p.requires_grad = False

    print(
        f"[INFO] trainable(selected-experts)={trainable:,} / total={total:,} "
        f"({trainable / max(1, total) * 100:.4f}%)",
        flush=True,
    )
    return [p for p in model.parameters() if p.requires_grad]


def is_attention_param_name(name: str, layer_start: int = 0, layer_end: int = -1, n_layers: int = 94) -> bool:
    """Attention関連のパラメータか判定（層範囲指定可能）"""
    lname = name.lower()
    
    # Exclude MLP/Expert-related first
    mlp_keywords = ['mlp', 'expert', 'gate', 'up_proj', 'down_proj', 'gate_proj']
    if any(kw in lname for kw in mlp_keywords):
        return False
    
    attention_keywords = [
        'self_attn',
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'q_a_proj', 'q_b_proj',  # Qwen3 MoE uses q_a_proj, q_b_proj for GQA
        'kv_a_proj_with_mqa', 'kv_b_proj',  # Qwen3 MoE KV projection
        'query', 'key', 'value',
        'attention',
    ]
    
    if not any(kw in lname for kw in attention_keywords):
        return False
    
    # Check layer number if specified
    # Pattern: .layers.XX. or model.layers.XX.
    layer_match = re.search(r'\.layers\.(\d+)\.', name)
    if layer_match:
        layer_idx = int(layer_match.group(1))
        # Handle layer_end=-1 as "all layers"
        actual_end = layer_end if layer_end >= 0 else n_layers
        if not (layer_start <= layer_idx < actual_end):
            return False
    
    return True


def mark_trainable_attention_only(model: nn.Module, layer_start: int = 0, layer_end: int = -1, n_layers: int = 94):
    """Attentionのみ訓練可能にする（層範囲指定可能）"""
    total = 0
    trainable = 0
    trainable_names = []
    for name, p in model.named_parameters():
        total += p.numel()
        if is_attention_param_name(name, layer_start, layer_end, n_layers):
            p.requires_grad = True
            trainable += p.numel()
            trainable_names.append(name)
        else:
            p.requires_grad = False

    # Describe layer range
    actual_end = layer_end if layer_end >= 0 else n_layers
    layer_desc = f"layers {layer_start}-{actual_end-1}" if layer_start > 0 or layer_end >= 0 else "all layers"
    
    print(
        f"[INFO] trainable(attention-only, {layer_desc})={trainable:,} / total={total:,} "
        f"({trainable / max(1, total) * 100:.4f}%)",
        flush=True,
    )
    # Print sample of trainable params
    if trainable_names:
        print(f"[INFO] sample trainable params: {trainable_names[:5]}", flush=True)
    return [p for p in model.parameters() if p.requires_grad]


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


# -------- HTTP session helpers --------

def _make_retrying_session(
    total_retries: int = 8,
    backoff_factor: float = 0.5,
    status_forcelist: Tuple[int, ...] = (408, 409, 425, 429, 500, 502, 503, 504),
    pool_connections: int = 128,
    pool_maxsize: int = 128,
) -> requests.Session:
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


# -------- Dropout disabled context manager --------

@contextmanager
def dropout_disabled(model: nn.Module):
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


class ExpertFullparamGRPOTrainer:
    def __init__(self, args):
        self.args = args
        self.wandb_run = None
        self._wandb_mod = None

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

        if int(os.environ.get("RANK", "0")) == 0:
            print("[INIT] Loading model config to detect layer class...", flush=True)

        temp_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        with torch.device("meta"):
            try:
                temp_model = AutoModelForCausalLM.from_config(temp_config, trust_remote_code=True)
            except Exception:
                temp_model = AutoModelForCausalLM.from_config(temp_config, trust_remote_code=True)

        if hasattr(temp_model, "model") and hasattr(temp_model.model, "layers"):
            layer_cls = temp_model.model.layers[0].__class__
        elif hasattr(temp_model, "layers"):
            layer_cls = temp_model.layers[0].__class__
        else:
            layer_cls = None

        del temp_model

        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[INIT] Detected Transformer Layer Class for Wrapping: {layer_cls}", flush=True)

        auto_wrap_policy = None
        if layer_cls is not None:
            auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={layer_cls},
            )

        fsdp_plugin = FullyShardedDataParallelPlugin(
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            cpu_ram_efficient_loading=True,
            sync_module_states=True,
            limit_all_gathers=True,
            auto_wrap_policy=auto_wrap_policy,
            activation_checkpointing=True,
            state_dict_type=StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
        )

        self.accelerator = Accelerator(
            mixed_precision="bf16",
            fsdp_plugin=fsdp_plugin,
        )

        self.rank = self.accelerator.process_index
        self.world_size = self.accelerator.num_processes
        self.device = self.accelerator.device
        our_device = self.device
        self.model_device = our_device

        local_rank = int(os.environ.get("LOCAL_RANK", str(self.accelerator.local_process_index)))
        self.local_rank = local_rank

        # CPU comm group (prefer gloo) for long object collectives
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

        # keep-alive + retry session for judge
        self._judge_sess = _make_retrying_session(
            total_retries=int(getattr(self.args, "judge_http_retries", 8)),
            backoff_factor=float(getattr(self.args, "judge_http_backoff", 0.5)),
        )
        self._judge_tls = threading.local()
        self._judge_tls.sess = self._judge_sess

        if self.accelerator.is_main_process:
            print(
                f"[DIST] rank={self.rank} / world_size={self.world_size} local_rank={self.local_rank} "
                f"(Accelerate + FSDP Embedded Config)",
                flush=True,
            )

        # ===== samples jsonl path =====
        self.samples_jsonl_path = str(getattr(self.args, "samples_jsonl_path", "samples.jsonl") or "samples.jsonl")
        if not os.path.isabs(self.samples_jsonl_path):
            self.samples_jsonl_path = os.path.join(str(self.args.save_dir), self.samples_jsonl_path)
        self.samples_jsonl_every = int(getattr(self.args, "samples_jsonl_every", 1) or 1)
        self.samples_jsonl_text_max_chars = int(getattr(self.args, "samples_jsonl_text_max_chars", 0) or 0)

        if self.accelerator.is_main_process:
            os.makedirs(os.path.dirname(self.samples_jsonl_path) or ".", exist_ok=True)
            if bool(getattr(self.args, "samples_jsonl_overwrite", False)):
                with open(self.samples_jsonl_path, "w", encoding="utf-8") as f:
                    f.write("")
            print(f"[SAMPLES] jsonl_path={self.samples_jsonl_path} every={self.samples_jsonl_every}", flush=True)

        self.use_vllm = bool(getattr(args, "use_vllm", False))
        self.vllm_client: Optional[Any] = None
        self.vllm_clients: List[Any] = []
        self.vllm_enabled: List[bool] = []

        if self.accelerator.is_main_process:
            print("[LOAD] actor:", args.model, flush=True)
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

        # Gradient Checkpointing + Frozen Embeddings 対策
        base_model.gradient_checkpointing_enable()
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        base_model.train()

        self.decoder = getattr(base_model, "model", None)
        if self.decoder is None and hasattr(base_model, "get_decoder"):
            try:
                self.decoder = base_model.get_decoder()
            except Exception:
                self.decoder = None

        self.layer_masks = None
        if args.mask_json:
            self.layer_masks = load_mask(
                args.mask_json,
                n_layers=args.layers,
                n_experts=args.experts,
            )
            if self.accelerator.is_main_process:
                allowed = int(sum(m.sum().item() for m in self.layer_masks))
                print(f"[INFO] loaded coverage-mask allow entries = {allowed}", flush=True)

        # Choose training mode: attention-only or expert-only
        self.train_attention_only = bool(getattr(args, "train_attention_only", False))
        if self.train_attention_only:
            attn_layer_start = int(getattr(args, "attention_layer_start", 0) or 0)
            attn_layer_end = int(getattr(args, "attention_layer_end", -1) if getattr(args, "attention_layer_end", -1) is not None else -1)
            if self.accelerator.is_main_process:
                print("[INFO] Training mode: ATTENTION-ONLY (FFN/Experts frozen)", flush=True)
            _ = mark_trainable_attention_only(
                base_model,
                layer_start=attn_layer_start,
                layer_end=attn_layer_end,
                n_layers=args.layers,
            )
        else:
            if self.accelerator.is_main_process:
                print("[INFO] Training mode: EXPERT-ONLY (Attention frozen)", flush=True)
            _ = mark_trainable_experts_only(base_model, self.layer_masks)

        # DataLoader Sharding
        self.ds = ChatMLJsonl(args.dataset, rank=self.rank, world_size=self.world_size)
        loader = DataLoader(
            self.ds,
            batch_size=args.batch_size,
            drop_last=True,  # ★変更: rank間でBがズレないようにする
        )

        # Prepare model only
        self.model = self.accelerator.prepare(base_model)
        self.loader = loader

        self.is_fsdp = isinstance(self.model, FSDP)

        try:
            unwrapped = self.accelerator.unwrap_model(self.model)
        except Exception:
            unwrapped = self.model
        self.decoder = getattr(unwrapped, "model", None)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            mode_str = "attention" if self.train_attention_only else "expert"
            raise RuntimeError(f"No trainable {mode_str} parameters found.")
        self.opt = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            betas=(0.9, 0.98),
            weight_decay=args.weight_decay,
        )

        self.pad_token_id = (
            self.tok.eos_token_id
            if self.tok.eos_token_id is not None
            else self.tok.pad_token_id
        )

        if self.accelerator.is_main_process:
            print("[INFO] judge backend:", args.judge_backend, flush=True)
            if args.judge_backend != "local":
                print(
                    f"[INFO] external judge model='{self.args.judge_model_name}' "
                    f"api_base='{self.args.judge_api_base}'",
                    flush=True,
                )

        # reward_cache を rank別ファイルにする
        if getattr(args, "reward_cache", None):
            base = args.reward_cache
            root, ext = os.path.splitext(base)
            ext = ext if ext else ".jsonl"
            args.reward_cache = f"{root}.rank{self.rank}{ext}"
        self.cache = ScoreCache(args.reward_cache)

        # smart_reward_cache も rank別ファイルにする
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

        if self.use_vllm:
            self._init_vllm_client()

        self.global_step = 0
        self.baseline: Optional[float] = None

        # NaN/Inf tracking for early abort
        self.consecutive_nan_count: int = 0
        self.max_consecutive_nan: int = int(getattr(self.args, "max_consecutive_nan", 10) or 10)

        # optimizer update tracking
        self.optimizer_updates: int = 0
        self.optimizer_updates_skipped: int = 0

        # progress / ETA tracking
        self.total_steps: int = int(getattr(self.args, "steps", 0) or 0)
        self.train_wall_start: Optional[float] = None
        self.step_time_ema: Optional[float] = None
        self.step_time_ema_alpha: float = 0.1

        # epoch progress tracking
        self.epoch_total: int = int(getattr(self.args, "epochs", 0) or 0)
        self.epoch_idx: int = 0
        self.steps_per_epoch: int = 0
        self.step_in_epoch: int = 0

        if self.args.wandb and self.accelerator.is_main_process:
            # Auto-generate wandb run name if not specified
            if self.args.wandb_run_name is None:
                reward_tag = str(getattr(self.args, "reward_mode", "grpo") or "grpo").lower()
                self.args.wandb_run_name = (
                    f"expert_{reward_tag}_lr{self.args.lr}_gs{self.args.grpo_group_size}"
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
            print("========== Expert Full-Param GRPO w/ LLM-as-judge ==========")
            print(f"actor model       : {self.args.model}")
            print(f"judge backend     : {self.args.judge_backend}")
            print(f"dataset           : {self.args.dataset}")
            print(f"layers/experts    : {self.args.layers}/{self.args.experts}")
            print(f"train_attention_only: {self.train_attention_only}")
            if self.train_attention_only:
                attn_start = int(getattr(self.args, "attention_layer_start", 0) or 0)
                attn_end = int(getattr(self.args, "attention_layer_end", -1) if getattr(self.args, "attention_layer_end", -1) is not None else -1)
                attn_end_str = str(attn_end) if attn_end >= 0 else f"{self.args.layers} (all)"
                print(f"attention_layers  : {attn_start} - {attn_end_str}")
            print(f"grpo_group_size   : {self.args.grpo_group_size}")
            print(f"use_vllm          : {self.args.use_vllm}")
            print(f"wandb             : {self.args.wandb}")
            print(f"use_smart_reward  : {self.use_smart_reward}")
            print(f"smart_reward_weight: {getattr(self.args,'smart_reward_weight',0.0)}")
            print(f"reward_mode       : {getattr(self.args,'reward_mode','grpo')}")
            print(f"centralize_actor_judge: {getattr(self.args,'centralize_actor_judge',False)}")
            print(f"wandb_table_rows  : {getattr(self.args,'wandb_table_rows',4)}")
            print(f"wandb_table_text_max: {getattr(self.args,'wandb_table_text_max_chars',2000)}")
            print("=======================================================================", flush=True)

        # ---- eval (temp=0) setup ----
        self.eval_every = int(getattr(self.args, "eval_every", 0))
        self.eval_size = int(getattr(self.args, "eval_size", 100) or 100)
        self.eval_prompts: List[str] = []
        self.eval_refs: List[str] = []

        if self.eval_every > 0 and self.accelerator.is_main_process:
            src = getattr(self.args, "eval_dataset", None) or self.args.dataset
            use_tail = (getattr(self.args, "eval_dataset", None) is None)

            if use_tail:
                from collections import deque
                tail_buffer: deque = deque(maxlen=self.eval_size)
                with open(src, "r", encoding="utf-8") as f:
                    for line in f:
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
                        p = extract_text_from_content(umsgs[0].get("content")).strip()
                        r = extract_text_from_content(amsgs[0].get("content")).strip() if amsgs else ""
                        if not p:
                            continue
                        tail_buffer.append((p, r))
                for p, r in tail_buffer:
                    self.eval_prompts.append(p)
                    self.eval_refs.append(r)
                print(f"[EVAL] loaded {len(self.eval_prompts)} prompts from TAIL of {src}", flush=True)
            else:
                with open(src, "r", encoding="utf-8") as f:
                    for line in f:
                        if len(self.eval_prompts) >= self.eval_size:
                            break
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
                        p = extract_text_from_content(umsgs[0].get("content")).strip()
                        r = extract_text_from_content(amsgs[0].get("content")).strip() if amsgs else ""
                        if not p:
                            continue
                        self.eval_prompts.append(p)
                        self.eval_refs.append(r)
                print(f"[EVAL] loaded {len(self.eval_prompts)} prompts from {src}", flush=True)

    def _comm_group(self):
        return self.cpu_group if self.cpu_group is not None else None

    def _get_thread_judge_session(self) -> requests.Session:
        s = getattr(self._judge_tls, "sess", None)
        if s is None:
            s = _make_retrying_session(
                total_retries=int(getattr(self.args, "judge_http_retries", 8)),
                backoff_factor=float(getattr(self.args, "judge_http_backoff", 0.5)),
            )
            self._judge_tls.sess = s
        return s

    @staticmethod
    def _fmt_hms(seconds: float) -> str:
        s = int(max(0, round(seconds)))
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:02d}"

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
        prompts: List[str],
        refs: List[str],
        candidate_groups: List[List[str]],
        final_scores_group: torch.Tensor,
        med_scores_group: torch.Tensor,
        smart_scores_group: torch.Tensor,
        med_judge_outputs_group: List[List[str]],
        smart_judge_outputs_group: List[List[str]],
    ) -> None:
        """Log samples table to wandb."""
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
            row: List[Any] = [int(self.optimizer_updates), self._clip_text(prompts[i]), self._clip_text(refs[i])]
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

        self.wandb_run.log({"samples/table": table}, step=int(self.optimizer_updates))

    def _write_samples_jsonl(
        self,
        prompts: List[str],
        refs: List[str],
        candidate_groups: List[List[str]],
        final_scores_group: torch.Tensor,
        med_scores_group: torch.Tensor,
        smart_scores_group: torch.Tensor,
        med_judge_outputs_group: List[List[str]],
        smart_judge_outputs_group: List[List[str]],
    ) -> None:
        """Write samples to jsonl file."""
        if not self.accelerator.is_main_process:
            return
        if self.global_step % self.samples_jsonl_every != 0:
            return

        group_size = int(self.args.grpo_group_size)

        with open(self.samples_jsonl_path, "a", encoding="utf-8") as f:
            for i, (p, r, cands) in enumerate(zip(prompts, refs, candidate_groups)):
                rec = {
                    "step": int(self.global_step),
                    "optimizer_updates": int(self.optimizer_updates),
                    "prompt": self._clip_text_jsonl(p),
                    "reference": self._clip_text_jsonl(r),
                    "candidates": [],
                }
                for k in range(group_size):
                    cand_rec = {
                        "text": self._clip_text_jsonl(cands[k] if k < len(cands) else ""),
                        "med_score": 0.0,
                        "smart_score": 0.0,
                        "final_score": 0.0,
                        "med_judge_output": "",
                        "smart_judge_output": "",
                    }
                    try:
                        cand_rec["med_score"] = float(med_scores_group[i, k].item())
                    except Exception:
                        pass
                    try:
                        cand_rec["smart_score"] = float(smart_scores_group[i, k].item())
                    except Exception:
                        pass
                    try:
                        cand_rec["final_score"] = float(final_scores_group[i, k].item())
                    except Exception:
                        pass
                    try:
                        cand_rec["med_judge_output"] = self._clip_text_jsonl(med_judge_outputs_group[i][k])
                    except Exception:
                        pass
                    try:
                        cand_rec["smart_judge_output"] = self._clip_text_jsonl(smart_judge_outputs_group[i][k])
                    except Exception:
                        pass
                    rec["candidates"].append(cand_rec)
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
        """Write samples to jsonl for centralized mode (rank0 writes ALL samples)."""
        if not self.accelerator.is_main_process:
            return

        group_size = int(self.args.grpo_group_size)

        with open(self.samples_jsonl_path, "a", encoding="utf-8") as f:
            ptr = 0
            for rank_idx, b in enumerate(per_rank_B):
                for local_i in range(b):
                    global_i = ptr + local_i
                    p = prompts_all[global_i]
                    r = refs_all[global_i]
                    cands = candidate_groups_all[global_i]

                    rec = {
                        "step": step,
                        "optimizer_updates": int(self.optimizer_updates),
                        "epoch": epoch,
                        "rank": rank_idx,
                        "prompt": self._clip_text_jsonl(p),
                        "reference": self._clip_text_jsonl(r),
                        "candidates": [],
                    }
                    for k in range(group_size):
                        cand_rec = {
                            "text": self._clip_text_jsonl(cands[k] if k < len(cands) else ""),
                            "med_score": float(med_scores_all[global_i, k].item()) if global_i < med_scores_all.shape[0] else 0.0,
                            "smart_score": float(smart_scores_all[global_i, k].item()) if global_i < smart_scores_all.shape[0] else 0.0,
                            "med_judge_output": "",
                            "smart_judge_output": "",
                        }
                        try:
                            cand_rec["med_judge_output"] = self._clip_text_jsonl(med_judge_outputs_all[global_i][k])
                        except Exception:
                            pass
                        try:
                            cand_rec["smart_judge_output"] = self._clip_text_jsonl(smart_judge_outputs_all[global_i][k])
                        except Exception:
                            pass
                        rec["candidates"].append(cand_rec)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                ptr += b

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

        # Single process fallback
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

        # Multi-GPU: use cpu_group for long object collectives
        comm = self._comm_group()
        if comm is None:
            raise RuntimeError(
                "[CENTRALIZE] cpu_group (gloo) is None; cannot run all_gather_object safely."
            )

        payload: Dict[str, Any] = {"prompt": local_prompts, "reference": local_refs}
        gathered: List[Any] = [None] * int(self.world_size)

        print(f"[rank{self.rank}][CENTRALIZE] entering all_gather_object with {len(local_prompts)} prompts", flush=True)

        t_gather = time.time()
        dist.all_gather_object(gathered, payload, group=comm)
        dt_gather = time.time() - t_gather

        if self.accelerator.is_main_process:
            print(f"[CENTRALIZE] all_gather_object done in {dt_gather:.2f}s", flush=True)

        rank0_med_judge_slice: List[List[str]] = []
        rank0_smart_judge_slice: List[List[str]] = []
        per_rank_pack: List[Dict[str, Any]] = []

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

            ptr = 0
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

            # rank0 writes ALL samples to jsonl before scatter
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

        # Scatter results to each rank
        recv: List[Any] = [None]
        if self.accelerator.is_main_process:
            print(f"[CENTRALIZE] rank0 entering scatter_object_list", flush=True)

        t_scatter = time.time()
        if hasattr(dist, "scatter_object_list"):
            if self.rank == 0:
                dist.scatter_object_list(recv, per_rank_pack, src=0, group=comm)
            else:
                dist.scatter_object_list(recv, None, src=0, group=comm)
            pack = recv[0]
        else:
            # Fallback: broadcast all and pick slice
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

        med_scores_group = torch.tensor(med_scores_group_list, dtype=torch.float32, device=self.model_device)
        smart_scores_group = torch.tensor(smart_scores_group_list, dtype=torch.float32, device=self.model_device)

        # Judge outputs: only rank0 has them, others get empty strings
        if self.rank == 0:
            med_judge_groups = rank0_med_judge_slice
            smart_judge_groups = rank0_smart_judge_slice
        else:
            b = len(cand_groups)
            med_judge_groups = [[""] * group_size for _ in range(b)]
            smart_judge_groups = [[""] * group_size for _ in range(b)]

        return cand_groups, med_scores_group, smart_scores_group, med_judge_groups, smart_judge_groups

    def _init_vllm_client(self):
        if not _TRL_VLLM_AVAILABLE:
            if self.accelerator.is_main_process:
                print("[WARN] TRL VLLMClient not available. Skipping vLLM init.", flush=True)
            return

        timeout = self.args.vllm_connection_timeout

        if self.accelerator.is_main_process:
            print("[VLLM] initializing VLLMClient(s)...", flush=True)

        urls: List[Optional[str]] = []
        if getattr(self.args, "vllm_base_urls", None):
            urls = [u.strip() for u in self.args.vllm_base_urls.split(",") if u.strip()]
        elif self.args.vllm_base_url:
            urls = [self.args.vllm_base_url]
        else:
            urls = [None]

        clients: List[Any] = []
        for u in urls:
            if u:
                c = VLLMClient(u, connection_timeout=timeout)
            else:
                c = VLLMClient(connection_timeout=timeout)
            c.init_communicator(self.device)
            clients.append(c)

        # 初回syncは全サーバへ（元スクリプトと同じ形式）
        for c in clients:
            try:
                if self.is_fsdp:
                    c.update_model_params_streaming_fsdp(self.model, trainable_only=True)
                else:
                    c.update_model_params(self.model, trainable_only=True)
            except Exception as e:
                if self.accelerator.is_main_process:
                    print(f"[WARN] VLLM initial sync failed: {e}", flush=True)

        self.vllm_clients = clients
        self.vllm_client = clients[0] if clients else None
        self.vllm_enabled = [True for _ in clients]  # Track enabled status

        if self.accelerator.is_main_process:
            print(f"[VLLM] ready clients={len(self.vllm_clients)}", flush=True)

    def _vllm_sync_backoff(self, attempt: int) -> float:
        """Exponential backoff with jitter for vLLM sync retries."""
        base = float(getattr(self.args, "vllm_sync_retry_backoff", 2.0) or 2.0)
        jitter = float(getattr(self.args, "vllm_sync_retry_jitter", 0.2) or 0.2)
        if attempt <= 1:
            return 0.0
        wait = base * (2 ** (attempt - 2))
        wait = wait * (1.0 + random.random() * jitter)
        return min(wait, 60.0)  # cap at 60s

    def _sync_one_vllm_with_retry(self, idx: int) -> Tuple[int, bool, Optional[str]]:
        """Sync model params to one vLLM server with retries."""
        if not self.vllm_enabled[idx]:
            return idx, False, "disabled"

        c = self.vllm_clients[idx]
        max_retries = int(getattr(self.args, "vllm_sync_max_retries", 3) or 3)

        last_err: Optional[str] = None
        for attempt in range(1, max_retries + 1):
            wait = self._vllm_sync_backoff(attempt)
            if wait > 0:
                time.sleep(wait)

            try:
                if self.is_fsdp:
                    c.update_model_params_streaming_fsdp(self.model, trainable_only=True)
                else:
                    c.update_model_params(self.model, trainable_only=True)
                return idx, True, None
            except Exception as e:
                last_err = str(e)
                if self.accelerator.is_main_process:
                    print(f"[VLLM][RETRY] client#{idx} attempt={attempt}/{max_retries} failed: {e}", flush=True)

        return idx, False, last_err

    def _sync_vllm_params_parallel(self, tag: str = ""):
        """Sync model params to multiple vLLM servers in parallel with retries."""
        if not self.vllm_clients:
            return

        enabled_idxs = [i for i, en in enumerate(self.vllm_enabled) if en]
        if not enabled_idxs:
            if self.accelerator.is_main_process:
                print(f"[WARN][VLLM] No enabled clients for sync {tag}", flush=True)
            return

        t0 = time.time()
        ok_count = 0
        failed: List[Tuple[int, str]] = []

        max_workers = min(32, len(enabled_idxs))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self._sync_one_vllm_with_retry, i): i for i in enabled_idxs}
            for fut in as_completed(futures):
                idx, success, err = fut.result()
                if success:
                    ok_count += 1
                else:
                    failed.append((idx, err or "unknown"))

        # Disable failed servers
        for idx, err in failed:
            self.vllm_enabled[idx] = False
            if self.accelerator.is_main_process:
                print(f"[WARN][VLLM] Disabled client#{idx} after sync failure: {err}", flush=True)

        dt = time.time() - t0
        if self.accelerator.is_main_process:
            enabled_cnt = sum(1 for x in self.vllm_enabled if x)
            print(f"[VLLM] sync {tag}: {ok_count}/{len(enabled_idxs)} ok in {dt:.2f}s (enabled={enabled_cnt}/{len(self.vllm_clients)})", flush=True)

        # Check if all servers are down
        if enabled_cnt == 0:
            raise RuntimeError("[VLLM] All servers disabled after sync failure. Abort.")

    @torch.no_grad()
    def _generate_with_vllm_grpo(self, prompts: List[str], group_size: int) -> List[List[str]]:
        assert self.vllm_client is not None or self.vllm_clients
        clients = self.vllm_clients if self.vllm_clients else [self.vllm_client]  # type: ignore

        orig_B = len(prompts)
        force_n1 = bool(getattr(self.args, "vllm_force_n1", False))

        call_prompts = prompts
        call_n = int(group_size)
        call_group_size = int(group_size)

        if force_n1:
            expanded: List[str] = []
            for p in prompts:
                expanded.extend([p] * int(group_size))
            call_prompts = expanded
            call_n = 1
            call_group_size = 1

        # === FIX: apply chat template for vLLM prompts (to avoid immediate special-token-only outputs) ===
        if hasattr(self.tok, "apply_chat_template") and getattr(self.tok, "chat_template", None):
            call_prompts = [
                self.tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for p in call_prompts
            ]

        gen_kwargs = {
            "max_tokens": int(self.args.gen_max_new_tokens),
            "repetition_penalty": float(self.args.gen_repetition_penalty),
            "n": int(call_n),
            "temperature": float(self.args.gen_temperature),
            "top_p": float(self.args.gen_top_p),
            "min_p": 0.0,
        }

        def _decode_one(resp, local_prompts: List[str], local_group_size: int, debug_tag: str = "") -> List[List[str]]:
            if self.args.debug_print and self.accelerator.is_main_process:
                try:
                    print(f"========== VLLM RAW RESPONSE STRUCT {debug_tag} ==========", flush=True)
                    print(f"[vllm] type(resp)={type(resp)}", flush=True)
                    if isinstance(resp, dict):
                        print(f"[vllm] keys={list(resp.keys())}", flush=True)
                        for k, v in resp.items():
                            try:
                                ln = len(v)
                            except Exception:
                                ln = "N/A"
                            print(f"[vllm]  - {k}: type={type(v)}, len={ln}", flush=True)
                            if isinstance(v, list) and len(v) > 0:
                                print(f"[vllm]    first_elem_type={type(v[0])}", flush=True)
                    else:
                        try:
                            print(f"[vllm] len(resp)={len(resp)}", flush=True)
                            if len(resp) > 0:
                                print(f"[vllm] first_elem_type={type(resp[0])}", flush=True)
                        except Exception:
                            pass
                    print("=============================================", flush=True)
                except Exception as e:
                    print(f"[WARN] debug_print vllm struct failed: {e}", flush=True)

            token_seqs = resp.get("completion_ids", resp) if isinstance(resp, dict) else resp

            if self.args.debug_print and self.accelerator.is_main_process:
                try:
                    comp = None
                    if isinstance(resp, dict) and "completion_ids" in resp:
                        comp = resp["completion_ids"]

                    print(f"========== VLLM COMPLETION_IDS HEAD {debug_tag} ==========", flush=True)
                    if comp is None:
                        print("[vllm] completion_ids not found in resp dict.", flush=True)
                    else:
                        if isinstance(comp, list) and comp:
                            if isinstance(comp[0], list) and (len(comp[0]) == 0 or isinstance(comp[0][0], int)):
                                show_n = min(len(comp), 8)
                                for i in range(show_n):
                                    head = comp[i][:10] if isinstance(comp[i], list) else comp[i]
                                    print(f"[vllm] completion_ids[{i}] head={head}", flush=True)
                            elif isinstance(comp[0], list) and comp[0] and isinstance(comp[0][0], list):
                                show_b = min(len(comp), 2)
                                for b in range(show_b):
                                    inner = comp[b]
                                    show_n = min(len(inner), 4)
                                    for i in range(show_n):
                                        head = inner[i][:10] if isinstance(inner[i], list) else inner[i]
                                        print(f"[vllm] completion_ids[B{b}][{i}] head={head}", flush=True)
                            else:
                                print(f"[vllm] completion_ids unexpected shape; first_elem_type={type(comp[0])}", flush=True)
                        else:
                            print("[vllm] completion_ids empty or not a list.", flush=True)
                    print("==============================================", flush=True)
                except Exception as e:
                    print(f"[WARN] vLLM completion_ids head debug failed: {e}", flush=True)

            outs_flat: List[str] = []

            def _to_list(x):
                if isinstance(x, torch.Tensor):
                    return x.tolist()
                return x

            def _append_decoded_ids(ids):
                ids = _to_list(ids)
                if not ids:
                    outs_flat.append("")
                    return

                if isinstance(ids, list) and ids and isinstance(ids[0], list):
                    for sub in ids:
                        sub = _to_list(sub)
                        try:
                            txt = self.tok.decode(sub, skip_special_tokens=True)
                        except Exception:
                            txt = ""
                        outs_flat.append((txt or "").strip())
                    return

                try:
                    txt = self.tok.decode(ids, skip_special_tokens=True)
                except Exception:
                    txt = ""
                outs_flat.append((txt or "").strip())

            def _handle_seq(seq):
                seq = _to_list(seq)

                if seq is None:
                    outs_flat.append("")
                    return

                if isinstance(seq, dict):
                    text = None
                    ids = None

                    for k in [
                        "output_token_ids",
                        "token_ids",
                        "tokens",
                        "output_ids",
                        "generated_token_ids",
                        "completion_ids",
                    ]:
                        if k in seq:
                            ids = seq[k]
                            break

                    if "text" in seq:
                        text = seq["text"]
                    if "generated_text" in seq:
                        text = seq["generated_text"]

                    if ids is not None:
                        _append_decoded_ids(ids)
                    else:
                        outs_flat.append((text or "").strip())
                    return

                if isinstance(seq, list) and seq and isinstance(seq[0], dict):
                    for sub in seq:
                        _handle_seq(sub)
                    return

                if isinstance(seq, list) and seq and isinstance(seq[0], list):
                    for sub in seq:
                        _handle_seq(sub)
                    return

                _append_decoded_ids(seq)

            _handle_seq(token_seqs)

            B = len(local_prompts)
            expected = B * int(call_n)

            if self.args.debug_print and self.accelerator.is_main_process:
                try:
                    print(f"========== VLLM DECODED OUTS_FLAT {debug_tag} ==========", flush=True)
                    print(f"[vllm] outs_flat_len={len(outs_flat)} expected={expected}", flush=True)
                    show = min(len(outs_flat), 8)
                    for i in range(show):
                        t = outs_flat[i] or ""
                        t_short = t.replace("\n", "\\n")
                        if len(t_short) > 160:
                            t_short = t_short[:160] + "..."
                        print(f"[vllm] outs_flat[{i}]='{t_short}'", flush=True)
                    print("============================================", flush=True)
                except Exception as e:
                    print(f"[WARN] vLLM outs_flat debug failed: {e}", flush=True)

            limit = min(len(outs_flat), expected)
            outs_flat = outs_flat[:limit]

            grouped: List[List[str]] = []
            idx = 0
            for _ in range(B):
                group = outs_flat[idx: idx + int(call_n)]
                if len(group) < int(call_n):
                    group = group + [""] * (int(call_n) - len(group))
                grouped.append(group)
                idx += int(call_n)

            return grouped

        if len(clients) == 1:
            resp = clients[0].generate(call_prompts, **gen_kwargs)
            grouped = _decode_one(resp, call_prompts, call_group_size)
            if not force_n1:
                return grouped
            out: List[List[str]] = []
            ptr = 0
            for _ in range(orig_B):
                g: List[str] = []
                for _ in range(int(group_size)):
                    g.append((grouped[ptr][0] if grouped[ptr] else "").strip())
                    ptr += 1
                out.append(g)
            return out

        K = len(clients)
        buckets: List[List[str]] = [[] for _ in range(K)]
        idxs: List[List[int]] = [[] for _ in range(K)]
        for i, p in enumerate(call_prompts):
            k = i % K
            buckets[k].append(p)
            idxs[k].append(i)

        def _run(k: int):
            if not buckets[k]:
                return k, []
            resp_k = clients[k].generate(buckets[k], **gen_kwargs)
            tag = f"(server#{k})"
            grouped_k = _decode_one(resp_k, buckets[k], call_group_size, debug_tag=tag)
            return k, grouped_k

        per_prompt_out: List[Optional[List[str]]] = [None] * len(call_prompts)
        with ThreadPoolExecutor(max_workers=K) as ex:
            for k, grouped_k in ex.map(_run, range(K)):
                for pos, g in zip(idxs[k], grouped_k):
                    per_prompt_out[pos] = g

        for i in range(len(per_prompt_out)):
            if per_prompt_out[i] is None:
                per_prompt_out[i] = [""] * int(call_n)

        if not force_n1:
            return per_prompt_out  # type: ignore

        out: List[List[str]] = []
        ptr = 0
        for _ in range(orig_B):
            g: List[str] = []
            for _ in range(int(group_size)):
                g.append((per_prompt_out[ptr][0] if per_prompt_out[ptr] else "").strip())
                ptr += 1
            out.append(g)
        return out

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
                synced_gpus=True,  # ★追加
            )
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask.repeat(group_size, 1)

            gen_ids = self.model.generate(**gen_kwargs)
            for k in range(group_size):
                g = gen_ids[k]
                generated = g[input_ids.shape[-1]:]
                txt = self.tok.decode(generated, skip_special_tokens=True)
                group.append(txt.strip())
            outs_grouped.append(group)
        self.model.train()
        return outs_grouped

    @torch.no_grad()
    def generate_candidates_grpo(self, prompts: List[str]) -> List[List[str]]:
        group_size = int(self.args.grpo_group_size)
        if group_size <= 0:
            raise ValueError("--grpo_group_size must be >= 1")
        if self.vllm_client is not None or self.vllm_clients:
            return self._generate_with_vllm_grpo(prompts, group_size)
        return self._generate_local_grpo(prompts, group_size)

    # ===== External judge call (generic with fallback) =====
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
        if "+" in backend:
            backends = [b.strip() for b in backend.split("+") if b.strip()]
        else:
            backends = [backend]

        last_result = ""
        for i, be in enumerate(backends):
            be_api_base = api_base
            be_api_key = api_key
            be_model = model_name

            if i > 0:
                be_api_base = None
                be_api_key = None
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
                max_attempts=max_attempts if i == 0 else max(1, max_attempts // 2),
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
            "User-Agent": "expert_fullparam_grpo/1.0",
        }
        if backend == "openrouter":
            headers.setdefault("HTTP-Referer", os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"))
            headers.setdefault("X-Title", os.getenv("OPENROUTER_X_TITLE", "expert_fullparam_grpo"))

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            "temperature": float(temperature),
        }
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

                if resp.status_code in (400, 401, 403, 404):
                    body_snip = (resp.text or "")[:400].replace("\n", "\\n")
                    if self.accelerator.is_main_process:
                        print(f"[WARN] external HTTP {resp.status_code} (no retry, fail to fallback): '{body_snip}'", flush=True)
                    return ""

                if resp.status_code in (429, 500, 502, 503, 504):
                    ra = resp.headers.get("Retry-After")
                    if ra is not None:
                        try:
                            wait = float(ra)
                        except Exception:
                            wait = None
                    else:
                        wait = min(base_sleep * (2 ** (attempt - 1)), max_sleep)
                    if wait is not None:
                        time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = str(e)
                if self.accelerator.is_main_process:
                    print(f"[WARN] external judge failed attempt {attempt}/{max_attempts}: {e}", flush=True)
                wait = min(base_sleep * (2 ** (attempt - 1)), max_sleep)
                time.sleep(wait)
        return ""

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
            self.model.eval()
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
                            print(f"[ERROR][MED_JUDGE] idx={idx} failed: {e}", flush=True)
                        s, jo = 0.0, ""
                scores.append(s)
                judge_outputs.append(jo)
            self.model.train()
            dt = time.time() - t0
            if self.accelerator.is_main_process:
                print(f"[MED_JUDGE] done: {N} samples in {dt:.2f}s, errors={error_count}", flush=True)
            return torch.tensor(scores, dtype=torch.float32, device=self.model_device), judge_outputs

        # Parallel execution
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

        # Parse SMART score [0]..[10]
        matches = re.findall(r"\[(\d+)\]", gen_txt)
        if matches:
            try:
                raw_val = int(matches[-1])
            except Exception:
                raw_val = 0
        else:
            matches = re.findall(r"\b(\d+)\b", gen_txt)
            if matches:
                try:
                    raw_val = int(matches[-1])
                except Exception:
                    raw_val = 0
            else:
                raw_val = 0

        val = max(0, min(10, raw_val))
        return float(val), gen_txt

    @torch.no_grad()
    def smart_scores_flat(
        self,
        prompts_flat: List[str],
        cands_flat: List[str],
        refs_flat: List[str],
    ) -> Tuple[torch.Tensor, List[str]]:
        if not self.use_smart_reward:
            N = len(prompts_flat)
            return torch.zeros(N, dtype=torch.float32, device=self.model_device), [""] * N

        backend = str(self.args.smart_backend)
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
                            print(f"[ERROR][SMART_JUDGE] idx={idx} failed: {e}", flush=True)
                        s, jo = 0.0, ""
                scores.append(s)
                judge_outputs.append(jo)
            dt = time.time() - t0
            if self.accelerator.is_main_process:
                print(f"[SMART_JUDGE] done: {N} samples in {dt:.2f}s, errors={error_count}", flush=True)
            return torch.tensor(scores, dtype=torch.float32, device=self.model_device), judge_outputs

        # Parallel execution
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

    def _combine_med_smart(self, med: torch.Tensor, smart: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Combine medical and SMART scores with gating and hack penalty."""
        smart_max = float(getattr(self.args, "smart_max_score", 10.0) or 10.0)
        smart_weight = float(getattr(self.args, "smart_reward_weight", 0.0) or 0.0)
        smart_norm = smart / max(1e-6, smart_max)
        smart_add = smart_norm * smart_weight

        gate_mode = str(getattr(self.args, "smart_gate_mode", "soft") or "soft")
        gate_th = float(getattr(self.args, "smart_gate_threshold", 0.5) or 0.5)
        if gate_mode == "soft":
            smart_add = smart_add * med
        elif gate_mode == "hard":
            smart_add = smart_add * (med >= gate_th).float()

        smart_cap = float(getattr(self.args, "smart_reward_cap", 0.2) or 0.2)
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

    def _build_policy_batch(self, prompts_flat: List[str], cands_flat: List[str]):
        input_ids_list = []
        labels_list = []

        if hasattr(self.tok, "apply_chat_template") and getattr(self.tok, "chat_template", None):
            for p, c in zip(prompts_flat, cands_flat):
                msgs_user = [{"role": "user", "content": p}]
                msgs_full = [{"role": "user", "content": p}, {"role": "assistant", "content": c}]
                enc_user = self.tok.apply_chat_template(msgs_user, add_generation_prompt=True, tokenize=True, return_tensors="pt")
                enc_full = self.tok.apply_chat_template(msgs_full, add_generation_prompt=False, tokenize=True, return_tensors="pt")

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

    def step(self, batch):
        t_step0 = time.time()
        current_step_num = self.global_step + 1

        prompts: List[str] = batch["prompt"]
        refs: List[str] = batch["reference"]
        B = len(prompts)
        if B == 0:
            return
        group_size = int(self.args.grpo_group_size)

        if self.accelerator.is_main_process:
            print(f"[STEP] ===== start step={current_step_num} B={B} =====", flush=True)

        # Determine reward mode
        reward_mode = str(getattr(self.args, "reward_mode", "grpo") or "grpo").lower()

        # Candidate generation + judge
        if bool(getattr(self.args, "centralize_actor_judge", False)):
            # Centralized: rank0 generates and judges all, then distributes
            candidate_groups, med_scores_group, smart_scores_group, med_judge_outputs_group, smart_judge_outputs_group = \
                self._centralized_generate_and_score(prompts, refs)

            # Flatten for policy batch
            prompts_flat, cands_flat = [], []
            for p, group_c in zip(prompts, candidate_groups):
                for c in group_c:
                    prompts_flat.append(p)
                    cands_flat.append(c)

            med_scores_flat = med_scores_group.contiguous().view(-1)
            smart_scores_flat = smart_scores_group.contiguous().view(-1)
        else:
            # Normal: each rank generates and judges independently
            candidate_groups = self.generate_candidates_grpo(prompts)

            if self.args.debug_print and self.accelerator.is_main_process:
                for i, (p, r, group) in enumerate(zip(prompts, refs, candidate_groups)):
                    print("========== SAMPLE-GRPO ==========", flush=True)
                    print(f"[idx {i}] PROMPT   :", p, flush=True)
                    print(f"[idx {i}] REFERENCE:", r, flush=True)
                    for j, c in enumerate(group):
                        print(f"  [cand {j}] {c}", flush=True)
                    print("================================", flush=True)

            prompts_flat, refs_flat, cands_flat = [], [], []
            for p, r, group in zip(prompts, refs, candidate_groups):
                for c in group:
                    prompts_flat.append(p)
                    refs_flat.append(r)
                    cands_flat.append(c)

            # Medical scores
            med_scores_flat, med_judge_outputs_flat = self.judge_scores_flat(prompts_flat, cands_flat, refs_flat)
            # SMART scores
            smart_scores_flat, smart_judge_outputs_flat = self.smart_scores_flat(prompts_flat, cands_flat, refs_flat)

            med_scores_group = med_scores_flat.view(B, group_size)
            smart_scores_group = smart_scores_flat.view(B, group_size)

            # Reshape judge outputs to group
            med_judge_outputs_group: List[List[str]] = []
            smart_judge_outputs_group: List[List[str]] = []
            for i in range(B):
                med_judge_outputs_group.append(med_judge_outputs_flat[i * group_size:(i + 1) * group_size])
                smart_judge_outputs_group.append(smart_judge_outputs_flat[i * group_size:(i + 1) * group_size])

        # Combine med + smart -> final scores
        final_scores_flat, smart_add_flat, hack_mask_flat = self._combine_med_smart(
            med_scores_group.contiguous().view(-1),
            smart_scores_group.contiguous().view(-1)
        )
        final_scores_group = final_scores_flat.view(B, group_size)

        mean_score = final_scores_flat.mean().item()
        if self.baseline is None:
            self.baseline = mean_score
        else:
            self.baseline = (1 - self.args.ema_alpha) * self.baseline + self.args.ema_alpha * mean_score

        mean_group = final_scores_group.mean(dim=1, keepdim=True)
        std_group = final_scores_group.std(dim=1, keepdim=True, unbiased=False)
        gdpo_eps = float(getattr(self.args, "gdpo_eps", 1e-6) or 1e-6)

        if reward_mode == "gdpo":
            # GDPO: Decoupled normalization
            med_g = med_scores_group.to(dtype=torch.float32)
            smart_g = smart_scores_group.to(dtype=torch.float32)
            smart_max = float(getattr(self.args, "smart_max_score", 10.0) or 10.0)
            smart_g = smart_g / max(1e-6, smart_max)

            med_mean = med_g.mean(dim=1, keepdim=True)
            med_std = med_g.std(dim=1, keepdim=True, unbiased=False)
            z_med = (med_g - med_mean) / (med_std + gdpo_eps)

            sm_mean = smart_g.mean(dim=1, keepdim=True)
            sm_std = smart_g.std(dim=1, keepdim=True, unbiased=False)
            z_smart = (smart_g - sm_mean) / (sm_std + gdpo_eps)

            w_smart = float(getattr(self.args, "smart_reward_weight", 0.0) or 0.0)
            adv_tilde = z_med + w_smart * z_smart

            if bool(getattr(self.args, "gdpo_whiten_global", True)):
                flat = adv_tilde.view(-1)
                g_mean = flat.mean().item()
                g_std = flat.std().item()
                adv_group = (adv_tilde - g_mean) / (g_std + gdpo_eps)
            else:
                adv_group = adv_tilde
        else:
            # Classic GRPO: group normalize on final reward
            adv_group = (final_scores_group - mean_group) / (std_group + 1e-6)

        if self.args.reward_scale is not None:
            adv_group = adv_group * self.args.reward_scale
        if self.args.reward_clip > 0:
            adv_group = adv_group.clamp(-self.args.reward_clip, self.args.reward_clip)
        advantage = adv_group.view(-1)

        input_ids, attention_mask, labels = self._build_policy_batch(prompts_flat, cands_flat)

        # Determine inner epochs
        if reward_mode in ("grpo", "gdpo"):
            inner_epochs = int(getattr(self.args, "grpo_epochs", 1) or 1)
        elif reward_mode == "gspo":
            inner_epochs = int(getattr(self.args, "gspo_epochs", 4) or 4)
        else:
            inner_epochs = 1

        # Initial forward pass
        self.model.train()
        with dropout_disabled(self.model):
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_router_logits=False,
                return_dict=True,
            )

        # Compute log probabilities
        logits = out.logits[:, :-1, :]
        labels_shifted = labels[:, 1:]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        valid_mask = (labels_shifted != -100).float()
        safe_labels = labels_shifted.clone()
        safe_labels[safe_labels == -100] = 0
        token_logp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        token_logp = token_logp * valid_mask
        lengths = valid_mask.sum(dim=-1).clamp_min(1.0)
        logp_mean = token_logp.sum(dim=-1) / lengths

        # For PPO-clip
        logp_old_mean = logp_mean.detach().clone()
        token_logp_old = token_logp.detach().clone()

        for inner_epoch in range(inner_epochs):
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
                logits = out.logits[:, :-1, :]
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                token_logp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
                token_logp = token_logp * valid_mask
                logp_mean = token_logp.sum(dim=-1) / lengths

            if reward_mode in ("grpo", "gdpo"):
                # PPO-clip on token-level
                grpo_clip_eps = float(getattr(self.args, "grpo_clip_eps", 0.2) or 0.2)
                grpo_logratio_clip = float(getattr(self.args, "grpo_logratio_clip", 20.0) or 20.0)

                log_ratio = (token_logp - token_logp_old).clamp(-grpo_logratio_clip, grpo_logratio_clip)
                ratio = torch.exp(log_ratio)
                clipped_ratio = ratio.clamp(1.0 - grpo_clip_eps, 1.0 + grpo_clip_eps)

                adv_tok = advantage.detach().unsqueeze(-1)
                surr1 = ratio * adv_tok
                surr2 = clipped_ratio * adv_tok
                surr = torch.min(surr1, surr2)

                loss_per_sample = -((surr * valid_mask).sum(dim=-1) / lengths)
                loss = loss_per_sample.mean()

            elif reward_mode == "gspo":
                # GSPO: sequence-level PPO-clip
                gspo_logratio_clip = float(getattr(self.args, "gspo_logratio_clip", 20.0) or 20.0)
                log_ratio = (logp_mean - logp_old_mean).clamp(-gspo_logratio_clip, gspo_logratio_clip)
                ratio = torch.exp(log_ratio)

                gspo_eps_low = float(getattr(self.args, "gspo_clip_eps_low", 3e-4) or 3e-4)
                gspo_eps_high = float(getattr(self.args, "gspo_clip_eps_high", 4e-4) or 4e-4)
                clipped_ratio = ratio.clamp(1.0 - gspo_eps_low, 1.0 + gspo_eps_high)

                surr1 = ratio * advantage.detach()
                surr2 = clipped_ratio * advantage.detach()
                loss = -torch.min(surr1, surr2).mean()
            else:
                # Simple REINFORCE-style
                loss = -(advantage * logp_mean).mean()

            # ===== NaN/Inf チェック（全rank同期で判定）=====
            # NOTE: rank-local skip (continue) would desync FSDP collectives
            # (clip_grad_norm_/opt.step), so the decision must be identical on all ranks.
            loss_bad_local = torch.tensor(
                1 if (torch.isnan(loss) or torch.isinf(loss)) else 0,
                device=self.model_device,
                dtype=torch.int32,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(loss_bad_local, op=dist.ReduceOp.SUM)
            skip_update = (loss_bad_local.item() > 0)

            if skip_update:
                # Replace loss with a graph-connected zero so backward still runs
                # and FSDP communication stays in sync across ranks.
                safe_logp_sum = torch.nan_to_num(
                    logp_mean, nan=0.0, posinf=0.0, neginf=0.0
                ).sum()
                loss = safe_logp_sum * 0.0
                self.consecutive_nan_count += 1
                self.optimizer_updates_skipped += 1
                print(f"[rank{self.rank}][WARN] NaN/Inf loss detected (count={self.consecutive_nan_count}), skipping update", flush=True)
                if self.consecutive_nan_count >= self.max_consecutive_nan:
                    raise RuntimeError(f"[ABORT] {self.consecutive_nan_count} consecutive NaN/Inf losses")
            else:
                self.consecutive_nan_count = 0

            self.opt.zero_grad(set_to_none=True)
            self.accelerator.backward(loss)

            if not skip_update:
                if self.args.grad_clip > 0:
                    self.accelerator.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.args.grad_clip,
                    )
                self.opt.step()
                self.optimizer_updates += 1

        current_step = self.global_step + 1

        # vLLM sync（元スクリプトと同じ形式）
        if self.vllm_clients:
            if current_step % int(getattr(self.args, "vllm_update_every", 1)) == 0:
                for c in self.vllm_clients:
                    try:
                        if self.is_fsdp:
                            c.update_model_params_streaming_fsdp(self.model, trainable_only=True)
                        else:
                            c.update_model_params(self.model, trainable_only=True)
                    except Exception as e:
                        if self.accelerator.is_main_process:
                            print(f"[WARN] VLLM update_model_params failed: {e}", flush=True)

        # Compute diagnostic metrics
        # Tie fraction (groups with zero std)
        final_group_std = final_scores_group.std(dim=1, unbiased=False)
        med_group_std = med_scores_group.std(dim=1, unbiased=False)
        tie_final_frac = float((final_group_std < 1e-6).float().mean().item())
        tie_med_frac = float((med_group_std < 1e-6).float().mean().item())

        # Best score in each group
        best_final_per_group = final_scores_group.max(dim=1).values
        best_med_per_group = med_scores_group.max(dim=1).values
        best_final_mean = float(best_final_per_group.mean().item())
        best_final_std = float(best_final_per_group.std().item())
        best_med_mean = float(best_med_per_group.mean().item())
        best_med_std = float(best_med_per_group.std().item())

        # Mean per prompt
        mean_final_pp = final_scores_group.mean(dim=1)
        mean_med_pp = med_scores_group.mean(dim=1)
        mean_final_pp_mean = float(mean_final_pp.mean().item())
        mean_med_pp_mean = float(mean_med_pp.mean().item())

        # smart_add, hack stats
        smart_add_mean = float(smart_add_flat.mean().item())
        smart_add_std = float(smart_add_flat.std().item())
        hack_frac = float(hack_mask_flat.mean().item())

        # Progress / time
        done = int(self.optimizer_updates)
        total = int(self.total_steps) if self.total_steps > 0 else done
        pct = 100.0 * done / max(1, total)

        # Step time EMA
        now = time.time()
        if self.train_wall_start is None:
            self.train_wall_start = now
        elapsed_s = now - self.train_wall_start
        step_s = elapsed_s / max(1, done)
        if self.step_time_ema is None:
            self.step_time_ema = step_s
        else:
            self.step_time_ema = (1 - self.step_time_ema_alpha) * self.step_time_ema + self.step_time_ema_alpha * step_s
        eta_s = self.step_time_ema * max(0, total - done) if self.step_time_ema else 0.0
        finish_dt = datetime.now() + timedelta(seconds=eta_s)

        # Epoch progress string
        ep_str = ""
        if self.epoch_total > 0:
            ep_str = f"[epoch {self.epoch_idx+1}/{self.epoch_total} step {self.step_in_epoch}/{self.steps_per_epoch}] "

        # Logging
        if current_step % self.args.log_every == 0 and self.accelerator.is_main_process:
            global_mean_final = float(final_scores_flat.mean().item())
            global_std_final = float(final_scores_flat.std().item())
            med_mean = float(med_scores_flat.mean().item())
            med_std = float(med_scores_flat.std().item())
            smart_mean = float(smart_scores_flat.mean().item())
            smart_std = float(smart_scores_flat.std().item())

            mode_tag = reward_mode.upper()
            print(
                f"{ep_str}[step {done:6d}/{total} ({pct:5.1f}%)] {mode_tag} loss={loss.item():.4f} "
                f"final(mean±std)={global_mean_final:.3f}±{global_std_final:.3f} "
                f"med(mean±std)={med_mean:.3f}±{med_std:.3f} "
                f"smart(mean±std)={smart_mean:.3f}±{smart_std:.3f} "
                f"smart_add(mean)={smart_add_mean:.3f} hack_frac={hack_frac:.3f} "
                f"baseline(global-EMA)={self.baseline:.3f} | "
                f"elapsed {self._fmt_hms(elapsed_s)} "
                f"eta {self._fmt_hms(eta_s)} "
                f"finish {finish_dt.strftime('%Y-%m-%d %H:%M:%S')}",
                flush=True,
            )

        if self.wandb_run is not None and self.accelerator.is_main_process:
            try:
                global_mean_final = float(final_scores_flat.mean().item())
                global_std_final = float(final_scores_flat.std().item())
                med_mean = float(med_scores_flat.mean().item())
                med_std = float(med_scores_flat.std().item())
                smart_mean = float(smart_scores_flat.mean().item())
                smart_std = float(smart_scores_flat.std().item())

                log_dict = {
                    "train/loss": float(loss.item()),
                    "train/reward_mode": reward_mode,
                    "reward/final_mean": global_mean_final,
                    "reward/final_std": global_std_final,
                    "reward/med_mean": med_mean,
                    "reward/med_std": med_std,
                    "reward/smart_mean": smart_mean,
                    "reward/smart_std": smart_std,
                    "reward/smart_add_mean": smart_add_mean,
                    "reward/smart_add_std": smart_add_std,
                    "reward/hack_frac": hack_frac,
                    "reward/baseline_ema": float(self.baseline),
                    "opt/lr": float(self.opt.param_groups[0]["lr"]),
                    "progress/step": done,
                    "progress/total_steps": total,
                    "progress/pct": pct,
                    "progress/optimizer_updates": int(self.optimizer_updates),
                    "progress/optimizer_updates_skipped": int(self.optimizer_updates_skipped),
                    "time/step_s_ema": float(self.step_time_ema or step_s),
                    "time/elapsed_s": float(elapsed_s),
                    "time/eta_s": float(eta_s),
                    "epoch/idx": int(self.epoch_idx),
                    "epoch/total": int(self.epoch_total),
                    "epoch/step": int(self.step_in_epoch),
                    "epoch/steps_per_epoch": int(self.steps_per_epoch),
                    # Diagnostic metrics
                    "diag/tie_final_frac": tie_final_frac,
                    "diag/tie_med_frac": tie_med_frac,
                    "diag/best_final_mean": best_final_mean,
                    "diag/best_final_std": best_final_std,
                    "diag/best_med_mean": best_med_mean,
                    "diag/best_med_std": best_med_std,
                    "diag/mean_final_per_prompt": mean_final_pp_mean,
                    "diag/mean_med_per_prompt": mean_med_pp_mean,
                }

                self.wandb_run.log(log_dict, step=int(self.optimizer_updates))

                # Samples table
                self._wandb_log_samples_table(
                    prompts=prompts,
                    refs=refs,
                    candidate_groups=candidate_groups,
                    final_scores_group=final_scores_group,
                    med_scores_group=med_scores_group,
                    smart_scores_group=smart_scores_group,
                    med_judge_outputs_group=med_judge_outputs_group,
                    smart_judge_outputs_group=smart_judge_outputs_group,
                )
            except Exception as e:
                if self.accelerator.is_main_process:
                    print(f"[WARN] wandb logging failed: {e}", flush=True)

        # Write samples jsonl (skip if centralize - already written in _centralized_generate_and_score)
        if not bool(getattr(self.args, "centralize_actor_judge", False)):
            self._write_samples_jsonl(
                prompts=prompts,
                refs=refs,
                candidate_groups=candidate_groups,
                final_scores_group=final_scores_group,
                med_scores_group=med_scores_group,
                smart_scores_group=smart_scores_group,
                med_judge_outputs_group=med_judge_outputs_group,
                smart_judge_outputs_group=smart_judge_outputs_group,
            )

        # temp=0 eval
        self._maybe_run_temp0_eval()

        self.global_step += 1

        if self.accelerator.is_main_process:
            dt_total = time.time() - t_step0
            print(f"[STEP] ===== step {self.global_step} done in {dt_total:.2f}s =====", flush=True)

    def _maybe_run_temp0_eval(self, force: bool = False):
        """Run temp=0 evaluation periodically (rank0 only)."""
        # Guard: without vLLM, generation would run locally on rank0 only.
        # Under FSDP (or any multi-rank setup) a rank0-only forward/generate
        # requires parameter all-gathers from every rank and deadlocks, so
        # skip the eval entirely in that configuration.
        if not (self.use_vllm and self.vllm_clients):
            if self.is_fsdp or self.world_size > 1:
                if self.accelerator.is_main_process:
                    print(
                        "[EVAL][WARN] temp=0 eval skipped: no vLLM clients available and "
                        "rank0-only local generation would deadlock FSDP all-gather. "
                        "Enable --use_vllm to run temp=0 eval.",
                        flush=True,
                    )
                return
        if not self.accelerator.is_main_process:
            return
        if self.eval_every <= 0 or len(self.eval_prompts) == 0:
            return

        if not force:
            # Check if we crossed an eval_every boundary
            reward_mode = str(getattr(self.args, "reward_mode", "grpo") or "grpo").lower()
            if reward_mode in ("grpo", "gdpo"):
                inner_epochs = int(getattr(self.args, "grpo_epochs", 1) or 1)
            else:  # gspo
                inner_epochs = int(getattr(self.args, "gspo_epochs", 4) or 4)

            prev_updates = self.optimizer_updates - inner_epochs
            crossed = (prev_updates // self.eval_every) < (self.optimizer_updates // self.eval_every)
            if not crossed:
                return

        prompts = self.eval_prompts
        refs = self.eval_refs

        label = "baseline" if self.optimizer_updates == 0 else f"optimizer_updates={self.optimizer_updates}"
        print(f"[EVAL] running temp=0 eval (n={len(prompts)}, {label})...", flush=True)
        t0 = time.time()

        # Generate with temp=0
        if self.use_vllm and self.vllm_clients:
            enabled = [i for i, en in enumerate(self.vllm_enabled) if en]
            if not enabled:
                print("[EVAL] no vLLM clients available, skipping", flush=True)
                return

            print(f"[EVAL] using vLLM for temp=0 generation (clients={len(enabled)})", flush=True)

            # Format all prompts with chat template
            formatted_prompts = []
            for p in prompts:
                if hasattr(self.tok, "apply_chat_template") and getattr(self.tok, "chat_template", None):
                    p2 = self.tok.apply_chat_template([{"role": "user", "content": p}],
                                                      add_generation_prompt=True, tokenize=False)
                else:
                    p2 = p
                formatted_prompts.append(p2)

            # Use vLLM generate API
            outs = [""] * len(formatted_prompts)
            
            gen_kwargs = {
                "max_tokens": int(self.args.eval_max_new_tokens),
                "n": 1,
                "temperature": float(self.args.eval_temperature),
                "top_p": float(self.args.eval_top_p),
                "repetition_penalty": float(self.args.gen_repetition_penalty),
            }

            def _eval_one(idx: int, prompt: str) -> Tuple[int, str]:
                client_idx = enabled[idx % len(enabled)]
                client = self.vllm_clients[client_idx]
                try:
                    resp = client.generate([prompt], **gen_kwargs)
                    # Parse response
                    if isinstance(resp, dict):
                        if "completion_ids" in resp:
                            ids = resp["completion_ids"]
                            if ids and isinstance(ids, list) and len(ids) > 0:
                                seq = ids[0]
                                if isinstance(seq, list) and len(seq) > 0:
                                    seq = seq[0]  # n=1
                                if isinstance(seq, (list, torch.Tensor)):
                                    if isinstance(seq, torch.Tensor):
                                        seq = seq.tolist()
                                    txt = self.tok.decode(seq, skip_special_tokens=True)
                                    return idx, txt.strip()
                    return idx, ""
                except Exception as e:
                    print(f"[EVAL] vLLM error idx={idx}: {e}", flush=True)
                    return idx, ""

            max_workers = min(len(formatted_prompts), len(enabled) * 2)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_eval_one, i, p) for i, p in enumerate(formatted_prompts)]
                for fut in as_completed(futures):
                    idx, txt = fut.result()
                    outs[idx] = txt

            non_empty = sum(1 for o in outs if o.strip())
            print(f"[EVAL] vLLM generation done: {non_empty}/{len(outs)} non-empty", flush=True)
        else:
            # local generate
            outs = []
            self.model.eval()
            for p in prompts:
                if hasattr(self.tok, "apply_chat_template") and getattr(self.tok, "chat_template", None):
                    enc = self.tok.apply_chat_template([{"role": "user", "content": p}],
                                                       add_generation_prompt=True, return_tensors="pt")
                    if isinstance(enc, dict):
                        input_ids = enc["input_ids"].to(self.model_device)
                        attn = enc.get("attention_mask")
                        if attn is not None:
                            attn = attn.to(self.model_device)
                    else:
                        input_ids = enc.to(self.model_device)
                        attn = None
                else:
                    enc = self.tok(p, return_tensors="pt")
                    input_ids = enc["input_ids"].to(self.model_device)
                    attn = enc.get("attention_mask")
                    if attn is not None:
                        attn = attn.to(self.model_device)

                gen_kwargs = dict(
                    input_ids=input_ids,
                    max_new_tokens=int(self.args.eval_max_new_tokens),
                    do_sample=(float(self.args.eval_temperature) > 0),
                    temperature=float(self.args.eval_temperature) if float(self.args.eval_temperature) > 0 else None,
                    top_p=float(self.args.eval_top_p) if float(self.args.eval_temperature) > 0 else None,
                    repetition_penalty=float(self.args.gen_repetition_penalty),
                    use_cache=True,
                    pad_token_id=self.pad_token_id,
                    output_router_logits=False,
                    synced_gpus=True,
                )
                if attn is not None:
                    gen_kwargs["attention_mask"] = attn
                try:
                    gen = self.model.generate(**gen_kwargs)
                    txt = self.tok.decode(gen[0][input_ids.shape[-1]:], skip_special_tokens=True)
                    outs.append(txt)
                except Exception as e:
                    print(f"[EVAL] local gen error: {e}", flush=True)
                    outs.append("")
            self.model.train()

        # Score with judge
        try:
            med_scores, _ = self.judge_scores_flat(prompts, outs, refs)
            med_mean = float(med_scores.mean().item())
            dt = time.time() - t0
            print(f"[EVAL] temp0 med_mean={med_mean:.4f} (n={len(prompts)}, {dt:.1f}s)", flush=True)

            if self.wandb_run is not None:
                self.wandb_run.log({"eval/temp0_med_mean": float(med_mean)}, step=int(self.optimizer_updates))
        except Exception as e:
            print(f"[EVAL] judge error: {e}", flush=True)

    @torch.no_grad()
    def save_model_sharded(self, save_dir: str):
        """
        NEW (default): Saves the FULL model state (sharded) to ensure no data loss and avoid hangs.
        """
        os.makedirs(save_dir, exist_ok=True)

        if self.is_fsdp:
            with FSDP.state_dict_type(
                self.model,
                StateDictType.SHARDED_STATE_DICT,
                ShardedStateDictConfig(offload_to_cpu=True),
            ):
                sd = self.model.state_dict()

            path = os.path.join(save_dir, f"model_sharded.rank{self.rank}.pt")
            torch.save(sd, path)

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                meta = {
                    "format": "full_sharded_model",
                    "pattern": "model_sharded.rank{rank}.pt",
                    "world_size": int(self.world_size),
                    "steps": int(getattr(self.args, "steps", 0)),
                    "mask_json": self.args.mask_json,
                    "model": self.args.model,
                    "grpo_group_size": int(self.args.grpo_group_size),
                    "is_fsdp": True,
                    "note": "FSDP: each rank saves ALL local shards (SHARDED_STATE_DICT). No filtering.",
                }
                with open(os.path.join(save_dir, "model_sharded_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                print(f"[SAVE][NEW] full sharded params -> {meta['pattern']}", flush=True)
        else:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                unwrapped = self.accelerator.unwrap_model(self.model)
                sd = unwrapped.state_dict()
                path = os.path.join(save_dir, "model.pt")
                torch.save(sd, path)
                meta = {
                    "format": "full_model",
                    "steps": int(getattr(self.args, "steps", 0)),
                    "mask_json": self.args.mask_json,
                    "model": self.args.model,
                    "grpo_group_size": int(self.args.grpo_group_size),
                    "is_fsdp": False,
                    "num_tensors": int(len(sd)),
                    "note": "Non-FSDP: rank0 saves full state dict.",
                }
                with open(os.path.join(save_dir, "model_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                print(f"[SAVE][NEW] full params -> {path} (tensors={len(sd)})", flush=True)

    @torch.no_grad()
    def save_updated_params_only(self, save_dir: str):
        """
        NEW (optional): Saves ONLY parameters that are actually updated (requires_grad=True).
        """
        os.makedirs(save_dir, exist_ok=True)

        def clean_key(k: str) -> str:
            k = k.replace("_fsdp_wrapped_module.", "")
            k = k.replace("_checkpoint_wrapped_module.", "")
            k = k.replace("module.", "")
            if k.startswith("model."):
                k = k[6:]
            return k

        unwrapped = self.accelerator.unwrap_model(self.model)
        raw_trainable_names = {n for n, p in unwrapped.named_parameters() if p.requires_grad}
        clean_trainable_names = {clean_key(n) for n in raw_trainable_names}

        if self.accelerator.is_main_process:
            print(f"[SAVE-SELECT][NEW] Found {len(raw_trainable_names)} trainable params.", flush=True)

        if self.is_fsdp:
            with FSDP.state_dict_type(
                self.model,
                StateDictType.SHARDED_STATE_DICT,
                ShardedStateDictConfig(offload_to_cpu=True),
            ):
                sd = self.model.state_dict()

            sub = {}
            matched_count = 0
            for k, v in sd.items():
                ck = clean_key(k)
                if ck in clean_trainable_names:
                    sub[k] = v
                    matched_count += 1

            if self.accelerator.is_main_process:
                print(f"[SAVE-SELECT][NEW] Matched {matched_count} entries in state_dict.", flush=True)

            path = os.path.join(save_dir, f"updated_params_sharded.rank{self.rank}.pt")
            torch.save(sub, path)

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                meta = {
                    "format": "trainable_params_only_sharded",
                    "pattern": "updated_params_sharded.rank{rank}.pt",
                    "world_size": int(self.world_size),
                    "steps": int(getattr(self.args, "steps", 0)),
                    "mask_json": self.args.mask_json,
                    "model": self.args.model,
                    "grpo_group_size": int(self.args.grpo_group_size),
                    "is_fsdp": True,
                    "note": "FSDP: Selective save (normalized key match).",
                }
                with open(os.path.join(save_dir, "updated_params_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                print(f"[SAVE-SELECT][NEW] updated-only sharded params -> {meta['pattern']}", flush=True)
        else:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                sub = {n: p.detach().cpu() for n, p in unwrapped.named_parameters() if p.requires_grad}
                path = os.path.join(save_dir, "updated_params.pt")
                torch.save(sub, path)
                meta = {
                    "format": "trainable_params_only",
                    "steps": int(getattr(self.args, "steps", 0)),
                    "mask_json": self.args.mask_json,
                    "model": self.args.model,
                    "grpo_group_size": int(self.args.grpo_group_size),
                    "is_fsdp": False,
                    "num_tensors": int(len(sub)),
                    "note": "Non-FSDP: rank0 saves requires_grad=True params only.",
                }
                with open(os.path.join(save_dir, "updated_params_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                print(f"[SAVE-SELECT][NEW] updated-only params -> {path} (tensors={len(sub)})", flush=True)

    def train(self):
        if self.accelerator.is_main_process:
            self.train_wall_start = time.time()

        # Track epoch/step info
        self.total_steps = int(self.args.steps) if int(getattr(self.args, "epochs", 0)) <= 0 else 0

        # Run baseline eval before training (skip if resuming)
        if self.global_step == 0 and self.eval_every > 0:
            self._maybe_run_temp0_eval(force=True)

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

            if self.accelerator.is_main_process:
                print(f"[TRAIN] epochs={epochs} steps_per_epoch={steps_per_epoch} total_steps={self.total_steps}", flush=True)

            step = 0
            it = iter(self.loader)
            for _ep in range(epochs):
                self.epoch_idx = _ep
                self.step_in_epoch = 0
                for _ in range(int(steps_per_epoch)):
                    try:
                        batch = next(it)
                    except StopIteration:
                        it = iter(self.loader)
                        batch = next(it)
                    self.step(batch)
                    step += 1
                    self.step_in_epoch += 1
        else:
            self.total_steps = int(self.args.steps)
            step = 0
            while step < self.args.steps:
                for batch in self.loader:
                    self.step(batch)
                    step += 1
                    if step >= self.args.steps:
                        break

        if self.vllm_clients:
            for c in self.vllm_clients:
                try:
                    c.close_communicator()
                except Exception as e:
                    if self.accelerator.is_main_process:
                        print(f"[WARN] VLLM close_communicator failed: {e}", flush=True)

        # =========================
        # SAVE (NEW main + optional LEGACY)
        # =========================
        # Legacy-only: keep old behavior (save_state into --save_dir)
        if getattr(self.args, "save_legacy_only", False):
            if self.accelerator.is_main_process:
                print(f"[SAVE][LEGACY-ONLY] start accelerator.save_state() -> {self.args.save_dir}", flush=True)
            t0 = time.time()
            self.accelerator.save_state(self.args.save_dir)
            self.accelerator.wait_for_everyone()
            t1 = time.time()
            if self.accelerator.is_main_process:
                print(f"[SAVE][LEGACY-ONLY] finish accelerator.save_state() elapsed={t1 - t0:.1f}s", flush=True)
                print("[SAVE][LEGACY-ONLY] FSDP sharded checkpoint written (accelerate.save_state).", flush=True)
            return

        # NEW default save to --save_dir
        if getattr(self.args, "save_updated_only", False):
            if self.accelerator.is_main_process:
                print(f"[SAVE][NEW] start save_updated_params_only() -> {self.args.save_dir}", flush=True)
            t0 = time.time()
            self.save_updated_params_only(self.args.save_dir)
            self.accelerator.wait_for_everyone()
            t1 = time.time()
            if self.accelerator.is_main_process:
                print(f"[SAVE][NEW] finish save_updated_params_only() elapsed={t1 - t0:.1f}s", flush=True)
                print("[SAVE][NEW] updated-only checkpoint written.", flush=True)
        else:
            if self.accelerator.is_main_process:
                print(f"[SAVE][NEW] start save_model_sharded() -> {self.args.save_dir}", flush=True)
            t0 = time.time()
            self.save_model_sharded(self.args.save_dir)
            self.accelerator.wait_for_everyone()
            t1 = time.time()
            if self.accelerator.is_main_process:
                print(f"[SAVE][NEW] finish save_model_sharded() elapsed={t1 - t0:.1f}s", flush=True)
                print("[SAVE][NEW] full sharded checkpoint written (optimizer state excluded).", flush=True)

        # Optional: run legacy after NEW into subdir under --save_dir
        if getattr(self.args, "save_legacy_after_new", False):
            legacy_subdir = getattr(self.args, "legacy_save_subdir", "legacy_accelerate_state")
            legacy_dir = os.path.join(self.args.save_dir, legacy_subdir)
            os.makedirs(legacy_dir, exist_ok=True)
            if self.accelerator.is_main_process:
                print(f"[SAVE][LEGACY-AFTER-NEW] start accelerator.save_state() -> {legacy_dir}", flush=True)
            t0 = time.time()
            try:
                self.accelerator.save_state(legacy_dir)
                self.accelerator.wait_for_everyone()
                t1 = time.time()
                if self.accelerator.is_main_process:
                    print(f"[SAVE][LEGACY-AFTER-NEW] finish accelerator.save_state() elapsed={t1 - t0:.1f}s", flush=True)
                    print("[SAVE][LEGACY-AFTER-NEW] legacy accelerate checkpoint written.", flush=True)
            except Exception as e:
                if self.accelerator.is_main_process:
                    print(f"[WARN][LEGACY-AFTER-NEW] accelerator.save_state failed: {e}", flush=True)

    def save_selected_experts(self, save_dir: str, full_state_dict: Optional[dict] = None):
        os.makedirs(save_dir, exist_ok=True)

        if full_state_dict is None:
            full_state_dict = self.accelerator.get_state_dict(self.model)

        sub = {k: v for k, v in full_state_dict.items() if is_selected_expert_param(k, self.layer_masks)}
        path = os.path.join(save_dir, "selected_experts_grpo.pt")
        torch.save(sub, path)
        meta = {
            "steps": self.args.steps,
            "mask_json": self.args.mask_json,
            "model": self.args.model,
            "fsdp_config_active": True,
            "grpo_group_size": self.args.grpo_group_size,
        }
        with open(os.path.join(save_dir, "selected_experts_grpo_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] selected-experts (GRPO) weights -> {path}", flush=True)


def build_argparser():
    ap = argparse.ArgumentParser(description="Expert-only full-param GRPO/GDPO/GSPO (Accelerate Config Embedded)")
    ap.add_argument("--model", type=str, default=os.environ.get("BASE_MODEL", "/path/to/base_model"))
    ap.add_argument("--layers", type=int, default=94)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--dataset", type=str, default=os.environ.get("SPECIALIST_CHATML_PATH", "/path/to/train.jsonl"))
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--mask_json", type=str, default="/path/to/selected_experts.json")
    ap.add_argument("--train_attention_only", action="store_true",
                    help="Train only Attention layers (Q/K/V/O projections). FFN/Experts are frozen. "
                         "This may preserve language abilities while adapting attention to a target context.")
    ap.add_argument("--attention_layer_start", type=int, default=0,
                    help="Start layer index for attention-only training (inclusive). Default: 0")
    ap.add_argument("--attention_layer_end", type=int, default=-1,
                    help="End layer index for attention-only training (exclusive). -1 means all layers. "
                         "Example: --attention_layer_start 47 --attention_layer_end 94 trains only the second half.")
    ap.add_argument("--grpo_group_size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--max_consecutive_nan", type=int, default=10, help="Abort training after this many consecutive NaN/Inf losses")
    ap.add_argument("--gen_max_new_tokens", type=int, default=192)
    ap.add_argument("--gen_temperature", type=float, default=0.7)
    ap.add_argument("--gen_top_p", type=float, default=0.9)
    ap.add_argument("--gen_repetition_penalty", type=float, default=1.05)
    ap.add_argument("--use_vllm", action="store_true")
    ap.add_argument("--vllm_base_url", type=str, default=None)
    ap.add_argument("--vllm_base_urls", type=str, default=None)
    ap.add_argument("--vllm_update_every", type=int, default=1)
    ap.add_argument("--vllm_force_n1", action="store_true")
    ap.add_argument("--vllm_connection_timeout", type=int, default=240)
    ap.add_argument("--vllm_sync_max_retries", type=int, default=3,
                    help="Max retries for vLLM param sync per server")
    ap.add_argument("--vllm_sync_retry_backoff", type=float, default=2.0,
                    help="Base backoff seconds for vLLM sync retry")
    ap.add_argument("--vllm_sync_retry_jitter", type=float, default=0.2,
                    help="Jitter factor for vLLM sync retry backoff")

    # judge (medical)
    ap.add_argument("--judge_max_new_tokens", type=int, default=2048)
    ap.add_argument("--judge_temperature", type=float, default=0.0)
    ap.add_argument("--judge_backend", type=str, default="local",
                    choices=["local", "openrouter", "openai", "cerebras",
                             "cerebras+openrouter", "openrouter+cerebras"],
                    help="Judge backend. Use 'X+Y' for fallback.")
    ap.add_argument("--judge_api_base", type=str, default=None)
    ap.add_argument("--judge_api_key", type=str, default=None)
    ap.add_argument("--judge_model_name", type=str, default=None)
    ap.add_argument("--judge_fallback_model_name", type=str, default=None,
                    help="Model name for fallback backend.")
    ap.add_argument("--judge_max_retries", type=int, default=1000)
    ap.add_argument("--judge_connect_timeout", type=float, default=10.0)
    ap.add_argument("--judge_read_timeout", type=float, default=60.0)
    ap.add_argument("--judge_retry_sleep", type=float, default=1.0)
    ap.add_argument("--judge_retry_sleep_max", type=float, default=60.0)
    ap.add_argument("--judge_http_retries", type=int, default=8)
    ap.add_argument("--judge_http_backoff", type=float, default=0.5)
    ap.add_argument("--judge_parallel_workers", type=int, default=0,
                    help="External judge parallel workers. 0=auto.")

    # SMART judge
    ap.add_argument("--smart_backend", type=str, default=None,
                    choices=[None, "local", "openrouter", "openai", "cerebras",
                             "cerebras+openrouter", "openrouter+cerebras"],
                    help="SMART backend. Defaults to judge_backend.")
    ap.add_argument("--smart_api_base", type=str, default=None)
    ap.add_argument("--smart_api_key", type=str, default=None)
    ap.add_argument("--smart_model_name", type=str, default=None)
    ap.add_argument("--smart_fallback_model_name", type=str, default=None)
    ap.add_argument("--smart_max_new_tokens", type=int, default=None)
    ap.add_argument("--smart_temperature", type=float, default=None)
    ap.add_argument("--smart_max_retries", type=int, default=None)
    ap.add_argument("--smart_parallel_workers", type=int, default=0)

    # reward shaping
    ap.add_argument("--reward_mode", type=str, default="grpo", choices=["grpo", "gdpo", "gspo"],
                    help="grpo: token PPO-clip + group norm. gdpo: decoupled norm. gspo: seq PPO-clip.")
    # GRPO/GDPO
    ap.add_argument("--grpo_clip_eps", type=float, default=0.2)
    ap.add_argument("--grpo_logratio_clip", type=float, default=20.0)
    ap.add_argument("--grpo_epochs", type=int, default=1)
    # GDPO
    ap.add_argument("--gdpo_eps", type=float, default=1e-6)
    ap.add_argument("--gdpo_whiten_global", dest="gdpo_whiten_global", action="store_true")
    ap.add_argument("--no_gdpo_whiten_global", dest="gdpo_whiten_global", action="store_false")
    ap.set_defaults(gdpo_whiten_global=True)
    # GSPO
    ap.add_argument("--gspo_clip_eps_low", type=float, default=3e-4)
    ap.add_argument("--gspo_clip_eps_high", type=float, default=4e-4)
    ap.add_argument("--gspo_logratio_clip", type=float, default=20.0)
    ap.add_argument("--gspo_epochs", type=int, default=4)

    ap.add_argument("--reward_scale", type=float, default=1.0)
    ap.add_argument("--reward_clip", type=float, default=1.0)
    ap.add_argument("--ema_alpha", type=float, default=0.05)
    ap.add_argument("--reward_cache", type=str, default="judge_score_cache.jsonl")

    # SMART reward settings
    ap.add_argument("--smart_reward_cache", type=str, default="smart_score_cache.jsonl")
    ap.add_argument("--smart_max_score", type=float, default=10.0)
    ap.add_argument("--smart_reward_weight", type=float, default=0.2)
    ap.add_argument("--smart_gate_mode", type=str, default="soft", choices=["soft", "hard", "none"])
    ap.add_argument("--smart_gate_threshold", type=float, default=0.5)
    ap.add_argument("--smart_reward_cap", type=float, default=0.2)
    ap.add_argument("--smart_hack_penalty", type=float, default=0.2)
    ap.add_argument("--smart_hack_med_le", type=float, default=0.0)
    ap.add_argument("--smart_hack_smart_ge", type=float, default=8.0)

    # centralize generation+judge
    ap.add_argument("--centralize_actor_judge", action="store_true",
                    help="Rank0 generates and judges all batches, then distributes. Reduces rank skew.")

    # misc
    ap.add_argument("--debug_print", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="expert_fullparam_grpo")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_table_rows", type=int, default=4)
    ap.add_argument("--wandb_table_text_max_chars", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_dir", type=str, default="expert_ckpts_qwen94x128_grpo")

    # samples jsonl
    ap.add_argument("--samples_jsonl_path", type=str, default="samples.jsonl")
    ap.add_argument("--samples_jsonl_every", type=int, default=1)
    ap.add_argument("--samples_jsonl_overwrite", action="store_true")
    ap.add_argument("--samples_jsonl_text_max_chars", type=int, default=0)

    # eval (temp=0)
    ap.add_argument("--eval_every", type=int, default=50, help="Run temp=0 eval every N optimizer updates. 0 disables.")
    ap.add_argument("--eval_size", type=int, default=100)
    ap.add_argument("--eval_max_new_tokens", type=int, default=2048)
    ap.add_argument("--eval_temperature", type=float, default=0.0)
    ap.add_argument("--eval_top_p", type=float, default=1.0)
    ap.add_argument("--eval_dataset", type=str, default=None)

    # === SAVE OPTIONS ===
    ap.add_argument(
        "--save_updated_only",
        action="store_true",
        help="(NEW) Save only updated params (requires_grad=True).",
    )
    ap.add_argument(
        "--save_legacy_only",
        action="store_true",
        help="(LEGACY) Use accelerator.save_state() ONLY.",
    )
    ap.add_argument(
        "--save_legacy_after_new",
        action="store_true",
        help="(LEGACY) After NEW save, ALSO run accelerator.save_state().",
    )
    ap.add_argument(
        "--legacy_save_subdir",
        type=str,
        default="legacy_accelerate_state",
    )

    return ap


def main():
    args = build_argparser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU が必要です。")
    set_seed(42)
    trainer = ExpertFullparamGRPOTrainer(args)
    trainer.train()
    if trainer.wandb_run is not None and trainer.accelerator.is_main_process:
        try:
            trainer.wandb_run.finish()
        except Exception as e:
            print(f"[WARN] wandb finish failed: {e}", flush=True)


if __name__ == "__main__":
    main()
