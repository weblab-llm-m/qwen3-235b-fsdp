#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expert+Attention full-parameter DPO for Qwen3-235B-A22B (FSDP, BF16).

【235B オリジナル機能】
  - mask_json による expert-only 学習
  - attn_layers / attn_proj による attention 層選択学習
  - save_model_sharded / save_updated_params_only
  - rank0 のみ重みロード、他 rank は from_config で meta モデル (新 transformers 対応)

【DPO fixes (80B DPO から継承)】
  FIX 1  Collator が chosen+rejected を同一 max_len にパッド (torch.cat compat)
  FIX 2  Ref precompute に concat forward (MoE routing consistency)
  FIX 3  ref_bs=1 強制 (batch=2 と学習時完全一致)
  FIX 4  step() で model.eval() (ref との dropout 一致)
  FIX 5  clip_grad_norm_ を sync_gradients でガード
  FIX 6  GradientAccumulationPlugin(sync_each_batch=True) (FSDP OOM 防止)
  FIX 7  Checkpointed cross-entropy (BxTxV float32 softmax OOM 回避)

【SFT v2 から継承した改善】
  FIX-C  MixedPrecision 明示設定
  FIX-D  rank0 のみ from_pretrained、他 rank は from_config (meta)
  FIX-E  _param_init_fn / meta バッファ実体化
  FIX-F  InitProcessGroupKwargs(timeout=7200s)
  FIX-G  AdamW(foreach=False)
  FIX-H  accelerator.accumulate() コンテキスト
  FIX-I  ETA 付きステップログ
"""

import gc
import os
import re
import json
import math
import random
import argparse
import datetime
import time
from typing import List, Tuple, Optional, Dict, Any
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import InitProcessGroupKwargs, GradientAccumulationPlugin
from torch.distributed.fsdp import (
    ShardingStrategy,
    StateDictType,
    ShardedStateDictConfig,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.distributed as dist

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    _FSDP_AVAILABLE = True
except Exception:
    _FSDP_AVAILABLE = False

try:
    from torch.distributed.optim import ZeroRedundancyOptimizer
    _ZRO_AVAILABLE = True
except Exception:
    _ZRO_AVAILABLE = False


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_sdp_backends():
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    try:
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


# ─────────────────────────────────────────────
# Expert mask utilities
# ─────────────────────────────────────────────

_EXPERT_RE = re.compile(r"\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")


def parse_layer_expert_from_name(name: str) -> Optional[Tuple[int, int]]:
    m = _EXPERT_RE.search(name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def load_mask(mask_json: Optional[str], n_layers: int, n_experts: int) -> Optional[List[torch.Tensor]]:
    if not mask_json:
        return None
    with open(mask_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    layer_masks = [torch.zeros(n_experts, dtype=torch.bool) for _ in range(n_layers)]
    sel = obj.get("selected_layer_experts")
    if sel:
        for it in sel:
            li, ei = int(it["layer_idx"]), int(it["expert_id"])
            if 0 <= li < n_layers and 0 <= ei < n_experts:
                layer_masks[li][ei] = True
    else:
        ex = obj.get("selected_experts_unique") or obj.get("selected_experts") or obj.get("picked_experts")
        if not ex:
            raise RuntimeError("Mask JSON lacks selected_* fields.")
        for li in range(n_layers):
            for ei in ex:
                if 0 <= int(ei) < n_experts:
                    layer_masks[li][int(ei)] = True
    for li in range(n_layers):
        if layer_masks[li].sum().item() == 0:
            layer_masks[li] = torch.ones(n_experts, dtype=torch.bool)
    return layer_masks


def is_router_param_name(name: str) -> bool:
    lname = name.lower()
    return (lname.endswith("mlp.gate.weight") or ".mlp.gate.weight" in lname
            or ("router" in lname and "router_logits" not in lname)
            or "routing" in lname)


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
    return bool(mask[ei].item()) if 0 <= ei < mask.numel() else False


# ─────────────────────────────────────────────
# Attention layer utilities
# ─────────────────────────────────────────────

_ATTN_LAYER_RE = re.compile(r"\.layers\.(\d+)\.(?:self_attn|attention)\.")
_DEFAULT_ATTN_PROJ = {"q_proj", "k_proj", "v_proj", "o_proj"}


def parse_attn_layers(spec: str) -> Optional[set]:
    if not spec:
        return None
    result = set()
    for token in spec.split(","):
        token = token.strip()
        if "-" in token:
            parts = token.split("-", 1)
            try:
                start, end = int(parts[0]), int(parts[1])
                result.update(range(start, end + 1))
            except ValueError:
                raise ValueError(f"Invalid attn_layers token: '{token}'")
        else:
            try:
                result.add(int(token))
            except ValueError:
                raise ValueError(f"Invalid attn_layers token: '{token}'")
    return result


def parse_attn_proj(spec: Optional[str]) -> set:
    if not spec:
        return set(_DEFAULT_ATTN_PROJ)
    return {s.strip() for s in spec.split(",") if s.strip()}


def is_selected_attn_param(name: str, attn_layer_set: Optional[set], attn_proj_set: set) -> bool:
    if attn_layer_set is None:
        return False
    m = _ATTN_LAYER_RE.search(name)
    if not m:
        return False
    li = int(m.group(1))
    if li not in attn_layer_set:
        return False
    return any(proj in name for proj in attn_proj_set)


def mark_trainable_params(
    model: nn.Module,
    layer_masks: Optional[List[torch.Tensor]],
    attn_layer_set: Optional[set],
    attn_proj_set: set,
):
    total = trainable_expert = trainable_attn = 0
    for name, p in model.named_parameters():
        total += p.numel()
        is_router = is_router_param_name(name)
        is_expert = is_selected_expert_param(name, layer_masks)
        is_attn = is_selected_attn_param(name, attn_layer_set, attn_proj_set)
        if is_router:
            p.requires_grad = False
        elif is_expert:
            p.requires_grad = True
            trainable_expert += p.numel()
        elif is_attn:
            p.requires_grad = True
            trainable_attn += p.numel()
        else:
            p.requires_grad = False
    trainable_total = trainable_expert + trainable_attn
    print(
        f"[INFO] trainable: expert={trainable_expert:,} attn={trainable_attn:,} "
        f"total={trainable_total:,} / {total:,} "
        f"({trainable_total / max(1, total) * 100:.4f}%)",
        flush=True,
    )


# ─────────────────────────────────────────────
# DPO Dataset
# ─────────────────────────────────────────────

class DPOMapDataset(Dataset):
    def __init__(self, path: str, tok, seq_len: int):
        self.tok = tok
        self.seq_len = seq_len
        self.samples: List[Dict] = []
        self.ref_chosen_logps: Optional[List[float]] = None
        self.ref_rejected_logps: Optional[List[float]] = None

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                result = self._tokenize_pair(obj)
                if result is not None:
                    self.samples.append(result)

        print(f"[DPO-DS] Loaded {len(self.samples)} pairs from {path}", flush=True)

    def _tokenize_with_labels(self, messages) -> Optional[Dict]:
        try:
            formatted = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            try:
                formatted = self.tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )
            except Exception:
                return None
        except Exception:
            return None

        if not formatted:
            return None

        enc = self.tok(formatted, add_special_tokens=False)
        input_ids = enc["input_ids"]
        if not input_ids:
            return None

        labels = [-100] * len(input_ids)
        assistant_header = "<|im_start|>assistant\n"
        im_end_str = "<|im_end|>"
        pos = 0
        while True:
            start = formatted.find(assistant_header, pos)
            if start == -1:
                break
            content_start = start + len(assistant_header)
            end = formatted.find(im_end_str, content_start)
            if end == -1:
                end = len(formatted)
            content_end = end + len(im_end_str)
            prefix_before = formatted[:content_start]
            prefix_after = formatted[:content_end]
            tok_start = len(self.tok(prefix_before, add_special_tokens=False)["input_ids"])
            tok_end = len(self.tok(prefix_after, add_special_tokens=False)["input_ids"])
            for j in range(tok_start, min(tok_end, len(labels))):
                labels[j] = input_ids[j]
            pos = content_end

        if all(l == -100 for l in labels):
            labels = list(input_ids)

        input_ids = input_ids[:self.seq_len]
        labels = labels[:self.seq_len]
        return {"input_ids": input_ids, "labels": labels}

    def _tokenize_pair(self, obj: dict) -> Optional[Dict]:
        prompt = obj.get("prompt")
        chosen = obj.get("chosen")
        rejected = obj.get("rejected")
        if not prompt or not chosen or not rejected:
            return None

        chosen_msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen}]
        rejected_msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected}]

        chosen_tok = self._tokenize_with_labels(chosen_msgs)
        rejected_tok = self._tokenize_with_labels(rejected_msgs)
        if chosen_tok is None or rejected_tok is None:
            return None

        if sum(1 for l in chosen_tok["labels"] if l != -100) == 0:
            return None
        if sum(1 for l in rejected_tok["labels"] if l != -100) == 0:
            return None

        return {
            "chosen_input_ids": chosen_tok["input_ids"],
            "chosen_labels": chosen_tok["labels"],
            "rejected_input_ids": rejected_tok["input_ids"],
            "rejected_labels": rejected_tok["labels"],
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        result = {
            "chosen_input_ids": s["chosen_input_ids"],
            "chosen_labels": s["chosen_labels"],
            "rejected_input_ids": s["rejected_input_ids"],
            "rejected_labels": s["rejected_labels"],
            "index": idx,
        }
        if self.ref_chosen_logps is not None:
            result["ref_chosen_logps"] = self.ref_chosen_logps[idx]
            result["ref_rejected_logps"] = self.ref_rejected_logps[idx]
        return result


# ─────────────────────────────────────────────
# FIX 1: Collator
# ─────────────────────────────────────────────

def dpo_collate_fn(batch, pad_id: int, seq_len: int):
    all_c_ids    = [b["chosen_input_ids"]    for b in batch]
    all_c_labels = [b["chosen_labels"]       for b in batch]
    all_r_ids    = [b["rejected_input_ids"]  for b in batch]
    all_r_labels = [b["rejected_labels"]     for b in batch]

    c_max  = max(len(ids) for ids in all_c_ids)
    r_max  = max(len(ids) for ids in all_r_ids)
    max_len = min(max(c_max, r_max), seq_len)

    def _pad_to(all_ids, all_labels):
        ids_t    = torch.full((len(batch), max_len), pad_id,  dtype=torch.long)
        mask_t   = torch.zeros(len(batch), max_len,            dtype=torch.long)
        labels_t = torch.full((len(batch), max_len), -100,    dtype=torch.long)
        for i, (ids, labs) in enumerate(zip(all_ids, all_labels)):
            slen = min(len(ids), max_len)
            ids_t[i, :slen]    = torch.tensor(ids[:slen],  dtype=torch.long)
            mask_t[i, :slen]   = 1
            labels_t[i, :slen] = torch.tensor(labs[:slen], dtype=torch.long)
        return ids_t, mask_t, labels_t

    c_ids, c_mask, c_labels = _pad_to(all_c_ids, all_c_labels)
    r_ids, r_mask, r_labels = _pad_to(all_r_ids, all_r_labels)

    result = {
        "chosen_input_ids":        c_ids,
        "chosen_attention_mask":   c_mask,
        "chosen_labels":           c_labels,
        "rejected_input_ids":      r_ids,
        "rejected_attention_mask": r_mask,
        "rejected_labels":         r_labels,
        "index": torch.tensor([b["index"] for b in batch], dtype=torch.long),
    }
    if "ref_chosen_logps" in batch[0]:
        result["ref_chosen_logps"]   = torch.tensor(
            [b["ref_chosen_logps"]   for b in batch], dtype=torch.float32)
        result["ref_rejected_logps"] = torch.tensor(
            [b["ref_rejected_logps"] for b in batch], dtype=torch.float32)
    return result


# ─────────────────────────────────────────────
# FIX 7: Checkpointed cross-entropy
# ─────────────────────────────────────────────

def _cross_entropy_no_save(logits_flat: torch.Tensor, labels_flat: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits_flat, labels_flat, reduction="none")


def compute_per_sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = (shift_labels != -100)
    B, T, V = shift_logits.shape
    per_token_ce = torch.utils.checkpoint.checkpoint(
        _cross_entropy_no_save,
        shift_logits.view(-1, V),
        shift_labels.clamp(min=0).view(-1),
        use_reentrant=False,
    ).view(B, T)
    return (-per_token_ce * mask.float()).sum(dim=-1)


def compute_per_sequence_logps_with_length(
    logits: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = (shift_labels != -100)
    B, T, V = shift_logits.shape
    per_token_ce = torch.utils.checkpoint.checkpoint(
        _cross_entropy_no_save,
        shift_logits.view(-1, V),
        shift_labels.clamp(min=0).view(-1),
        use_reentrant=False,
    ).view(B, T)
    per_token_logps = -per_token_ce * mask.float()
    return per_token_logps.sum(dim=-1), mask.float().sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
    loss_type: str = "sigmoid",
    length_normalise: bool = False,
    chosen_lengths: Optional[torch.Tensor] = None,
    rejected_lengths: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pc = policy_chosen_logps
    pr = policy_rejected_logps
    rc = ref_chosen_logps.to(pc.device)
    rr = ref_rejected_logps.to(pr.device)

    if length_normalise and chosen_lengths is not None and rejected_lengths is not None:
        pc = pc / chosen_lengths.clamp(min=1).to(pc.device)
        pr = pr / rejected_lengths.clamp(min=1).to(pr.device)
        rc = rc / chosen_lengths.clamp(min=1).to(rc.device)
        rr = rr / rejected_lengths.clamp(min=1).to(rr.device)

    chosen_rewards   = beta * (pc - rc)
    rejected_rewards = beta * (pr - rr)

    if loss_type == "sigmoid":
        logits = chosen_rewards - rejected_rewards
        if label_smoothing > 0:
            loss = -(
                (1 - label_smoothing) * F.logsigmoid(-logits)
                + label_smoothing * F.logsigmoid(logits)
            ).mean()
        else:
            loss = -F.logsigmoid(logits).mean()
    elif loss_type == "hinge":
        loss = torch.relu(1.0 - (chosen_rewards - rejected_rewards)).mean()
    elif loss_type == "ipo":
        logits = (pc - pr) - (rc - rr)
        loss = ((logits - 1.0 / (2.0 * beta)) ** 2).mean()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss, chosen_rewards.detach(), rejected_rewards.detach()


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────

class ExpertFullparamDPOTrainer:
    def __init__(self, args):
        self.args = args
        self.use_wandb = False

        if not _FSDP_AVAILABLE:
            raise RuntimeError("[FSDP] torch.distributed.fsdp が使えません。")
        if not torch.cuda.is_available():
            raise RuntimeError("[CUDA] GPU が必要です。")

        set_seed(args.seed)
        configure_sdp_backends()

        # ── Layer class discovery ─────────────────────────────────────────────
        rank_env = int(os.environ.get("RANK", "0"))
        if rank_env == 0:
            print("[INIT] Loading model config to detect layer class...", flush=True)

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

        if rank_env == 0:
            print(f"[INIT] Detected Transformer Layer Class: {layer_cls}", flush=True)

        auto_wrap_policy = None
        if layer_cls is not None:
            auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={layer_cls},
            )

        # ── FIX-C: MixedPrecision ────────────────────────────────────────────
        bf16_mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        fsdp_plugin = FullyShardedDataParallelPlugin(
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            cpu_ram_efficient_loading=True,
            sync_module_states=True,
            limit_all_gathers=True,
            auto_wrap_policy=auto_wrap_policy,
            activation_checkpointing=args.activation_checkpointing,
        )
        fsdp_plugin.mixed_precision_policy = bf16_mp

        # ── FIX-E: _param_init_fn ─────────────────────────────────────────────
        def _param_init_fn(module):
            has_meta = any(
                p.device.type == "meta" for p in module.parameters(recurse=False)
            ) or any(
                b is not None and b.device.type == "meta"
                for b in module.buffers(recurse=False)
            )
            if has_meta:
                module.to_empty(device=torch.device("cpu"), recurse=False)
            return module

        fsdp_plugin.param_init_fn = _param_init_fn

        # ── FIX-F: タイムアウト延長 ──────────────────────────────────────────
        pg_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=7200))

        # ── FIX 6: GradientAccumulationPlugin ────────────────────────────────
        ga_plugin = GradientAccumulationPlugin(
            num_steps=args.gradient_accumulation_steps,
            sync_each_batch=True,
        )

        self.accelerator = Accelerator(
            mixed_precision="no",
            fsdp_plugin=fsdp_plugin,
            gradient_accumulation_plugin=ga_plugin,
            kwargs_handlers=[pg_kwargs],
        )

        self.rank       = self.accelerator.process_index
        self.world_size = self.accelerator.num_processes
        self.device     = self.accelerator.device

        if self.accelerator.is_main_process:
            print(f"[DIST] rank={self.rank} / world_size={self.world_size}", flush=True)

        # ── Tokenizer ────────────────────────────────────────────────────────
        self.tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        # ── FIX-D: rank0 のみ重みをロード、他 rank は from_config (meta) ─────
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        config.torch_dtype = torch.bfloat16

        if self.rank == 0:
            print("[LOAD] rank 0: loading weights to CPU (BF16)...", flush=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        else:
            print(f"[LOAD] rank {self.rank}: creating meta model (no weight load)...", flush=True)
            with torch.device("meta"):
                base_model = AutoModelForCausalLM.from_config(
                    config,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
            print(f"[LOAD] rank {self.rank}: meta model ready.", flush=True)

        # FIX-E: 非 rank0 の meta バッファを CPU テンソルに実体化
        if self.rank != 0:
            for module in base_model.modules():
                for bname in list(module._buffers.keys()):
                    buf = module._buffers[bname]
                    if buf is not None and buf.device.type == "meta":
                        module._buffers[bname] = torch.empty(
                            buf.shape, dtype=buf.dtype, device="cpu")

        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = False
        if hasattr(base_model.config, "output_router_logits"):
            base_model.config.output_router_logits = False

        if args.activation_checkpointing:
            base_model.gradient_checkpointing_enable()
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        base_model.train()

        # ── Expert mask ───────────────────────────────────────────────────────
        self.layer_masks = None
        if args.mask_json:
            self.layer_masks = load_mask(args.mask_json, args.layers, args.experts)
            if self.accelerator.is_main_process:
                allowed = int(sum(m.sum().item() for m in self.layer_masks))
                print(f"[INFO] mask allow entries = {allowed}", flush=True)

        # ── Attention layer mask ──────────────────────────────────────────────
        self.attn_layer_set = None
        self.attn_proj_set = parse_attn_proj(getattr(args, "attn_proj", None))
        if getattr(args, "attn_layers", None):
            self.attn_layer_set = parse_attn_layers(args.attn_layers)
            if self.accelerator.is_main_process:
                print(
                    f"[INFO] attn_layers={sorted(self.attn_layer_set)} "
                    f"({len(self.attn_layer_set)} layers) | "
                    f"attn_proj={sorted(self.attn_proj_set)}",
                    flush=True,
                )

        mark_trainable_params(
            base_model,
            layer_masks=self.layer_masks,
            attn_layer_set=self.attn_layer_set,
            attn_proj_set=self.attn_proj_set,
        )

        self.accelerator.wait_for_everyone()

        # ── FSDP wrap ─────────────────────────────────────────────────────────
        if self.accelerator.is_main_process:
            print("[FSDP] Wrapping model...", flush=True)
        self.model = self.accelerator.prepare(base_model)
        self.is_fsdp = isinstance(self.model, FSDP)
        if self.accelerator.is_main_process:
            print("[FSDP] Model wrapped.", flush=True)

        # ── Detach hook: 最初の trainable 層の入力で勾配グラフを切断 ──────────
        # GLM-5 と同じ手法。frozen 層 (0 ~ first_train-1) の activations を
        # backward グラフから切り離すことで、それらを保持する必要がなくなり
        # activation checkpointing なしでもメモリが激減する。
        # attn_layer_set と layer_masks (expert mask) の両方が指定されている場合、
        # どちらか一方だけを見て first_train_idx を決めると、もう一方の trainable
        # 層がそれより浅い位置にあったとき detach hook が勾配を遮断してしまう。
        # → 両者の最小層の min を採用する。
        _first_candidates = []
        if self.attn_layer_set is not None:
            _first_candidates.append(min(self.attn_layer_set))
        if self.layer_masks is not None:
            _first_mask_idx = next(
                (li for li in range(args.layers) if self.layer_masks[li].any()), None)
            if _first_mask_idx is not None:
                _first_candidates.append(_first_mask_idx)
        first_train_idx = min(_first_candidates) if _first_candidates else 0

        # FSDP wrap 後のモデルから layers ModuleList を探す
        def _find_layers_list(model, num_layers):
            for mod in model.modules():
                if isinstance(mod, nn.ModuleList) and len(mod) == num_layers:
                    return mod
            for attr_path in ["model.layers", "layers",
                               "_fsdp_wrapped_module.model.layers",
                               "_fsdp_wrapped_module.layers"]:
                obj = model
                try:
                    for part in attr_path.split("."):
                        obj = getattr(obj, part)
                    if isinstance(obj, nn.ModuleList):
                        return obj
                except AttributeError:
                    continue
            return None

        fsdp_layers = _find_layers_list(self.model, args.layers)
        _detach_installed = False

        if fsdp_layers is not None and first_train_idx < len(fsdp_layers):
            target_layer = fsdp_layers[first_train_idx]

            def _detach_hook(module, args_tuple):
                new_args = []
                for a in args_tuple:
                    if isinstance(a, torch.Tensor) and a.is_floating_point():
                        new_args.append(a.detach().requires_grad_(True))
                    else:
                        new_args.append(a)
                return tuple(new_args)

            target_layer.register_forward_pre_hook(_detach_hook)
            _detach_installed = True
            if self.accelerator.is_main_process:
                print(
                    f"[DETACH] Installed detach hook on layer {first_train_idx} "
                    f"(frozen layers 0-{first_train_idx-1} excluded from backward graph)",
                    flush=True,
                )
        else:
            if self.accelerator.is_main_process:
                print(f"[DETACH] WARNING: Could not find layer {first_train_idx} "
                      f"in FSDP model. Falling back to activation_checkpointing.",
                      flush=True)
            # detach に失敗した場合は activation_checkpointing を強制 ON
            if not args.activation_checkpointing:
                if self.accelerator.is_main_process:
                    print("[DETACH] Forcing activation_checkpointing=True as fallback.",
                          flush=True)
                self.model.gradient_checkpointing_enable()

        if self.accelerator.is_main_process:
            print(f"[DETACH] detach_hook={'OK' if _detach_installed else 'FAILED'} | "
                  f"act_ckpt={args.activation_checkpointing}", flush=True)

        # ── MLP output detach: trainable 層内の frozen MLP の activations を切断 ──
        # attn のみ学習時、各層内の MLP は frozen だが detach なしだと
        # backward 中に MLP activations も保持されてしまう。
        # input detach + MLP output detach の両方で初めて
        # attn の activations だけ保持すればよくなる。
        # ※ mask_json (expert mask) 指定時は MLP/MoE 内の expert が trainable なので、
        #    MLP 出力を detach すると expert の勾配が遮断される → 無効化する。
        mlp_detach_count = 0
        if self.layer_masks is not None:
            if self.accelerator.is_main_process and self.attn_layer_set is not None:
                print("[DETACH] MLP output detach disabled: --mask_json specifies "
                      "trainable experts inside MLP/MoE modules.", flush=True)
        elif self.attn_layer_set is not None and fsdp_layers is not None:
            for tidx in sorted(self.attn_layer_set):
                if tidx >= len(fsdp_layers):
                    continue
                layer_mod = fsdp_layers[tidx]
                inner = getattr(layer_mod, "_fsdp_wrapped_module", layer_mod)
                inner = getattr(inner, "_checkpoint_wrapped_module", inner)

                mlp_module = None
                for attr_name in ("mlp", "moe", "feed_forward", "ffn",
                                  "block_sparse_moe", "shared_experts"):
                    mlp_module = getattr(inner, attr_name, None)
                    if mlp_module is not None:
                        break

                if mlp_module is not None:
                    def _make_mlp_detach(idx):
                        def _hook(module, input, output):
                            if isinstance(output, torch.Tensor) and output.requires_grad:
                                return output.detach().requires_grad_(True)
                            elif isinstance(output, tuple):
                                new = []
                                for o in output:
                                    if isinstance(o, torch.Tensor) and o.requires_grad:
                                        new.append(o.detach().requires_grad_(True))
                                    else:
                                        new.append(o)
                                return tuple(new)
                            return output
                        return _hook
                    mlp_module.register_forward_hook(_make_mlp_detach(tidx))
                    mlp_detach_count += 1

        if self.accelerator.is_main_process:
            print(f"[DETACH] MLP output detach: {mlp_detach_count} layers "
                  f"(frozen MLP activations excluded from backward graph)", flush=True)

        # ── Optimizer ─────────────────────────────────────────────────────────
        train_params = [p for p in self.model.parameters() if p.requires_grad]
        if not train_params:
            raise RuntimeError("No trainable parameters found.")

        self.opt = torch.optim.AdamW(
            train_params, lr=args.lr, betas=(0.9, 0.98),
            weight_decay=args.weight_decay,
            foreach=False,  # FIX-G
        )
        self.opt = self.accelerator.prepare(self.opt)

        # ── Pad token ─────────────────────────────────────────────────────────
        if self.tok.pad_token_id is not None:
            self.pad_token_id = int(self.tok.pad_token_id)
        elif self.tok.eos_token_id is not None:
            self.pad_token_id = int(self.tok.eos_token_id)
        else:
            raise RuntimeError("Tokenizer has neither pad_token_id nor eos_token_id.")

        # ── Dataset ───────────────────────────────────────────────────────────
        if self.accelerator.is_main_process:
            print(f"[DATA] dataset: {args.dataset}", flush=True)
        self.dataset = DPOMapDataset(args.dataset, tok=self.tok, seq_len=args.seq_len)

        if self.accelerator.is_main_process and len(self.dataset) > 0:
            s0 = self.dataset.samples[0]
            for side in ("chosen", "rejected"):
                ids  = s0[f"{side}_input_ids"]
                labs = s0[f"{side}_labels"]
                n_train = sum(1 for l in labs if l != -100)
                n_mask  = sum(1 for l in labs if l == -100)
                print(f"[DATA] sample 0 {side}: {len(ids)} tokens, "
                      f"{n_train} trainable, {n_mask} masked", flush=True)

        # ── Ref log-probs precompute ──────────────────────────────────────────
        if self.accelerator.is_main_process:
            print(f"\n[REF] Precomputing reference log-probs "
                  f"({len(self.dataset)} pairs)...", flush=True)
        self._precompute_ref_logprobs()
        if self.accelerator.is_main_process:
            rc = self.dataset.ref_chosen_logps
            rr = self.dataset.ref_rejected_logps
            print(f"[REF] Done. avg ref_chosen={sum(rc)/max(len(rc),1):.2f}, "
                  f"avg ref_rejected={sum(rr)/max(len(rr),1):.2f}", flush=True)

        # ── Training DataLoader ───────────────────────────────────────────────
        self.train_sampler = DistributedSampler(
            self.dataset, num_replicas=self.world_size, rank=self.rank,
            shuffle=True, seed=args.seed, drop_last=True,
        )
        _collate = partial(dpo_collate_fn, pad_id=self.pad_token_id, seq_len=args.seq_len)
        self.loader = DataLoader(
            self.dataset, batch_size=args.batch_size,
            sampler=self.train_sampler, collate_fn=_collate,
            num_workers=2, pin_memory=True, drop_last=True,
        )

        gas = args.gradient_accumulation_steps
        samples_per_rank = len(self.dataset) // self.world_size
        micro_per_epoch  = max(1, samples_per_rank // args.batch_size)
        optim_per_epoch  = micro_per_epoch // gas
        self.estimated_total_steps = max(1, optim_per_epoch * (args.epochs or 1))

        self.global_step  = 0
        self.global_epoch = 0

        # ── W&B ───────────────────────────────────────────────────────────────
        if self.accelerator.is_main_process and args.wandb_project:
            try:
                import wandb
                wandb.init(
                    project=args.wandb_project,
                    name=args.wandb_run_name or None,
                    config=vars(args),
                    resume="allow",
                )
                self.use_wandb = True
                print(f"[WANDB] Initialized: project={args.wandb_project}", flush=True)
            except Exception as e:
                print(f"[WANDB] Init failed: {e}", flush=True)

        if self.accelerator.is_main_process:
            print("=" * 56)
            print("  Expert Full-Param DPO v1")
            print(f"  model              : {args.model}")
            print(f"  layers/experts     : {args.layers}/{args.experts}")
            print(f"  dataset            : {args.dataset}")
            print(f"  batch_size (micro) : {args.batch_size}")
            print(f"  grad_accum         : {gas}")
            print(f"  eff_batch          : {args.batch_size * gas * self.world_size}")
            print(f"  seq_len            : {args.seq_len}")
            print(f"  beta               : {args.beta}")
            print(f"  loss_type          : {args.loss_type}")
            print(f"  label_smoothing    : {args.label_smoothing}")
            print(f"  length_normalise   : {args.length_normalise}")
            print(f"  lr                 : {args.lr}")
            print(f"  mode               : {'epochs' if args.epochs > 0 else 'steps'}")
            print(f"  epochs             : {args.epochs}")
            print(f"  steps(cap)         : {args.steps}")
            attn_str = args.attn_layers if getattr(args, "attn_layers", None) else "none"
            proj_str = args.attn_proj if getattr(args, "attn_proj", None) else "q_proj,k_proj,v_proj,o_proj (default)"
            print(f"  attn_layers        : {attn_str}")
            print(f"  attn_proj          : {proj_str}")
            print(f"  act_ckpt           : {args.activation_checkpointing}")
            print(f"  estimated steps    : {self.estimated_total_steps}")
            print("=" * 56, flush=True)

    # ──────────────────────────────────────────
    # FIX 2 & 3: Ref log-probs precompute
    # ──────────────────────────────────────────

    @torch.no_grad()
    def _precompute_ref_logprobs(self):
        self.model.eval()
        n = len(self.dataset)
        n_padded = math.ceil(n / self.world_size) * self.world_size
        all_indices = list(range(n)) + [0] * (n_padded - n)
        per_rank  = n_padded // self.world_size
        my_start  = self.rank * per_rank
        my_end    = my_start + per_rank
        my_indices = all_indices[my_start:my_end]

        # FIX 3: ref_bs=1
        ref_bs   = 1
        _collate = partial(dpo_collate_fn, pad_id=self.pad_token_id, seq_len=self.args.seq_len)
        local_results: Dict[int, Tuple[float, float]] = {}
        t0 = time.time()
        num_batches = math.ceil(len(my_indices) / ref_bs)

        for bi in range(num_batches):
            batch_indices = my_indices[bi * ref_bs:(bi + 1) * ref_bs]
            batch_items   = [self.dataset[i] for i in batch_indices]
            batch         = _collate(batch_items)

            # FIX 2: concat forward
            c_ids    = batch["chosen_input_ids"].to(self.device, non_blocking=True)
            c_mask   = batch["chosen_attention_mask"].to(self.device, non_blocking=True)
            c_labels = batch["chosen_labels"].to(self.device, non_blocking=True)
            r_ids    = batch["rejected_input_ids"].to(self.device, non_blocking=True)
            r_mask   = batch["rejected_attention_mask"].to(self.device, non_blocking=True)
            r_labels = batch["rejected_labels"].to(self.device, non_blocking=True)

            all_ids    = torch.cat([c_ids,    r_ids],    dim=0)
            all_mask   = torch.cat([c_mask,   r_mask],   dim=0)
            all_labels = torch.cat([c_labels, r_labels], dim=0)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=all_ids, attention_mask=all_mask,
                    use_cache=False, output_router_logits=False, return_dict=True,
                )
            all_logps = compute_per_sequence_logps(outputs.logits, all_labels)
            bs = c_ids.size(0)
            c_logps = all_logps[:bs]
            r_logps = all_logps[bs:]
            del outputs, all_ids, all_mask, all_labels, all_logps

            for j, idx in enumerate(batch_indices):
                if idx < n:
                    local_results[idx] = (c_logps[j].item(), r_logps[j].item())

            if self.accelerator.is_main_process and (bi + 1) % max(1, num_batches // 10) == 0:
                print(f"  [REF] {bi+1}/{num_batches} batches, "
                      f"elapsed={time.time()-t0:.1f}s", flush=True)

        all_results = [None] * self.world_size
        dist.all_gather_object(all_results, local_results)

        ref_chosen   = [0.0] * n
        ref_rejected = [0.0] * n
        for rank_dict in all_results:
            for idx, (c, r) in rank_dict.items():
                ref_chosen[idx]   = c
                ref_rejected[idx] = r

        self.dataset.ref_chosen_logps   = ref_chosen
        self.dataset.ref_rejected_logps = ref_rejected
        self.model.train()

    # ──────────────────────────────────────────
    # Training step
    # ──────────────────────────────────────────

    def step(self, batch: Dict[str, torch.Tensor]):
        # FIX 4: eval() for dropout consistency with ref
        self.model.eval()

        c_ids    = batch["chosen_input_ids"].to(self.device, non_blocking=True)
        c_mask   = batch["chosen_attention_mask"].to(self.device, non_blocking=True)
        c_labels = batch["chosen_labels"].to(self.device, non_blocking=True)
        r_ids    = batch["rejected_input_ids"].to(self.device, non_blocking=True)
        r_mask   = batch["rejected_attention_mask"].to(self.device, non_blocking=True)
        r_labels = batch["rejected_labels"].to(self.device, non_blocking=True)
        ref_c    = batch["ref_chosen_logps"].to(self.device, non_blocking=True)
        ref_r    = batch["ref_rejected_logps"].to(self.device, non_blocking=True)

        # FIX-H: accumulate() コンテキスト
        with self.accelerator.accumulate(self.model):
            all_ids    = torch.cat([c_ids,    r_ids],    dim=0)
            all_mask   = torch.cat([c_mask,   r_mask],   dim=0)
            all_labels = torch.cat([c_labels, r_labels], dim=0)
            B = c_ids.size(0)
            del c_ids, r_ids, c_mask, r_mask, c_labels, r_labels

            torch.cuda.empty_cache()
            if self._micro_step == 0 and self.accelerator.is_local_main_process:
                torch.cuda.reset_peak_memory_stats()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=all_ids, attention_mask=all_mask,
                    use_cache=False, output_router_logits=False, return_dict=True,
                )
            del all_ids, all_mask

            logits = outputs.logits
            del outputs

            if self.args.length_normalise:
                all_logps, all_lengths = compute_per_sequence_logps_with_length(logits, all_labels)
                pc_logps  = all_logps[:B]
                pr_logps  = all_logps[B:]
                c_lengths = all_lengths[:B]
                r_lengths = all_lengths[B:]
            else:
                all_logps = compute_per_sequence_logps(logits, all_labels)
                pc_logps  = all_logps[:B]
                pr_logps  = all_logps[B:]
                c_lengths = r_lengths = None
            del logits, all_labels, all_logps

            # メモリログ (最初のステップのみ)
            if self._micro_step == 0 and self.accelerator.is_local_main_process:
                _alloc = torch.cuda.memory_allocated() / 1e9
                _peak  = torch.cuda.max_memory_allocated() / 1e9
                print(f"[MEM] after fwd+logps: alloc={_alloc:.1f}GB peak={_peak:.1f}GB",
                      flush=True)

            loss, chosen_rewards, rejected_rewards = dpo_loss(
                policy_chosen_logps=pc_logps,
                policy_rejected_logps=pr_logps,
                ref_chosen_logps=ref_c,
                ref_rejected_logps=ref_r,
                beta=self.args.beta,
                label_smoothing=self.args.label_smoothing,
                loss_type=self.args.loss_type,
                length_normalise=self.args.length_normalise,
                chosen_lengths=c_lengths,
                rejected_lengths=r_lengths if self.args.length_normalise else None,
            )
            del pc_logps, pr_logps

            torch.cuda.empty_cache()
            self.accelerator.backward(loss)

            # FIX 5: sync_gradients でガード
            if self.accelerator.sync_gradients and self.args.grad_clip > 0:
                self.accelerator.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.args.grad_clip,
                )
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)

        loss_val      = loss.item()
        reward_margin = (chosen_rewards - rejected_rewards).mean().item()
        accuracy      = (chosen_rewards > rejected_rewards).float().mean().item()
        del loss

        self._micro_step += 1
        if loss_val == loss_val:  # not NaN
            self._micro_loss_accum   += loss_val
            self._micro_margin_accum += reward_margin
            self._micro_acc_accum    += accuracy
            self._micro_count        += 1

        gas = self.args.gradient_accumulation_steps
        if self._micro_step % gas != 0:
            return

        self.global_step += 1
        current_step = self.global_step

        avg_loss   = self._micro_loss_accum   / max(1, self._micro_count)
        avg_margin = self._micro_margin_accum / max(1, self._micro_count)
        avg_acc    = self._micro_acc_accum    / max(1, self._micro_count)
        self._micro_loss_accum = self._micro_margin_accum = self._micro_acc_accum = 0.0
        self._micro_count = 0

        self._epoch_loss_accum   += avg_loss
        self._epoch_margin_accum += avg_margin
        self._epoch_acc_accum    += avg_acc
        self._epoch_count        += 1

        step_end = time.time()
        self._step_times.append(step_end)

        if current_step % self.args.log_every == 0 and self.accelerator.is_main_process:
            elapsed  = step_end - self._train_start_time
            run_avg_loss = self._epoch_loss_accum / max(1, self._epoch_count)

            if len(self._step_times) >= 2:
                window = min(50, len(self._step_times))
                steps_per_sec = (window - 1) / max(
                    self._step_times[-1] - self._step_times[-window], 1e-6)
                sec_per_step = 1.0 / max(steps_per_sec, 1e-6)
            else:
                steps_per_sec = sec_per_step = 0.0

            eta_str = "?"
            pct_str = ""
            if self.estimated_total_steps > 0:
                pct = current_step / self.estimated_total_steps * 100
                pct_str = f" ({pct:.1f}%)"
                remaining = max(0, self.estimated_total_steps - current_step)
                if steps_per_sec > 0:
                    eta_str = str(datetime.timedelta(seconds=int(remaining / steps_per_sec)))

            epoch_str = f"epoch={self.global_epoch} " if self.args.epochs > 0 else ""
            print(
                f"  {epoch_str}step {current_step}{pct_str} | "
                f"loss={avg_loss:.4f} (avg={run_avg_loss:.4f}) | "
                f"margin={avg_margin:.3f} | acc={avg_acc:.3f} | "
                f"{sec_per_step:.1f}s/step | "
                f"elapsed={str(datetime.timedelta(seconds=int(elapsed)))} | "
                f"ETA={eta_str}",
                flush=True,
            )

            if self.use_wandb:
                import wandb
                wandb.log({
                    "train/loss":          avg_loss,
                    "train/avg_loss":      run_avg_loss,
                    "train/reward_margin": avg_margin,
                    "train/accuracy":      avg_acc,
                    "train/epoch":         self.global_epoch,
                    "perf/sec_per_step":   sec_per_step,
                }, step=current_step)

        if self.args.save_every and current_step % self.args.save_every == 0 and current_step > 0:
            self.save(os.path.join(self.args.save_dir, f"step-{current_step}"))

    def _should_stop(self) -> bool:
        return (self.args.steps is not None) and (self.args.steps > 0) and (
            self.global_step >= self.args.steps)

    # ──────────────────────────────────────────
    # Save
    # ──────────────────────────────────────────

    @torch.no_grad()
    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        if self.is_fsdp:
            with FSDP.state_dict_type(
                self.model, StateDictType.SHARDED_STATE_DICT,
                ShardedStateDictConfig(offload_to_cpu=True),
            ):
                sd = self.model.state_dict()
            path = os.path.join(save_dir, f"model_sharded.rank{self.rank}.pt")
            torch.save(sd, path)
            del sd
            gc.collect()
            torch.cuda.empty_cache()
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                meta = {
                    "format":    "dpo_sharded",
                    "pattern":   "model_sharded.rank{rank}.pt",
                    "world_size": int(self.world_size),
                    "model":     self.args.model,
                    "mask_json": self.args.mask_json,
                    "beta":      self.args.beta,
                    "loss_type": self.args.loss_type,
                    "steps":     int(self.global_step),
                }
                with open(os.path.join(save_dir, "model_sharded_meta.json"), "w") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                print(f"[SAVE] sharded -> {save_dir}", flush=True)
        else:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                unwrapped = self.accelerator.unwrap_model(self.model)
                sd   = unwrapped.state_dict()
                path = os.path.join(save_dir, "model.pt")
                torch.save(sd, path)
                del sd
                gc.collect()
                torch.cuda.empty_cache()
                print(f"[SAVE] full model -> {path}", flush=True)

    # ──────────────────────────────────────────
    # Train
    # ──────────────────────────────────────────

    def train(self):
        self.model.train()
        self._train_start_time   = time.time()
        self._step_times         = [self._train_start_time]
        self._epoch_loss_accum   = 0.0
        self._epoch_margin_accum = 0.0
        self._epoch_acc_accum    = 0.0
        self._epoch_count        = 0
        self._micro_step         = 0
        self._micro_loss_accum   = 0.0
        self._micro_margin_accum = 0.0
        self._micro_acc_accum    = 0.0
        self._micro_count        = 0

        if self.args.epochs and self.args.epochs > 0:
            for ep in range(1, self.args.epochs + 1):
                self.global_epoch = ep
                self.train_sampler.set_epoch(ep)
                ep_start      = time.time()
                ep_step_start = self.global_step
                self._epoch_loss_accum = self._epoch_margin_accum = self._epoch_acc_accum = 0.0
                self._epoch_count = 0

                if self.accelerator.is_main_process:
                    print(f"\n[EPOCH] start {ep}/{self.args.epochs}", flush=True)

                for batch in self.loader:
                    self.step(batch)
                    if self._should_stop():
                        break

                if self.accelerator.is_main_process:
                    ep_elapsed = time.time() - ep_start
                    ep_steps   = self.global_step - ep_step_start
                    avg_loss   = self._epoch_loss_accum   / max(1, self._epoch_count)
                    avg_margin = self._epoch_margin_accum / max(1, self._epoch_count)
                    avg_acc    = self._epoch_acc_accum    / max(1, self._epoch_count)
                    print(
                        f"[EPOCH] finished {ep}/{self.args.epochs} | "
                        f"steps={ep_steps} | avg_loss={avg_loss:.4f} | "
                        f"margin={avg_margin:.3f} | acc={avg_acc:.3f} | "
                        f"time={str(datetime.timedelta(seconds=int(ep_elapsed)))}",
                        flush=True,
                    )
                gc.collect()
                torch.cuda.empty_cache()

                if self._should_stop():
                    if self.accelerator.is_main_process:
                        print(f"[EPOCH] stopped early by steps cap: {self.args.steps}", flush=True)
                    break
        else:
            if not (self.args.steps and self.args.steps > 0):
                raise RuntimeError("steps mode requires --steps > 0 (or use --epochs).")
            while not self._should_stop():
                self.train_sampler.set_epoch(self.global_step)
                for batch in self.loader:
                    self.step(batch)
                    if self._should_stop():
                        break

        if self.accelerator.is_main_process:
            total_elapsed = time.time() - self._train_start_time
            avg_loss = self._epoch_loss_accum / max(1, self._epoch_count)
            print(
                f"\n[DONE] steps={self.global_step} | avg_loss={avg_loss:.4f} | "
                f"total_time={str(datetime.timedelta(seconds=int(total_elapsed)))}",
                flush=True,
            )

        self.accelerator.wait_for_everyone()
        self.save(self.args.save_dir)

        if self.use_wandb:
            import wandb
            wandb.finish()


# ─────────────────────────────────────────────
# Argparse & main
# ─────────────────────────────────────────────

def build_argparser():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model",   type=str, default=os.environ.get("BASE_MODEL", ""))
    ap.add_argument("--layers",  type=int, default=94)
    ap.add_argument("--experts", type=int, default=128)

    ap.add_argument("--dataset", type=str, default="")
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--mask_json", type=str, default=None)

    # Attention layer mode
    ap.add_argument("--attn_layers", type=str, default=None,
                    help="学習させる attention 層。例: '70-93' / '0,1,2' / '0-5,8,10-12'")
    ap.add_argument("--attn_proj",   type=str, default=None,
                    help="attn_layers と併用する projection キーワード (カンマ区切り)。"
                         "デフォルト: q_proj,k_proj,v_proj,o_proj")

    # DPO hyperparams
    ap.add_argument("--beta",             type=float, default=0.1)
    ap.add_argument("--loss_type",        type=str,   default="sigmoid",
                    choices=["sigmoid", "hinge", "ipo"])
    ap.add_argument("--label_smoothing",  type=float, default=0.0)
    ap.add_argument("--length_normalise", action="store_true")

    # Training control
    ap.add_argument("--epochs",     type=int, default=3)
    ap.add_argument("--steps",      type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--seed",       type=int, default=42)

    # Optimizer
    ap.add_argument("--lr",           type=float, default=5e-6)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip",    type=float, default=1.0)

    # Logging
    ap.add_argument("--log_every",      type=int, default=1)
    ap.add_argument("--save_every",     type=int, default=0)
    ap.add_argument("--wandb_project",  type=str, default="")
    ap.add_argument("--wandb_run_name", type=str, default="")

    # Memory mode
    ap.add_argument("--activation_checkpointing", action="store_true", default=True,
                    help="activation checkpointing ON=メモリ節約(遅い) / OFF=高速(多メモリ)")
    ap.add_argument("--no_activation_checkpointing", dest="activation_checkpointing",
                    action="store_false",
                    help="activation checkpointing を無効化 (メモリ余裕時に高速化)")

    # Save
    ap.add_argument("--save_dir", type=str, default="outputs/ckpts_dpo")

    return ap


def main():
    args = build_argparser().parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU が必要です。")

    trainer = ExpertFullparamDPOTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
