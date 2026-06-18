#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expert-only full-parameter SFT (Supervised Fine-Tuning).
v2: 80B スクリプトのメモリ効率改善を 235B 向けにバックポート。

【235B オリジナル機能】
  - mask_json による expert-only 学習
  - Adafactor / AdamW / ZeroAdamW 選択
  - BucketedBatcher (length bucketing)
  - TokenizedJsonl / OnTheFlyTokenizedChatMLJsonl
  - save_model_sharded / save_updated_params_only
  - offline pretokenize モード

【80B から取り込んだ改善 (FIX番号は 80B スクリプトに対応)】
  FIX-A  GradientAccumulationPlugin(sync_each_batch=True)
           → FSDP grad buffer の OOM を防止
  FIX-B  clip_grad_norm_ を sync_gradients でガード
           → accumulation 途中での不正クリップを排除
  FIX-C  MixedPrecision を FSDPPlugin に明示設定
           → bf16 reduce/buffer を確実に有効化
  FIX-D  rank0 のみ from_pretrained, 他 rank は meta モデル
           → 非 rank0 の CPU RAM をほぼゼロに削減
  FIX-E  _param_init_fn で meta バッファを実体化
           → FSDP sync 時の meta tensor エラー回避
  FIX-F  InitProcessGroupKwargs(timeout=7200s)
           → 大規模クラスタでのタイムアウト対策
  FIX-G  AdamW に foreach=False
           → FSDP 下での foreach による OOM 回避
  FIX-H  accelerator.accumulate() コンテキスト使用
           → 正しい gradient sync タイミング
  FIX-I  タイミング / ETA 付きステップログ
"""

import gc
import os
import re
import json
import random
import argparse
import datetime
import time
from typing import List, Tuple, Optional, Dict, Any
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.optimization import Adafactor

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import InitProcessGroupKwargs, GradientAccumulationPlugin  # FIX-A/F
from torch.distributed.fsdp import (
    ShardingStrategy,
    StateDictType,
    ShardedStateDictConfig,
    MixedPrecision,               # FIX-C
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

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


def extract_text_from_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join([extract_text_from_content(x) for x in content if x])
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        inner = content.get("content")
        if isinstance(inner, (str, list, dict)):
            return extract_text_from_content(inner)
        return "\n".join(v for v in content.values() if isinstance(v, str))
    return ""


def _global_shard_params(rank: int, world_size: int) -> Tuple[int, int]:
    wi = get_worker_info()
    if wi is None:
        worker_id, num_workers = 0, 1
    else:
        worker_id, num_workers = wi.id, wi.num_workers
    return rank * num_workers + worker_id, world_size * num_workers


def tokenize_prompt_ref(tok, prompt: str, ref: str, seq_len: int) -> Tuple[List[int], List[int]]:
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        msgs_user = [{"role": "user", "content": prompt}]
        msgs_full = [{"role": "user", "content": prompt}, {"role": "assistant", "content": ref}]

        enc_user = tok.apply_chat_template(msgs_user, add_generation_prompt=True, tokenize=True, return_tensors="pt")
        enc_full = tok.apply_chat_template(msgs_full, add_generation_prompt=False, tokenize=True, return_tensors="pt")

        ids_user = enc_user["input_ids"][0] if isinstance(enc_user, dict) else enc_user[0]
        ids_full = enc_full["input_ids"][0] if isinstance(enc_full, dict) else enc_full[0]

        start = int(ids_user.shape[0])
        labels = torch.full_like(ids_full, -100)
        if start < ids_full.shape[0]:
            labels[start:] = ids_full[start:]
        else:
            labels[:] = ids_full
        ids_list, labels_list = ids_full.tolist(), labels.tolist()
    else:
        enc = tok(prompt + "\n" + ref, return_tensors="pt")
        ids = enc["input_ids"][0]
        p_len = int(tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1])
        labels = torch.full_like(ids, -100)
        if p_len < ids.shape[0]:
            labels[p_len:] = ids[p_len:]
        else:
            labels[:] = ids
        ids_list, labels_list = ids.tolist(), labels.tolist()

    if len(ids_list) > seq_len:
        ids_list, labels_list = ids_list[:seq_len], labels_list[:seq_len]
    return ids_list, labels_list


# ─────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────

class OnTheFlyTokenizedChatMLJsonl(IterableDataset):
    def __init__(self, raw_path, tok, seq_len, rank, world_size):
        self.raw_path = raw_path
        self.tok = tok
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        global_rank, global_world = _global_shard_params(self.rank, self.world_size)
        with open(self.raw_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i % global_world != global_rank:
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
                prompt = extract_text_from_content(umsgs[0].get("content")).strip()
                ref = extract_text_from_content(amsgs[0].get("content")).strip() if amsgs else ""
                if not prompt or not ref:
                    continue
                ids_list, labels_list = tokenize_prompt_ref(self.tok, prompt, ref, self.seq_len)
                if not ids_list:
                    continue
                yield {
                    "input_ids": torch.tensor(ids_list, dtype=torch.long),
                    "labels": torch.tensor(labels_list, dtype=torch.long),
                    "length": len(ids_list),
                }


class TokenizedJsonl(IterableDataset):
    def __init__(self, path, seq_len, rank, world_size):
        self.path = path
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        global_rank, global_world = _global_shard_params(self.rank, self.world_size)
        with open(self.path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i % global_world != global_rank:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                ids_list = rec.get("input_ids")
                labels_list = rec.get("labels")
                if not isinstance(ids_list, list) or not isinstance(labels_list, list):
                    continue
                if not ids_list:
                    continue
                if len(ids_list) > self.seq_len:
                    ids_list = ids_list[:self.seq_len]
                    labels_list = labels_list[:self.seq_len]
                yield {
                    "input_ids": torch.tensor(ids_list, dtype=torch.long),
                    "labels": torch.tensor(labels_list, dtype=torch.long),
                    "length": len(ids_list),
                }


class BucketedBatcher(IterableDataset):
    """Length bucketing でパディング量を削減するバッチャー。"""
    def __init__(self, base, batch_size, buffer_size, pad_token_id, seq_len, max_batch_len=0, drop_last=False):
        self.base = base
        self.batch_size = batch_size
        self.buffer_size = max(buffer_size, batch_size)
        self.pad_token_id = pad_token_id
        self.seq_len = seq_len
        self.max_batch_len = int(max_batch_len) if max_batch_len else 0
        self.drop_last = drop_last

    def _pad_and_stack(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = min(max(int(s["length"]) for s in samples), self.seq_len)
        if self.max_batch_len > 0:
            max_len = min(max_len, self.max_batch_len)
        ids_l, lbl_l, attn_l = [], [], []
        for s in samples:
            ids = s["input_ids"][:max_len]
            lbl = s["labels"][:max_len]
            cur = int(ids.shape[0])
            pad = max_len - cur
            if pad > 0:
                ids = torch.nn.functional.pad(ids, (0, pad), value=self.pad_token_id)
                lbl = torch.nn.functional.pad(lbl, (0, pad), value=-100)
                attn = torch.cat([torch.ones(cur, dtype=torch.long), torch.zeros(pad, dtype=torch.long)])
            else:
                attn = torch.ones(max_len, dtype=torch.long)
            ids_l.append(ids); lbl_l.append(lbl); attn_l.append(attn)
        return {
            "input_ids": torch.stack(ids_l),
            "attention_mask": torch.stack(attn_l),
            "labels": torch.stack(lbl_l),
        }

    def __iter__(self):
        buf: List[Dict[str, Any]] = []
        def flush(buffer):
            buffer.sort(key=lambda x: int(x["length"]))
            for i in range(0, len(buffer), self.batch_size):
                chunk = buffer[i: i + self.batch_size]
                if len(chunk) < self.batch_size and self.drop_last:
                    continue
                yield self._pad_and_stack(chunk)
        for ex in self.base:
            buf.append(ex)
            if len(buf) >= self.buffer_size:
                yield from flush(buf)
                buf = []
        if buf:
            yield from flush(buf)


class PackedDataset(IterableDataset):
    """
    複数サンプルを seq_len に詰め込むパッキング実装。
    パディングをほぼゼロにして GPU 利用率を最大化する。

    - input_ids / labels をサンプル境界で連結
    - labels はサンプル内で既に prompt 部分が -100 なのでそのまま連結
    - attention_mask は全 1 (パディングなし)
    - seq_len に満たないチャンクは末尾に eos でパッド（ロスはマスク）
    """
    def __init__(self, base, seq_len: int, pad_token_id: int, batch_size: int = 1):
        self.base = base
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.batch_size = batch_size  # パッキング後のバッチサイズ

    def __iter__(self):
        buf_ids: List[int] = []
        buf_lbl: List[int] = []

        def emit(ids, lbl):
            """seq_len に切り揃えて batch dict を返す。"""
            ids = ids[:self.seq_len]
            lbl = lbl[:self.seq_len]
            # 末尾が短い場合はパッド（ロスはマスク）
            pad = self.seq_len - len(ids)
            if pad > 0:
                ids = ids + [self.pad_token_id] * pad
                lbl = lbl + [-100] * pad
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long).unsqueeze(0),
                "attention_mask": torch.ones(1, self.seq_len, dtype=torch.long),
                "labels": torch.tensor(lbl, dtype=torch.long).unsqueeze(0),
            }

        for ex in self.base:
            new_ids = ex["input_ids"].tolist()
            new_lbl = ex["labels"].tolist()

            # 現バッファに追加したら seq_len を超えるならまず flush
            while len(buf_ids) + len(new_ids) >= self.seq_len:
                space = self.seq_len - len(buf_ids)
                buf_ids += new_ids[:space]
                buf_lbl += new_lbl[:space]
                yield emit(buf_ids, buf_lbl)
                new_ids = new_ids[space:]
                new_lbl = new_lbl[space:]
                buf_ids, buf_lbl = [], []

            buf_ids += new_ids
            buf_lbl += new_lbl

        # 残りを flush（末尾パッド）
        if buf_ids:
            yield emit(buf_ids, buf_lbl)


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

def mark_trainable_experts_only(model: nn.Module, layer_masks: Optional[List[torch.Tensor]]):
    total = trainable = 0
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
        f"({trainable / max(1, total) * 100:.4f}%)", flush=True
    )


# ─────────────────────────────────────────────
# Attention layer utilities
# ─────────────────────────────────────────────

# .layers.{N}.self_attn.{proj}.weight  (Qwen3 / LLaMA 系共通)
_ATTN_LAYER_RE = re.compile(r"\.layers\.(\d+)\.(?:self_attn|attention)\.")

# デフォルトで学習する projection キーワード
_DEFAULT_ATTN_PROJ = {"q_proj", "k_proj", "v_proj", "o_proj"}


def parse_attn_layers(spec: str) -> Optional[set]:
    """
    "0-10"    → {0,1,...,10}
    "0,1,2,10" → {0,1,2,10}
    "0-5,8,10-12" → 範囲とリストの混在も可
    None / "" → None (全層 or 無効)
    """
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
    """
    "q_proj,k_proj" → {"q_proj","k_proj"}
    None / ""       → デフォルト全部 (q/k/v/o)
    """
    if not spec:
        return set(_DEFAULT_ATTN_PROJ)
    return {s.strip() for s in spec.split(",") if s.strip()}


def is_selected_attn_param(
    name: str,
    attn_layer_set: Optional[set],
    attn_proj_set: set,
) -> bool:
    """指定層・指定 proj の attention パラメータかどうか判定。"""
    if attn_layer_set is None:
        return False
    m = _ATTN_LAYER_RE.search(name)
    if not m:
        return False
    li = int(m.group(1))
    if li not in attn_layer_set:
        return False
    # proj キーワードがパラメータ名に含まれるか確認
    return any(proj in name for proj in attn_proj_set)


def mark_trainable_params(
    model: nn.Module,
    layer_masks: Optional[List[torch.Tensor]],
    attn_layer_set: Optional[set],
    attn_proj_set: set,
):
    """
    expert mask と attention layer 指定を組み合わせて trainable を決定。
    どちらも指定なし → expert mask のみの従来動作。
    両方指定      → OR で両方学習。
    attn のみ指定 → attention 層のみ学習 (expert mask 無効)。
    """
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
# Offline pretokenize
# ─────────────────────────────────────────────

def pretokenize_jsonl(model_name_or_path: str, in_path: str, out_path: str, seq_len: int):
    print(f"[PRETOKENIZE] load tokenizer: {model_name_or_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    n_in = n_out = 0
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            n_in += 1
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
            prompt = extract_text_from_content(umsgs[0].get("content")).strip()
            ref = extract_text_from_content(amsgs[0].get("content")).strip() if amsgs else ""
            if not prompt or not ref:
                continue
            ids_list, labels_list = tokenize_prompt_ref(tok, prompt, ref, seq_len)
            if not ids_list:
                continue
            fout.write(json.dumps({"input_ids": ids_list, "labels": labels_list}, ensure_ascii=False) + "\n")
            n_out += 1
            if n_out % 1000 == 0:
                print(f"[PRETOKENIZE] wrote {n_out} / seen {n_in}", flush=True)
    print(f"[PRETOKENIZE] done: in={n_in} -> out={n_out} @ {out_path}", flush=True)


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────

class ExpertFullparamSFTTrainer:
    def __init__(self, args):
        self.args = args
        self.wandb_run = None

        if not _FSDP_AVAILABLE:
            raise RuntimeError("[FSDP] torch.distributed.fsdp が使えません。")
        if not torch.cuda.is_available():
            raise RuntimeError("[CUDA] GPU が必要です。")

        configure_sdp_backends()

        # ── Layer class discovery (meta モデルで検出) ──────────────────────────
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

        # ── FIX-C: MixedPrecision を明示 ──────────────────────────────────────
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
            activation_checkpointing=True,
        )
        fsdp_plugin.mixed_precision_policy = bf16_mp  # FIX-C

        # ── FIX-E: _param_init_fn で meta テンソルを実体化 ────────────────────
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

        fsdp_plugin.param_init_fn = _param_init_fn  # FIX-E

        # ── FIX-F: タイムアウト延長 ───────────────────────────────────────────
        pg_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=7200))

        # ── FIX-A: GradientAccumulationPlugin(sync_each_batch=True) ──────────
        ga_plugin = GradientAccumulationPlugin(
            num_steps=args.gradient_accumulation_steps,
            sync_each_batch=True,
        )

        self.accelerator = Accelerator(
            mixed_precision="bf16",
            fsdp_plugin=fsdp_plugin,
            gradient_accumulation_plugin=ga_plugin,  # FIX-A
            kwargs_handlers=[pg_kwargs],              # FIX-F
        )

        self.rank = self.accelerator.process_index
        self.world_size = self.accelerator.num_processes
        self.device = self.accelerator.device
        self.model_device = self.device

        if self.accelerator.is_main_process:
            print(f"[DIST] rank={self.rank} / world_size={self.world_size}", flush=True)

        # ── Tokenizer ─────────────────────────────────────────────────────────
        if self.accelerator.is_main_process:
            print("[LOAD] model:", args.model, flush=True)
        self.tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        # ── FIX-D: rank0 のみ重みをロード、他 rank は meta モデル ─────────────
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
            # 新しい transformers (4.51+) は meta デバイス内での from_pretrained を禁止。
            # from_config で構造だけ作り、重みロードをスキップする。
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

        # ── Optimizer ─────────────────────────────────────────────────────────
        expert_params = [p for p in self.model.parameters() if p.requires_grad]
        if not expert_params:
            raise RuntimeError("No trainable expert parameters found.")

        if args.optim == "adamw":
            self.opt = torch.optim.AdamW(
                expert_params, lr=args.lr, betas=(0.9, 0.98),
                weight_decay=args.weight_decay,
                foreach=False,  # FIX-G
            )
        elif args.optim == "zero_adamw":
            if not _ZRO_AVAILABLE:
                raise RuntimeError("ZeroRedundancyOptimizer not available.")
            self.opt = ZeroRedundancyOptimizer(
                expert_params, optimizer_class=torch.optim.AdamW,
                lr=args.lr, betas=(0.9, 0.98),
                weight_decay=args.weight_decay,
            )
        else:  # adafactor (default)
            self.opt = Adafactor(
                expert_params, lr=args.lr,
                scale_parameter=False, relative_step=False,
                warmup_init=False, weight_decay=args.weight_decay,
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
        if args.tokenized_dataset:
            if self.accelerator.is_main_process:
                print(f"[DATA] tokenized_dataset: {args.tokenized_dataset}", flush=True)
            base_ds = TokenizedJsonl(
                args.tokenized_dataset, seq_len=args.seq_len,
                rank=self.rank, world_size=self.world_size,
            )
        else:
            if self.accelerator.is_main_process:
                print(f"[DATA] raw dataset (on-the-fly tokenize): {args.dataset}", flush=True)
            base_ds = OnTheFlyTokenizedChatMLJsonl(
                args.dataset, tok=self.tok, seq_len=args.seq_len,
                rank=self.rank, world_size=self.world_size,
            )

        batched_ds: IterableDataset
        if getattr(args, "packing", False):
            if self.accelerator.is_main_process:
                print("[DATA] packing mode: ON", flush=True)
            if args.num_workers > 0 and self.accelerator.is_main_process:
                print(f"[WARN] packing + num_workers={args.num_workers}: "
                      "ワーカーごとに独立してパッキングされます。num_workers=0 推奨。", flush=True)
            batched_ds = PackedDataset(
                base=base_ds, seq_len=args.seq_len,
                pad_token_id=self.pad_token_id,
                batch_size=args.batch_size,
            )
        else:
            batched_ds = BucketedBatcher(
                base=base_ds, batch_size=args.batch_size,
                buffer_size=args.bucket_buffer, pad_token_id=self.pad_token_id,
                seq_len=args.seq_len, max_batch_len=args.max_batch_len, drop_last=False,
            )
        self.loader = DataLoader(batched_ds, batch_size=None,
                                 num_workers=args.num_workers, drop_last=False)

        # ── エポック総ステップ推定（ETA 表示用） ──────────────────────────────
        # データファイルをスキャンしてトークン総数から正確に推定する。
        # packing あり: total_tokens / seq_len / world_size
        # packing なし: ceil(samples/gpu / batch_size)
        self._estimated_epoch_steps = 0
        ds_path = args.tokenized_dataset or args.dataset
        if ds_path and os.path.isfile(ds_path):
            try:
                n_lines = 0
                total_tokens = 0
                is_tokenized = bool(args.tokenized_dataset)
                with open(ds_path, "r", encoding="utf-8") as _f:
                    for _line in _f:
                        _line = _line.strip()
                        if not _line:
                            continue
                        n_lines += 1
                        try:
                            _rec = json.loads(_line)
                            if is_tokenized:
                                total_tokens += len(_rec.get("input_ids", []))
                            else:
                                # raw jsonl: messages から大まかなトークン数を推定
                                # 正確ではないが文字数 / 3 で近似（日本語・英語混合）
                                msgs = _rec.get("messages") or []
                                text = " ".join(
                                    m.get("content", "") if isinstance(m.get("content"), str)
                                    else str(m.get("content", ""))
                                    for m in msgs
                                )
                                total_tokens += max(1, len(text) // 3)
                        except Exception:
                            pass

                samples_per_gpu = max(1, (n_lines + self.world_size - 1) // self.world_size)
                if getattr(args, "packing", False):
                    tokens_per_gpu = max(1, total_tokens // self.world_size)
                    self._estimated_epoch_steps = max(1, tokens_per_gpu // args.seq_len)
                else:
                    self._estimated_epoch_steps = max(1, (samples_per_gpu + args.batch_size - 1) // args.batch_size)

                avg_tokens = total_tokens / max(1, n_lines)
                if self.accelerator.is_main_process:
                    print(
                        f"[DATA] dataset lines={n_lines} | total_tokens={total_tokens:,} | "
                        f"avg_tokens={avg_tokens:.0f} | "
                        f"estimated steps/epoch ≈ {self._estimated_epoch_steps}",
                        flush=True,
                    )
            except Exception as e:
                if self.accelerator.is_main_process:
                    print(f"[WARN] dataset scan failed: {e}", flush=True)

        self.global_step = 0
        self.global_epoch = 0

        # ── W&B ───────────────────────────────────────────────────────────────
        if self.args.wandb and self.accelerator.is_main_process:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=self.args.wandb_project,
                    entity=self.args.wandb_entity or None,
                    name=self.args.wandb_run_name,
                    config=vars(self.args),
                )
            except Exception as e:
                print(f"[WARN] wandb init failed: {e}", flush=True)

        if self.accelerator.is_main_process:
            gas = args.gradient_accumulation_steps
            print("=" * 56)
            print("  Expert Full-Param SFT v2")
            print(f"  model              : {args.model}")
            print(f"  layers/experts     : {args.layers}/{args.experts}")
            print(f"  dataset(raw)       : {args.dataset}")
            print(f"  dataset(tokenized) : {args.tokenized_dataset}")
            print(f"  batch_size (micro) : {args.batch_size}")
            print(f"  grad_accum         : {gas}")
            print(f"  eff_batch          : {args.batch_size * gas * self.world_size}")
            print(f"  bucket_buffer      : {args.bucket_buffer}")
            print(f"  packing            : {getattr(args, 'packing', False)}")
            print(f"  seq_len            : {args.seq_len}")
            print(f"  max_batch_len      : {args.max_batch_len}")
            print(f"  optim              : {args.optim}")
            print(f"  lr                 : {args.lr}")
            print(f"  mode               : {'epochs' if args.epochs > 0 else 'steps'}")
            print(f"  epochs             : {args.epochs}")
            print(f"  steps(cap)         : {args.steps}")
            attn_str = args.attn_layers if getattr(args, "attn_layers", None) else "none"
            proj_str = args.attn_proj if getattr(args, "attn_proj", None) else "q_proj,k_proj,v_proj,o_proj (default)"
            print(f"  attn_layers        : {attn_str}")
            print(f"  attn_proj          : {proj_str}")
            print("=" * 56, flush=True)

    # ──────────────────────────────────────────
    # Training step
    # ──────────────────────────────────────────

    def step(self, batch: Dict[str, torch.Tensor]):
        input_ids = batch["input_ids"].to(self.model_device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.model_device, non_blocking=True)
        labels = batch["labels"].to(self.model_device, non_blocking=True)

        self.model.train()

        # FIX-H: accumulate() コンテキストで正しい sync タイミングを制御
        with self.accelerator.accumulate(self.model):
            torch.cuda.empty_cache()  # forward 前にキャッシュ解放

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
                output_router_logits=False,
                return_dict=True,
            )
            loss = outputs.loss
            # 協調停止のダミーバッチ（labels 全 -100）は有効トークンが 0 個のため
            # loss が NaN になる。そのまま backward すると NaN 勾配が FSDP の
            # 勾配 all-reduce で全 rank に伝播するので、非有限 loss は
            # logits 経由のゼロ損失（勾配 0、FSDP 通信は同期維持）に差し替える。
            loss_is_finite = bool(torch.isfinite(loss).item())
            if not loss_is_finite:
                loss = outputs.logits.sum() * 0.0
            del outputs, input_ids, attention_mask, labels  # activation 参照を切る

            torch.cuda.empty_cache()  # backward 前にキャッシュ解放
            self.accelerator.backward(loss)

            # FIX-B: sync_gradients でガードしてから clip & step
            if self.accelerator.sync_gradients and self.args.grad_clip > 0:
                self.accelerator.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.args.grad_clip,
                )
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)

        # ダミーバッチ（非有限 loss をゼロ損失に差し替えた場合）は
        # 統計に含めないよう NaN として記録する
        loss_val = float(loss.item()) if loss_is_finite else float("nan")
        del loss
        self._micro_step += 1

        if loss_val == loss_val:  # not NaN
            self._micro_loss_accum += loss_val
            self._micro_loss_count += 1

        gas = self.args.gradient_accumulation_steps
        if self._micro_step % gas != 0:
            return  # accumulation 途中はカウントしない

        self.global_step += 1
        current_step = self.global_step
        accum_loss = self._micro_loss_accum / max(1, self._micro_loss_count)
        self._micro_loss_accum = 0.0
        self._micro_loss_count = 0
        self._loss_accum += accum_loss
        self._loss_count += 1

        step_end = time.time()
        self._step_times.append(step_end)

        if current_step % self.args.log_every == 0 and self.accelerator.is_main_process:
            elapsed = step_end - self._train_start_time
            avg_loss = self._loss_accum / max(1, self._loss_count)

            if len(self._step_times) >= 2:
                window = min(50, len(self._step_times))
                steps_per_sec = (window - 1) / max(self._step_times[-1] - self._step_times[-window], 1e-6)
                sec_per_step = 1.0 / max(steps_per_sec, 1e-6)
            else:
                steps_per_sec = sec_per_step = 0.0

            # ETA / 進捗表示
            # - steps モード: args.steps を分母にパーセントと ETA を表示
            # - epoch モード: エポック内総ステップは事前不明のため経過ステップのみ表示
            eta_str = "?"
            pct_str = ""
            is_epoch_mode = self.args.epochs and self.args.epochs > 0
            if not is_epoch_mode and self.args.steps and self.args.steps > 0:
                pct = current_step / self.args.steps * 100
                pct_str = f" ({pct:.1f}%)"
                remaining = max(0, self.args.steps - current_step)
                if steps_per_sec > 0:
                    eta_str = str(datetime.timedelta(seconds=int(remaining / steps_per_sec)))
            elif is_epoch_mode:
                # 確定済み総ステップ（2エポック目以降）を優先、なければ推定値を使う
                epoch_total = self._epoch_total_steps if self._epoch_total_steps > 0 else self._estimated_epoch_steps
                epoch_step = current_step - self._epoch_start_step
                if epoch_total > 0:
                    pct = min(epoch_step / epoch_total * 100, 100.0)
                    pct_str = f" ({pct:.1f}%{'~' if self._epoch_total_steps == 0 else ''})"
                    remaining = max(0, epoch_total - epoch_step)
                    if steps_per_sec > 0:
                        eta_str = str(datetime.timedelta(seconds=int(remaining / steps_per_sec)))

            epoch_str = f"epoch={self.global_epoch} " if self.args.epochs > 0 else ""
            print(
                f"  {epoch_str}step {current_step}{pct_str} | "
                f"loss={accum_loss:.4f} (avg={avg_loss:.4f}) | "
                f"{sec_per_step:.1f}s/step | "
                f"elapsed={str(datetime.timedelta(seconds=int(elapsed)))} | "
                f"ETA={eta_str}",
                flush=True,
            )

        if self.wandb_run is not None and self.accelerator.is_main_process:
            try:
                import wandb
                self.wandb_run.log(
                    {"loss": accum_loss, "avg_loss": self._loss_accum / max(1, self._loss_count),
                     "epoch": self.global_epoch},
                    step=current_step,
                )
            except Exception as e:
                print(f"[WARN] wandb log failed: {e}", flush=True)

    def _should_stop_by_steps(self) -> bool:
        return (self.args.steps is not None) and (self.args.steps > 0) and (self.global_step >= self.args.steps)

    # ──────────────────────────────────────────
    # Save
    # ──────────────────────────────────────────

    @torch.no_grad()
    def save_model_sharded(self, save_dir: str):
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
                    "format": "full_sharded_model",
                    "pattern": "model_sharded.rank{rank}.pt",
                    "world_size": int(self.world_size),
                    "mask_json": self.args.mask_json,
                    "model": self.args.model,
                    "is_fsdp": True, "type": "SFT",
                }
                with open(os.path.join(save_dir, "model_sharded_meta.json"), "w") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                print(f"[SAVE] full sharded -> {save_dir}", flush=True)
        else:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                unwrapped = self.accelerator.unwrap_model(self.model)
                sd = unwrapped.state_dict()
                path = os.path.join(save_dir, "model.pt")
                n_tensors = len(sd)
                torch.save(sd, path)
                del sd
                gc.collect()
                torch.cuda.empty_cache()
                meta = {"format": "full_model", "model": self.args.model,
                        "is_fsdp": False, "type": "SFT", "num_tensors": n_tensors}
                with open(os.path.join(save_dir, "model_meta.json"), "w") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                print(f"[SAVE] full model -> {path}", flush=True)

    @torch.no_grad()
    def save_updated_params_only(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        def clean_key(k: str) -> str:
            for prefix in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "module."):
                k = k.replace(prefix, "")
            if k.startswith("model."):
                k = k[6:]
            return k

        unwrapped = self.accelerator.unwrap_model(self.model)
        clean_trainable = {clean_key(n) for n, p in unwrapped.named_parameters() if p.requires_grad}

        if self.accelerator.is_main_process:
            print(f"[SAVE-SELECT] trainable param count: {len(clean_trainable)}", flush=True)

        if self.is_fsdp:
            with FSDP.state_dict_type(
                self.model, StateDictType.SHARDED_STATE_DICT,
                ShardedStateDictConfig(offload_to_cpu=True),
            ):
                sd = self.model.state_dict()
            sub = {k: v for k, v in sd.items() if clean_key(k) in clean_trainable}
            del sd
            matched = len(sub)
            if self.accelerator.is_main_process:
                print(f"[SAVE-SELECT] matched {matched} entries in state_dict.", flush=True)
                if matched == 0:
                    print("[WARN] matched_count==0. key mismatch? Use save_model_sharded instead.", flush=True)
            path = os.path.join(save_dir, f"updated_params_sharded.rank{self.rank}.pt")
            torch.save(sub, path)
            del sub
            gc.collect()
            torch.cuda.empty_cache()
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                meta = {"format": "trainable_params_only_sharded",
                        "pattern": "updated_params_sharded.rank{rank}.pt",
                        "world_size": int(self.world_size), "mask_json": self.args.mask_json,
                        "model": self.args.model, "is_fsdp": True, "type": "SFT"}
                with open(os.path.join(save_dir, "updated_params_meta.json"), "w") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                print(f"[SAVE-SELECT] updated-only sharded -> {save_dir}", flush=True)
        else:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                sub = {n: p.detach().cpu() for n, p in unwrapped.named_parameters() if p.requires_grad}
                n_tensors = len(sub)
                path = os.path.join(save_dir, "updated_params.pt")
                torch.save(sub, path)
                del sub
                gc.collect()
                torch.cuda.empty_cache()
                meta = {"format": "trainable_params_only", "model": self.args.model,
                        "is_fsdp": False, "type": "SFT", "num_tensors": n_tensors}
                with open(os.path.join(save_dir, "updated_params_meta.json"), "w") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                print(f"[SAVE-SELECT] updated-only -> {path}", flush=True)

    # ──────────────────────────────────────────
    # Train
    # ──────────────────────────────────────────

    def train(self):
        self.model.train()
        self._train_start_time = time.time()
        self._step_times = [self._train_start_time]
        self._loss_accum = 0.0
        self._loss_count = 0
        self._micro_step = 0
        self._micro_loss_accum = 0.0
        self._micro_loss_count = 0
        self._epoch_total_steps = 0   # エポック内総ステップ（1エポック完了後に確定）
        self._epoch_start_step = 0    # 各エポック開始時の global_step

        if self.args.epochs and self.args.epochs > 0:
            for ep in range(1, self.args.epochs + 1):
                self.global_epoch = ep
                self._epoch_start_step = self.global_step
                ep_start = time.time()
                if self.accelerator.is_main_process:
                    print(f"\n[EPOCH] start {ep}/{self.args.epochs}", flush=True)

                # ── 協調停止: rank 間のデータ枚数が不均等でもハングしないよう、
                #    データが尽きた rank はダミーバッチ（labels 全 -100）で参加し続け、
                #    全 rank が終わったらエポック終了する。
                #    ※ labels 全 -100 だと loss は NaN になるが、step() 側で
                #      非有限 loss をゼロ損失（勾配 0）に差し替えてから backward する。
                data_iter = iter(self.loader)
                _dummy_batch: Optional[Dict[str, torch.Tensor]] = None

                def _make_dummy(ref_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
                    """labels を全 -100 にしたダミーバッチを作る（loss は NaN になるので
                    step() 側でゼロ損失に差し替えられる）。"""
                    b = {k: v.clone() for k, v in ref_batch.items()}
                    b["labels"] = torch.full_like(b["labels"], -100)
                    return b

                # _dummy_batch の初期値を事前に確保（データ0件 rank のクラッシュ回避）
                _pad = self.pad_token_id
                _sl = self.args.seq_len
                _dummy_batch = {
                    "input_ids": torch.full((1, _sl), _pad, dtype=torch.long),
                    "attention_mask": torch.ones(1, _sl, dtype=torch.long),
                    "labels": torch.full((1, _sl), -100, dtype=torch.long),
                }

                while True:
                    try:
                        batch = next(data_iter)
                        _dummy_batch = _make_dummy(batch)  # 最新バッチを dummy の雛形に保存
                        has_data = torch.ones(1, dtype=torch.long, device=self.device)
                    except StopIteration:
                        # ダミーバッチで参加（step() が非有限 loss をゼロ損失に
                        # 差し替えるため勾配への影響なし、FSDP 通信は同期維持）
                        batch = _dummy_batch
                        has_data = torch.zeros(1, dtype=torch.long, device=self.device)

                    # 全 rank の has_data の最大値 → 全員 0 になったら終了
                    torch.distributed.all_reduce(has_data, op=torch.distributed.ReduceOp.MAX)
                    if has_data.item() == 0:
                        break

                    self.step(batch)
                    if self._should_stop_by_steps():
                        break

                ep_steps = self.global_step - self._epoch_start_step
                self._epoch_total_steps = ep_steps  # 次エポックの ETA 計算に使う
                if self.accelerator.is_main_process:
                    ep_elapsed = time.time() - ep_start
                    avg_loss = self._loss_accum / max(1, self._loss_count)
                    print(
                        f"[EPOCH] finished {ep}/{self.args.epochs} | "
                        f"steps={ep_steps} (total={self.global_step}) | avg_loss={avg_loss:.4f} | "
                        f"time={str(datetime.timedelta(seconds=int(ep_elapsed)))}",
                        flush=True,
                    )
                gc.collect()
                torch.cuda.empty_cache()

                if self._should_stop_by_steps():
                    if self.accelerator.is_main_process:
                        print(f"[EPOCH] stopped early by steps cap: {self.args.steps}", flush=True)
                    break
        else:
            if not (self.args.steps and self.args.steps > 0):
                raise RuntimeError("steps mode requires --steps > 0 (or use --epochs).")
            while self.global_step < self.args.steps:
                for batch in self.loader:
                    self.step(batch)
                    if self.global_step >= self.args.steps:
                        break

        if self.accelerator.is_main_process:
            total_elapsed = time.time() - self._train_start_time
            avg_loss = self._loss_accum / max(1, self._loss_count)
            print(
                f"\n[DONE] steps={self.global_step} | avg_loss={avg_loss:.4f} | "
                f"total_time={str(datetime.timedelta(seconds=int(total_elapsed)))}",
                flush=True,
            )

        # ── Save ─────────────────────────────────────────────────────────────
        self.accelerator.wait_for_everyone()

        if getattr(self.args, "save_legacy_only", False):
            if self.accelerator.is_main_process:
                print(f"[SAVE][LEGACY] accelerator.save_state() -> {self.args.save_dir}", flush=True)
            t0 = time.time()
            self.accelerator.save_state(self.args.save_dir)
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                print(f"[SAVE][LEGACY] done. elapsed={time.time() - t0:.1f}s", flush=True)
            return

        if getattr(self.args, "save_updated_only", False):
            t0 = time.time()
            self.save_updated_params_only(self.args.save_dir)
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                print(f"[SAVE][NEW] updated-only done. elapsed={time.time() - t0:.1f}s", flush=True)
        else:
            t0 = time.time()
            self.save_model_sharded(self.args.save_dir)
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                print(f"[SAVE][NEW] full sharded done. elapsed={time.time() - t0:.1f}s", flush=True)

        if getattr(self.args, "save_legacy_after_new", False):
            legacy_dir = os.path.join(self.args.save_dir,
                                      getattr(self.args, "legacy_save_subdir", "legacy_accelerate_state"))
            os.makedirs(legacy_dir, exist_ok=True)
            try:
                self.accelerator.save_state(legacy_dir)
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    print(f"[SAVE][LEGACY-AFTER] done -> {legacy_dir}", flush=True)
            except Exception as e:
                if self.accelerator.is_main_process:
                    print(f"[WARN][LEGACY-AFTER] save_state failed: {e}", flush=True)


# ─────────────────────────────────────────────
# Argparse & main
# ─────────────────────────────────────────────

def build_argparser():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", type=str, default=os.environ.get("BASE_MODEL", ""))
    ap.add_argument("--layers", type=int, default=94)
    ap.add_argument("--experts", type=int, default=128)

    ap.add_argument("--dataset", type=str, default=os.environ.get("SPECIALIST_CHATML_PATH", ""))
    ap.add_argument("--tokenized_dataset", type=str, default=None)

    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--mask_json", type=str, default=None)

    # Attention layer mode
    ap.add_argument(
        "--attn_layers", type=str, default=None,
        help=(
            "学習させる attention 層を指定。"
            "範囲: '0-10' / リスト: '0,1,2,10' / 混在: '0-5,8,10-12'。"
            "指定なし → attention 学習なし (expert mask のみ)。"
            "mask_json と同時指定可能 (OR で両方学習)。"
        ),
    )
    ap.add_argument(
        "--attn_proj", type=str, default=None,
        help=(
            "attn_layers と併用する projection キーワード (カンマ区切り)。"
            "例: 'q_proj,k_proj,v_proj,o_proj' (デフォルト=全部)、"
            "'q_proj,k_proj,v_proj' (o 除く)、'o_proj' (o のみ)。"
        ),
    )

    # Training control
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="Gradient accumulation steps. FIX-A/B が有効化される。")

    # Bucketing
    ap.add_argument("--bucket_buffer", type=int, default=1024)
    ap.add_argument("--packing", action="store_true",
                    help="複数サンプルを seq_len に詰め込むパッキングを有効化。パディングを削減して GPU 効率を最大化。")
    ap.add_argument("--max_batch_len", type=int, default=0,
                    help="Cap padded batch length. 0 disables.")
    ap.add_argument("--num_workers", type=int, default=0)

    # Optimizer
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--optim", type=str, default="adafactor",
                    choices=["adafactor", "adamw", "zero_adamw"])

    # Logging
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="expert_fullparam_sft")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--log_every", type=int, default=10)

    # Save
    ap.add_argument("--save_dir", type=str, default="expert_ckpts_sft")
    ap.add_argument("--save_updated_only", action="store_true")
    ap.add_argument("--save_legacy_only", action="store_true")
    ap.add_argument("--save_legacy_after_new", action="store_true")
    ap.add_argument("--legacy_save_subdir", type=str, default="legacy_accelerate_state")

    # Offline pretokenize
    ap.add_argument("--pretokenize", action="store_true")
    ap.add_argument("--pretokenize_out", type=str, default=None)

    return ap


def main():
    args = build_argparser().parse_args()

    if args.pretokenize:
        out_path = args.pretokenize_out or (args.dataset + f".tok.seqlen{args.seq_len}.jsonl")
        pretokenize_jsonl(args.model, args.dataset, out_path, args.seq_len)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU が必要です。")

    set_seed(42)
    trainer = ExpertFullparamSFTTrainer(args)
    trainer.train()

    if trainer.wandb_run is not None and trainer.accelerator.is_main_process:
        try:
            trainer.wandb_run.finish()
        except Exception as e:
            print(f"[WARN] wandb finish failed: {e}", flush=True)


if __name__ == "__main__":
    main()
