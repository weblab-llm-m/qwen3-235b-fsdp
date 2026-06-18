#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
1ノード CPU で以下を行うスクリプト:

1. FSDP のシャードチェックポイント (accelerator.save_state の出力) から
   torch.distributed.checkpoint を使って「選択された expert パラメータだけ」
   base モデルに in-place でロード
2. 学習した重み（＝selected experts）が想定どおりのキー名かどうか、
   また全てちゃんと焼き込めているかを fingerprint で確認
3. 焼き込み済みベースを save_pretrained で保存
"""

import os
import re
import json
import math
import argparse
from typing import Optional, Tuple, List, Dict

import torch
import torch.distributed.checkpoint as dist_cp
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer


# ====== expert 判定まわり (学習スクリプトと同じロジック) ======

_EXPERT_RE = re.compile(r"\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")


def parse_layer_expert_from_name(name: str) -> Optional[Tuple[int, int]]:
    m = _EXPERT_RE.search(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def load_mask(mask_json: Optional[str], n_layers: int, n_experts: int) -> Optional[List[torch.Tensor]]:
    if not mask_json:
        return None
    print(f"[MASK] loading mask_json={mask_json}", flush=True)
    with open(mask_json, "r", encoding="utf-8") as f:
        obj = json.load(f)

    layer_masks: List[torch.Tensor] = [torch.zeros(n_experts, dtype=torch.bool) for _ in range(n_layers)]

    sel = obj.get("selected_layer_experts")
    if sel:
        print("[MASK] using field: selected_layer_experts", flush=True)
        for it in sel:
            li = int(it["layer_idx"])
            ei = int(it["expert_id"])
            if 0 <= li < n_layers and 0 <= ei < n_experts:
                layer_masks[li][ei] = True
    else:
        ex = obj.get("selected_experts_unique") or obj.get("selected_experts") or obj.get("picked_experts")
        if not ex:
            raise RuntimeError("Mask JSON lacks selected_* fields.")
        print("[MASK] using field: selected_experts(_unique/picked)", flush=True)
        for li in range(n_layers):
            for ei in ex:
                ei = int(ei)
                if 0 <= ei < n_experts:
                    layer_masks[li][ei] = True

    total_on = int(sum(m.sum().item() for m in layer_masks))
    print(f"[MASK] total selected expert entries = {total_on}", flush=True)

    for li in range(n_layers):
        if layer_masks[li].sum().item() == 0:
            layer_masks[li] = torch.ones(n_experts, dtype=torch.bool)
            print(f"[MASK] layer {li}: no experts selected in mask, enabling all experts.", flush=True)

    return layer_masks


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


# ====== FSDP シャードディレクトリ探索 ======

def find_fsdp_model_dir(ckpt_dir: str) -> str:
    """accelerator.save_state() で作られたディレクトリの中から
    FSDP モデルシャード (pytorch_model_fsdp_0 など) を探す。
    """
    print(f"[SCAN] checkpoint_dir = {ckpt_dir}", flush=True)
    if not os.path.isdir(ckpt_dir):
        raise RuntimeError(f"checkpoint_dir does not exist or is not a dir: {ckpt_dir}")

    entries = sorted(os.listdir(ckpt_dir))
    print(f"[SCAN] contents of checkpoint_dir ({len(entries)} entries):", flush=True)
    for e in entries:
        print(f"  - {e}", flush=True)

    cand: List[str] = []
    for name in entries:
        full = os.path.join(ckpt_dir, name)
        if not os.path.isdir(full):
            continue
        if name.startswith("pytorch_model_fsdp_") or name.startswith("pytorch_model_"):
            cand.append(full)

    if not cand:
        raise RuntimeError(f"No FSDP model dir found under {ckpt_dir}")

    cand.sort()
    if len(cand) > 1:
        print(f"[SCAN] multiple FSDP model dirs found:", flush=True)
        for c in cand:
            print(f"  * {c}", flush=True)
        print(f"[SCAN] -> using first: {cand[0]}", flush=True)
    else:
        print(f"[SCAN] found FSDP model dir: {cand[0]}", flush=True)

    # ここが FileSystemReader に渡すパス (中に __0_0.distcp などがある)
    return cand[0]


# ====== DCP を使って「選択 expert」だけロード ======

def compute_fingerprint(t: torch.Tensor, max_elems: int = 1024) -> float:
    """テンソルが変わったかどうかの簡易判定用 fingerprint（平均値）"""
    if t.numel() == 0:
        return 0.0
    flat = t.detach().view(-1)
    n = min(flat.numel(), max_elems)
    return float(flat[:n].float().mean().item())


def load_selected_experts_from_fsdp(
    base_model: torch.nn.Module,
    fsdp_dir: str,
    layer_masks: List[torch.Tensor],
) -> Dict[str, float]:
    """FSDP の分散チェックポイントから「選択 expert パラメータだけ」を
    base_model に in-place で読み込む。

    戻り値: updated_fingerprint[k] = new_fingerprint（後で before/after 比較用）
    """
    base_state = base_model.state_dict()
    num_total = len(base_state)

    # どの key が「選択 expert」かを先にフィルタ
    selected_keys: List[str] = [k for k in base_state.keys() if is_selected_expert_param(k, layer_masks)]
    if not selected_keys:
        raise RuntimeError("No selected expert parameters found in base model state_dict. Mask/name pattern mismatch?")

    selected_keys_sorted = sorted(selected_keys)
    print(f"[CHECK] #selected expert params in BASE model state_dict: {len(selected_keys_sorted)} / {num_total}", flush=True)
    for i, k in enumerate(selected_keys_sorted[:50]):
        print(f"  [BASE_SEL {i:04d}] {k} shape={tuple(base_state[k].shape)}", flush=True)
    if len(selected_keys_sorted) > 50:
        print(f"  ... ({len(selected_keys_sorted) - 50} more)", flush=True)

    # まず before fingerprint を記録（後で「本当に値が変わったか」チェック用）
    before_fp: Dict[str, float] = {}
    print("[CHECK] computing fingerprints BEFORE DCP load...", flush=True)
    for i, k in enumerate(selected_keys_sorted):
        before_fp[k] = compute_fingerprint(base_state[k])
        if i < 5:
            print(f"  [FP_BEFORE {i:04d}] {k} -> {before_fp[k]:+.6e}", flush=True)

    # DCP 用 state_dict 構造: {"model": {selected_key: tensor, ...}}
    selected_state = {k: base_state[k] for k in selected_keys_sorted}
    state_dict_for_dcp = {"model": selected_state}

    print(f"[DCP] loading from FSDP checkpoint dir: {fsdp_dir}", flush=True)
    # FileSystemReader を使って FSDP シャードを読む
    reader = dist_cp.FileSystemReader(fsdp_dir)
    dist_cp.load_state_dict(
        state_dict=state_dict_for_dcp,
        storage_reader=reader,
        no_dist=True,  # 単一プロセスで完結
    )
    print("[DCP] load_state_dict() finished. Selected expert tensors in base_model are now updated (in-place).", flush=True)

    # AFTER fingerprint を計算して updated / unchanged を判定
    updated_keys: List[str] = []
    unchanged_keys: List[str] = []
    after_fp: Dict[str, float] = {}

    print("[CHECK] computing fingerprints AFTER DCP load...", flush=True)
    for i, k in enumerate(selected_keys_sorted):
        fp_new = compute_fingerprint(base_state[k])
        after_fp[k] = fp_new
        fp_old = before_fp[k]

        # ★ ここがパッチ: 単純な「値が変わったかどうか」判定に変更
        if fp_old != fp_new:
            updated_keys.append(k)
        else:
            unchanged_keys.append(k)

        if i < 5:
            print(f"  [FP_AFTER  {i:04d}] {k} -> {fp_new:+.6e} (delta={fp_new - fp_old:+.3e})", flush=True)

    print(f"[CHECK] #selected expert params total : {len(selected_keys_sorted)}", flush=True)
    print(f"[CHECK] #updated (fingerprint changed) : {len(updated_keys)}", flush=True)
    print(f"[CHECK] #unchanged (fingerprint same)  : {len(unchanged_keys)}", flush=True)

    if unchanged_keys:
        print("[WARN] Some selected expert params did NOT change after DCP load. Candidates that might not be loaded:", flush=True)
        for i, k in enumerate(unchanged_keys[:30]):
            print(f"  [UNCHANGED {i:04d}] {k} (fp_before={before_fp[k]:+.6e}, fp_after={after_fp[k]:+.6e})", flush=True)
        if len(unchanged_keys) > 30:
            print(f"  ... ({len(unchanged_keys) - 30} more)", flush=True)
    else:
        print("[CHECK] OK: All selected expert params changed fingerprint -> very likely all baked from checkpoint.", flush=True)

    return after_fp


# ====== メイン処理 ======

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", type=str, required=True,
                    help="accelerator.save_state() で保存したディレクトリ (例: outputs/ckpts_qwen3_235b_expert_grpo)")
    ap.add_argument("--base_model", type=str, required=True,
                    help="ベースモデルの HF ローカルパス")
    ap.add_argument("--mask_json", type=str, required=True,
                    help="selected experts 情報を持つ JSON")
    ap.add_argument("--layers", type=int, default=94)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--out_dir", type=str, required=True,
                    help="焼き込み済みベースを保存するディレクトリ")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"],
                    help="ベースモデル読み込み時の dtype (CPU)")
    args = ap.parse_args()

    print("=== Qwen3-235B Expert Merge & Bake (CPU, DCP-Direct) ===", flush=True)
    print(f"checkpoint_dir : {args.checkpoint_dir}", flush=True)
    print(f"base_model     : {args.base_model}", flush=True)
    print(f"mask_json      : {args.mask_json}", flush=True)
    print(f"out_dir        : {args.out_dir}", flush=True)
    print(f"dtype          : {args.dtype}", flush=True)
    print(f"layers/experts : {args.layers}/{args.experts}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) マスク読み込み
    layer_masks = load_mask(args.mask_json, n_layers=args.layers, n_experts=args.experts)
    print("[INFO] coverage-mask loaded.", flush=True)

    # 2) FSDP モデルシャードディレクトリ検出 (pytorch_model_fsdp_0 など)
    print("[STEP] locate FSDP model shard dir under checkpoint_dir", flush=True)
    fsdp_dir = find_fsdp_model_dir(args.checkpoint_dir)
    print(f"[STEP] FSDP model dir = {fsdp_dir}", flush=True)

    # 3) ベースモデルを CPU にロード
    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    print("[LOAD] Loading base model to CPU...", flush=True)
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    config.torch_dtype = torch_dtype

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        torch_dtype=torch_dtype,
        device_map={"": "cpu"},
        low_cpu_mem_usage=False,   # ここではフルモデルを素直に展開
        trust_remote_code=True,
    )
    base_model.eval()  # ここでは学習しないので eval で OK

    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"[INFO] Base model params: {total_params:,}", flush=True)

    # 4) FSDP チェックポイントから「選択 expert」だけ in-place で読み込み
    print("[MERGE] Injecting selected expert weights from FSDP checkpoint into base model...", flush=True)
    after_fp = load_selected_experts_from_fsdp(base_model, fsdp_dir, layer_masks)
    num_selected = len(after_fp)
    print(f"[MERGE] DONE. Updated fingerprints for {num_selected} expert tensors in base model.", flush=True)

    # 5) 焼き込み済みベースを save_pretrained
    print(f"[SAVE] Saving baked base model to: {args.out_dir}", flush=True)
    base_model.save_pretrained(args.out_dir, safe_serialization=True)
    try:
        tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        tok.save_pretrained(args.out_dir)
        print("[SAVE] Tokenizer saved.", flush=True)
    except Exception as e:
        print(f"[WARN] tokenizer save failed: {e}", flush=True)

    meta = {
        "checkpoint_dir": args.checkpoint_dir,
        "base_model": args.base_model,
        "mask_json": args.mask_json,
        "layers": args.layers,
        "experts": args.experts,
        "dtype": args.dtype,
        "num_expert_tensors_updated": num_selected,
        "note": "Expert weights loaded directly from FSDP DCP shards via torch.distributed.checkpoint.load_state_dict(storage_reader=...).",
    }
    meta_path = os.path.join(args.out_dir, "expert_merge_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] expert_merge_meta.json written to {meta_path}", flush=True)
    print("=== Expert Merge & Bake DONE ===", flush=True)


if __name__ == "__main__":
    main()
