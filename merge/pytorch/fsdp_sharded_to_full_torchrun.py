#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
torchrun(64 proc) で FSDP の SHARDED_STATE_DICT(.pt rankファイル群) から
「壊さずに FULL 相当」を rank0 に復元して HF save_pretrained するスクリプト。

狙い:
- 1プロセスで 64ファイル順次ロードすると I/O/Unpickle で止まりがち
- 各rankが自分のrankファイルをロードし、rank0へ send/recv で断片を流す
- rank0 は base_model の該当パラメータ slice に貼り付けていく（ストリーミング）
- 「完璧」寄り: dim0 連続ブロック分割（典型のFSDP）を strict に検証し、
  欠け/被り/範囲外があれば即エラーで止める（壊さない）

前提:
- checkpoint_dir に model_sharded.rank{0..63}.pt（または updated_params_sharded.rank*.pt）がある
- world_size=64 で torchrun
- base_model とチェックポイントが同一アーキテクチャ
- 典型的な FSDP 分割（dim0方向の連続ブロック）であること

実行例:
  export OMP_NUM_THREADS=1
  torchrun --standalone --nproc_per_node=64 fsdp_sharded_to_full_torchrun.py \
    --checkpoint_dir /path/to/ckpt_dir \
    --base_model /path/to/base_model \
    --out_dir /path/to/out \
    --dtype bf16 \
    --comm_dtype auto \
    --strict_dim0_only 1 \
    --require_all_named_parameters 0 \
    --send_and_free 1

メモ:
- gloo + send/recv は dtype のサポートが環境差あるので、bf16 断片は comm_dtype=auto で fp32 送受信にする（bf16->fp32->bf16は値が壊れない）
- “完全一致(bitwise)”を求めるなら comm_dtype=same が通る環境で（ただし bf16通信が非対応だと落ちます）
"""

from __future__ import annotations

import os
import glob
import argparse
import time
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# ----------------------------
# ShardedTensor unpickle hack (軽量)
# ----------------------------
HACK_ENABLED = False
HackShardedTensor = None  # type: ignore

try:
    import torch.distributed._shard.sharded_tensor.api as st_api

    class HackShardedTensor(torch.Tensor):
        """
        torch.load の unpickle を通すための最小ダミー。
        local_shards を shallow に拾う（深い再帰探索はしない＝高速）
        """
        class ProcessGroupState: pass
        class Metadata: pass

        def __new__(cls, *args, **kwargs):
            return super().__new__(cls)

        def __init__(self):
            self._local_shards = []

        def __setstate__(self, state):
            self._local_shards = []

            def is_shard_list(x: Any) -> bool:
                return isinstance(x, list) and len(x) > 0 and hasattr(x[0], "tensor")

            if isinstance(state, dict):
                for k in ("_local_shards", "local_shards", "_shards", "shards"):
                    v = state.get(k, None)
                    if is_shard_list(v):
                        self._local_shards = v
                        return
                # shallow scan
                for v in state.values():
                    if is_shard_list(v):
                        self._local_shards = v
                        return
                    if isinstance(v, dict):
                        for vv in v.values():
                            if is_shard_list(vv):
                                self._local_shards = vv
                                return
                return

            if isinstance(state, (list, tuple)):
                for it in state:
                    if is_shard_list(it):
                        self._local_shards = it
                        return
                for it in state:
                    if isinstance(it, dict):
                        for v in it.values():
                            if is_shard_list(v):
                                self._local_shards = v
                                return
                return

        def local_shards(self):
            return self._local_shards

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            raise RuntimeError("HackShardedTensor should not participate in torch ops.")

        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            raise RuntimeError("HackShardedTensor should not participate in torch ops.")

    st_api.ShardedTensor = HackShardedTensor
    HACK_ENABLED = True
except Exception:
    pass


def _build_safe_globals() -> List[Any]:
    safe_list: List[Any] = []
    if HackShardedTensor is not None:
        safe_list.append(HackShardedTensor)
    try:
        import torch.distributed._shard.sharded_tensor.shard as st_shard
        for name in ("Shard", "ShardMetadata", "TensorProperties", "ShardedTensorMetadata"):
            if hasattr(st_shard, name):
                safe_list.append(getattr(st_shard, name))
    except Exception:
        pass
    return safe_list


def load_pt(path: str, safe_list: List[Any]) -> Dict[str, Any]:
    """
    torch.load wrapper:
      - weights_only=False を試す
      - safe_globals があれば allowlist 付きで
      - mmap=True が使える環境なら試す（だめならfallback）
    """
    def _torch_load(**kwargs):
        if hasattr(torch.serialization, "safe_globals") and safe_list:
            with torch.serialization.safe_globals(safe_list):
                return torch.load(path, map_location="cpu", **kwargs)
        return torch.load(path, map_location="cpu", **kwargs)

    for use_mmap in (True, False):
        try:
            return _torch_load(weights_only=False, mmap=use_mmap)
        except TypeError as e:
            msg = str(e)
            if "unexpected keyword argument" in msg:
                try:
                    if use_mmap:
                        return _torch_load(mmap=True)
                    return _torch_load()
                except TypeError:
                    continue
            if "weights_only" in msg:
                try:
                    if use_mmap:
                        return _torch_load(mmap=True)
                    return _torch_load()
                except TypeError:
                    continue
            raise
        except Exception:
            if use_mmap:
                continue
            raise

    return _torch_load()


# ----------------------------
# Name/key normalize
# ----------------------------
def clean_name(name: str) -> str:
    name = name.replace("_fsdp_wrapped_module.", "")
    name = name.replace("_checkpoint_wrapped_module.", "")
    name = name.replace("module.", "")
    while name.startswith("model.model."):
        name = "model." + name[len("model.model."):]
    return name


def find_base_key(cleaned: str, base_keys: set[str]) -> Optional[str]:
    cands: List[str] = [cleaned]
    if cleaned.startswith("model."):
        cands.append(cleaned[len("model."):])
    else:
        cands.append("model." + cleaned)

    if cleaned.startswith("model."):
        cands.append("model.model." + cleaned[len("model."):])
    else:
        cands.append("model.model." + cleaned)

    seen = set()
    for x in cands:
        y = x
        while y.startswith("model.model."):
            y = "model." + y[len("model.model."):]
        for z in (x, y):
            if z not in seen:
                seen.add(z)
                if z in base_keys:
                    return z
    return None


# ----------------------------
# dtype helpers
# ----------------------------
def torch_dtype_from_str(dtype_str: str) -> torch.dtype:
    s = dtype_str.strip()
    if s.startswith("torch."):
        key = s.split("torch.", 1)[1]
    else:
        key = s
    if not hasattr(torch, key):
        raise ValueError(f"Unknown dtype string: {dtype_str}")
    return getattr(torch, key)


def pick_comm_dtype(param_dtype: torch.dtype, mode: str) -> torch.dtype:
    """
    mode:
      - auto : bf16 は fp32 で送受信（bf16->fp32->bf16 で壊れない）。それ以外は同dtype。
      - same : param_dtype のまま
      - fp32 : 常に fp32
    """
    if mode == "fp32":
        return torch.float32
    if mode == "same":
        return param_dtype
    # auto
    if param_dtype == torch.bfloat16:
        return torch.float32
    return param_dtype


# ----------------------------
# shard metadata helpers
# ----------------------------
def get_offsets_sizes(meta: Any) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    offsets = None
    sizes = None

    for k in ("shard_offsets", "offsets"):
        if hasattr(meta, k):
            try:
                offsets = [int(x) for x in getattr(meta, k)]
                break
            except Exception:
                pass

    for k in ("shard_sizes", "sizes"):
        if hasattr(meta, k):
            try:
                sizes = [int(x) for x in getattr(meta, k)]
                break
            except Exception:
                pass

    return offsets, sizes


def is_dim0_only_shard(offsets: List[int], sizes: List[int], full_shape: Tuple[int, ...]) -> bool:
    if len(full_shape) == 0:
        return False
    if len(offsets) != len(full_shape) or len(sizes) != len(full_shape):
        return False
    # dims 1.. は offset=0 かつ size=full
    for d in range(1, len(full_shape)):
        if int(offsets[d]) != 0:
            return False
        if int(sizes[d]) != int(full_shape[d]):
            return False
    return True


# ----------------------------
# main distributed merge
# ----------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", type=str, required=True)
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--backend", type=str, default="gloo", choices=["gloo", "nccl"])
    ap.add_argument("--timeout_sec", type=int, default=3600)
    ap.add_argument("--comm_dtype", type=str, default="auto", choices=["auto", "same", "fp32"])

    # strictness knobs
    ap.add_argument("--strict_dim0_only", type=int, default=1, help="1: dim0連続ブロック以外を検出したら即終了")
    ap.add_argument("--require_all_named_parameters", type=int, default=0, help="1: base_modelの全named_parametersがckptに存在しないと終了（tie等で落ちることあり）")

    # perf/mem knobs
    ap.add_argument("--send_and_free", type=int, default=1, help="1: 送信後にそのshard tensor参照を消してメモリを早めに解放")
    ap.add_argument("--log_every", type=int, default=2000, help="rank0がN shardごとに進捗ログ")
    ap.add_argument("--torch_threads", type=int, default=1)

    args = ap.parse_args()

    # threads
    torch.set_num_threads(max(1, int(args.torch_threads)))
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    # dist init
    dist.init_process_group(
        backend=args.backend,
        timeout=timedelta(seconds=int(args.timeout_sec)),
    )
    rank = dist.get_rank()
    world = dist.get_world_size()

    if world != 64 and rank == 0:
        print(f"[WARN] world_size={world}. 想定は64。rankファイル数と一致しているか要確認。", flush=True)

    if rank == 0:
        print("=== torchrun FULL rebuild (strict, rank0 assemble) ===", flush=True)
        print(f"checkpoint_dir : {args.checkpoint_dir}", flush=True)
        print(f"base_model     : {args.base_model}", flush=True)
        print(f"out_dir        : {args.out_dir}", flush=True)
        print(f"dtype          : {args.dtype}", flush=True)
        print(f"backend        : {args.backend}", flush=True)
        print(f"comm_dtype     : {args.comm_dtype}", flush=True)
        print(f"hack_enabled   : {HACK_ENABLED}", flush=True)
        print(f"strict_dim0_only: {int(args.strict_dim0_only)}", flush=True)
        print(f"require_all_named_parameters: {int(args.require_all_named_parameters)}", flush=True)
        print(f"send_and_free  : {int(args.send_and_free)}", flush=True)
        print(f"torch_threads  : {int(args.torch_threads)}", flush=True)

    # find my shard file
    f1 = os.path.join(args.checkpoint_dir, f"model_sharded.rank{rank}.pt")
    f2 = os.path.join(args.checkpoint_dir, f"updated_params_sharded.rank{rank}.pt")

    if os.path.exists(f1):
        my_path = f1
    elif os.path.exists(f2):
        my_path = f2
    else:
        # fallback glob (万一 rank 形式が違う)
        cand = sorted(glob.glob(os.path.join(args.checkpoint_dir, f"*rank{rank}*.pt")))
        if not cand:
            raise FileNotFoundError(f"[rank{rank}] shard file not found: {f1} or {f2}")
        my_path = cand[0]

    safe_list = _build_safe_globals()

    # load my shard dict
    t0 = time.time()
    shard_sd = load_pt(my_path, safe_list)
    dist.barrier()
    if rank == 0:
        print(f"[LOAD] all ranks loaded shard dicts (rank0 load={time.time()-t0:.1f}s) ", flush=True)

    # build local map and local infos
    # local_map key:
    #   sharded: (clean_name, tuple(offsets)) -> tensor (contig CPU)
    #   full:    (clean_name, ("FULL",))      -> tensor (rank0 only)
    local_map: Dict[Tuple[str, Tuple[Any, ...]], torch.Tensor] = {}
    local_infos: List[Dict[str, Any]] = []

    for raw_name, obj in shard_sd.items():
        name = clean_name(str(raw_name))

        # sharded tensor
        local_shards = None
        if hasattr(obj, "local_shards") and callable(getattr(obj, "local_shards")):
            try:
                local_shards = obj.local_shards()
            except Exception:
                local_shards = None

        if isinstance(local_shards, list):
            for sh in local_shards:
                st = getattr(sh, "tensor", None)
                meta = getattr(sh, "metadata", None)
                if st is None or meta is None or (not torch.is_tensor(st)):
                    continue
                offsets, sizes = get_offsets_sizes(meta)
                if offsets is None or sizes is None:
                    continue
                key = (name, tuple(offsets))
                t = st.detach().contiguous().cpu()
                local_map[key] = t
                local_infos.append({
                    "kind": "sharded",
                    "name": name,
                    "offsets": offsets,
                    "sizes": sizes,
                    "dtype": str(t.dtype),
                    "numel": int(t.numel()),
                    "owner": int(rank),
                })
            continue

        # plain tensor (replicated/unsharded)
        if torch.is_tensor(obj):
            # 重複送信を避ける：rank0だけ owner として扱う
            if rank == 0:
                t = obj.detach().contiguous().cpu()
                key = (name, ("FULL",))
                local_map[key] = t
                local_infos.append({
                    "kind": "full",
                    "name": name,
                    "dtype": str(t.dtype),
                    "numel": int(t.numel()),
                    "owner": 0,
                })
            continue

        # ignore unknown entries

    # free big dict container early (tensor refs are in local_map)
    del shard_sd

    # gather infos to rank0
    gathered: List[Any] = [None for _ in range(world)]
    dist.all_gather_object(gathered, local_infos)

    # rank0: load base model and build schedule (strict validate)
    schedule: List[Dict[str, Any]] = []
    name_to_base: Dict[str, str] = {}
    base_param_keys: set[str] = set()
    base_state = None
    base_model = None

    if rank == 0:
        # dtype
        if args.dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif args.dtype == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        os.makedirs(args.out_dir, exist_ok=True)

        print("[BASE] Loading base model on rank0 (CPU)...", flush=True)
        tb = time.time()
        config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
        try:
            config.torch_dtype = torch_dtype
        except Exception:
            pass

        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            config=config,
            dtype=torch_dtype,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        base_model.eval()
        base_state = base_model.state_dict()
        base_keys = set(base_state.keys())
        base_param_keys = set(n for n, _ in base_model.named_parameters())

        print(f"[BASE] loaded in {time.time()-tb:.1f}s. base_params={len(base_param_keys)} base_state_keys={len(base_keys)}", flush=True)

        # flatten gathered infos
        all_infos: List[Dict[str, Any]] = []
        for li in gathered:
            if li:
                all_infos.extend(li)

        # group by name
        by_name: Dict[str, List[Dict[str, Any]]] = {}
        for it in all_infos:
            by_name.setdefault(it["name"], []).append(it)

        # build schedule param-by-param (strict)
        for name, items in by_name.items():
            # map to base key
            if name not in name_to_base:
                bk = find_base_key(name, base_keys)
                if bk is None:
                    # ここで止めるのが “壊さない” 方針
                    raise KeyError(f"[STRICT] checkpoint param '{name}' not found in base_model.state_dict()")
                name_to_base[name] = bk
            bk = name_to_base[name]
            tgt = base_state[bk]
            full_shape = tuple(int(x) for x in tgt.shape)

            # split kinds
            fulls = [x for x in items if x.get("kind") == "full"]
            shds  = [x for x in items if x.get("kind") == "sharded"]

            if fulls and shds:
                raise RuntimeError(f"[STRICT] mixed full+sharded for param: {name}")

            if fulls:
                # rank0 only by construction
                it = fulls[0]
                # strict: sizes should match exactly
                if int(it["numel"]) != int(tgt.numel()):
                    raise RuntimeError(f"[STRICT] full tensor numel mismatch {name}: ckpt={it['numel']} base={tgt.numel()}")
                schedule.append({
                    "kind": "full",
                    "name": name,
                    "base_key": bk,
                    "owner": 0,
                    "dtype": it["dtype"],
                    "numel": int(it["numel"]),
                    "is_last": True,
                })
                continue

            # sharded
            if not shds:
                # nothing to do
                continue

            # validate each shard entry
            norm: List[Dict[str, Any]] = []
            for it in shds:
                offsets = list(it["offsets"])
                sizes   = list(it["sizes"])
                if len(offsets) != len(full_shape) or len(sizes) != len(full_shape):
                    raise RuntimeError(f"[STRICT] ndim mismatch {name}: offsets={len(offsets)} sizes={len(sizes)} base_ndim={len(full_shape)}")
                # bounds
                for d, (o, s, fs) in enumerate(zip(offsets, sizes, full_shape)):
                    if o < 0 or s < 0 or (o + s) > fs:
                        raise RuntimeError(f"[STRICT] OOB shard {name}: dim{d} off={o} size={s} full={fs}")
                if int(args.strict_dim0_only):
                    if not is_dim0_only_shard(offsets, sizes, full_shape):
                        raise RuntimeError(f"[STRICT] non-dim0 shard pattern detected for {name}. disable --strict_dim0_only to allow (but then 'perfect'保証は弱くなる).")

                norm.append({
                    "name": name,
                    "base_key": bk,
                    "owner": int(it["owner"]),
                    "offsets": offsets,
                    "sizes": sizes,
                    "dtype": it["dtype"],
                    "numel": int(it["numel"]),
                    "off0": int(offsets[0]),
                    "sz0": int(sizes[0]),
                })

            # strict coverage for dim0-only: contiguous partition along dim0
            norm = sorted(norm, key=lambda x: (x["off0"], x["owner"]))
            expect = 0
            for idx, it in enumerate(norm):
                if it["off0"] != expect:
                    raise RuntimeError(
                        f"[STRICT] gap/overlap in dim0 coverage for {name}: expected_off0={expect} got={it['off0']}"
                    )
                expect += it["sz0"]
            if expect != int(full_shape[0]):
                raise RuntimeError(f"[STRICT] dim0 coverage incomplete for {name}: covered={expect} full={full_shape[0]}")

            # schedule entries
            for j, it in enumerate(norm):
                # choose comm dtype per param dtype
                param_dtype = torch_dtype_from_str(it["dtype"])
                cd = pick_comm_dtype(param_dtype, args.comm_dtype)
                schedule.append({
                    "kind": "sharded",
                    "name": name,
                    "base_key": bk,
                    "owner": int(it["owner"]),
                    "offsets": it["offsets"],
                    "sizes": it["sizes"],
                    "dtype": it["dtype"],             # original dtype
                    "comm_dtype": str(cd),
                    "numel": int(it["numel"]),
                    "is_last": (j == len(norm) - 1),
                })

        if not schedule:
            raise RuntimeError("[STRICT] schedule is empty. (name cleaning mismatch or checkpoint has no params?)")

        # optional: require all named_parameters present
        if int(args.require_all_named_parameters):
            present = set()
            for it in schedule:
                present.add(it["base_key"])
            missing = sorted(list(base_param_keys - present))
            if missing:
                raise RuntimeError(
                    f"[STRICT] base_model has named_parameters not present in checkpoint (count={len(missing)}). "
                    f"Example: {missing[:5]}"
                )

        print(f"[SCHED] schedule_len={len(schedule)} unique_ckpt_names={len(by_name)}", flush=True)

    # broadcast schedule to all ranks
    obj = [schedule]
    dist.broadcast_object_list(obj, src=0)
    schedule = obj[0]
    dist.barrier()

    if rank == 0:
        print("[RUN] start streaming recv/assemble...", flush=True)

    # rank0: apply shards, others: send
    completed_params = 0
    last_name = None

    # rank0 safety
    assert (rank != 0) or (base_state is not None and base_model is not None)

    for idx, it in enumerate(schedule):
        kind = it["kind"]
        name = it["name"]

        if rank == 0 and (name != last_name):
            last_name = name

        if kind == "full":
            # rank0 only copy from local_map
            if rank == 0:
                key = (name, ("FULL",))
                src = local_map.get(key, None)
                if src is None:
                    raise KeyError(f"[STRICT] rank0 missing full tensor in local_map for {name}")
                tgt = base_state[it["base_key"]]
                if src.numel() != tgt.numel():
                    raise RuntimeError(f"[STRICT] full copy numel mismatch {name}")
                # dtype convert if needed
                if src.dtype != tgt.dtype:
                    src = src.to(dtype=tgt.dtype)
                tgt.copy_(src.reshape_as(tgt))
                if int(args.send_and_free):
                    del local_map[key]
                completed_params += 1
            continue

        # sharded
        owner = int(it["owner"])
        offsets = list(it["offsets"])
        sizes = list(it["sizes"])
        numel = int(it["numel"])
        comm_dtype = torch_dtype_from_str(it["comm_dtype"])

        # owner sends
        if rank == owner and owner != 0:
            key = (name, tuple(offsets))
            t = local_map.get(key, None)
            if t is None:
                raise KeyError(f"[STRICT] owner rank{owner} missing shard tensor for {name} offsets={offsets}")
            flat = t.reshape(-1)
            if flat.numel() != numel:
                raise RuntimeError(f"[STRICT] shard numel mismatch send {name}: flat={flat.numel()} meta={numel}")
            if flat.dtype != comm_dtype:
                flat = flat.to(dtype=comm_dtype)
            dist.send(flat, dst=0)
            if int(args.send_and_free):
                del local_map[key]

        # rank0 receives and applies
        if rank == 0:
            tgt = base_state[it["base_key"]]
            buf = torch.empty((numel,), dtype=comm_dtype, device="cpu")
            if owner == 0:
                # owner=0 の shard は rank0 が持っている（自分のlocal_mapから直接）
                key = (name, tuple(offsets))
                src = local_map.get(key, None)
                if src is None:
                    raise KeyError(f"[STRICT] rank0 missing shard tensor for {name} offsets={offsets}")
                flat = src.reshape(-1)
                if flat.numel() != numel:
                    raise RuntimeError(f"[STRICT] shard numel mismatch self {name}")
                if flat.dtype != comm_dtype:
                    flat = flat.to(dtype=comm_dtype)
                buf.copy_(flat)
                if int(args.send_and_free):
                    del local_map[key]
            else:
                dist.recv(buf, src=owner)

            # cast back to target dtype (bf16/...)
            if buf.dtype != tgt.dtype:
                buf2 = buf.to(dtype=tgt.dtype)
            else:
                buf2 = buf

            # reshape to sizes and slice apply
            if buf2.numel() != numel:
                raise RuntimeError(f"[STRICT] recv numel mismatch {name}")

            view = buf2.reshape(*sizes)

            sl = tuple(slice(int(o), int(o) + int(s)) for o, s in zip(offsets, sizes))
            tgt[sl].copy_(view)

            if bool(it.get("is_last", False)):
                completed_params += 1

            if (idx + 1) % int(args.log_every) == 0:
                print(f"[PROG] {idx+1}/{len(schedule)} shards done, completed_params={completed_params}", flush=True)

    dist.barrier()

    # rank0 save
    if rank == 0:
        print(f"[DONE] assembled. completed_params={completed_params}. Saving to {args.out_dir}", flush=True)
        base_model.save_pretrained(args.out_dir, safe_serialization=True)
        try:
            tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
            tok.save_pretrained(args.out_dir)
            print("[SAVE] tokenizer saved.", flush=True)
        except Exception as e:
            print(f"[WARN] tokenizer save failed: {e}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
