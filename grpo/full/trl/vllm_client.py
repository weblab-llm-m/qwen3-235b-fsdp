# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from trl/extras/vllm_client.py (HuggingFace TRL).
# NOTE: FSDP Prefix Cleaning + Debug Trace (No TQDM)
# NOTE: B-Plan Fix + Perf:
#   - rank0 builds a shard schedule, broadcasts it to all FSDP ranks,
#     and all ranks call broadcast in identical order/count with identical src.
#   - NO per-shard barrier
#   - NO FSDP CPU offload
#   - Batch metadata API support: /update_param_shards/ (fallback to legacy)

from __future__ import annotations

import atexit
import base64
import copy
import io
import logging
import time
from typing import Any
from urllib.parse import urlparse

import requests
import torch
import torch.distributed as dist
from PIL import Image
from torch import nn

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

try:
    from vllm.utils import is_torch_xpu_available
except Exception:
    def is_torch_xpu_available() -> bool:
        if not hasattr(torch, "xpu"):
            return False
        return getattr(torch.xpu, "is_available", lambda: False)()

from trl.import_utils import is_requests_available, is_vllm_available

logger = logging.getLogger(__name__)


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class VLLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        host: str = "0.0.0.0",
        server_port: int = 8000,
        group_port: int = 51216,
        connection_timeout: float = 0.0,
    ):
        if not is_requests_available():
            raise ImportError("requests is not installed. Please install it with `pip install requests`.")
        if not is_vllm_available():
            raise ImportError("vLLM is not installed. Please install it with `pip install trl[vllm]`.")

        self.session = requests.Session()

        if base_url is not None:
            parsed_url = urlparse(base_url)
            scheme = parsed_url.scheme or "http"
            hostname = parsed_url.hostname or host
            port = parsed_url.port or server_port
            netloc = parsed_url.netloc
            if not netloc:
                netloc = f"{hostname}:{port}"
            self.host = hostname
            self.server_port = port
            self.base_url = f"{scheme}://{netloc}{parsed_url.path or ''}"
        else:
            self.host = host
            self.server_port = server_port
            self.base_url = f"http://{self.host}:{self.server_port}"

        self.group_port = group_port
        self.check_server(connection_timeout)

        self.broadcast_device: torch.device | str | int | None = None
        self.fsdp_rank: int | None = None
        self.fsdp_world_size: int | None = None
        self.rank: int | None = None

        # cache
        self.vllm_world_size: int | None = None

        # communicator handles
        self.communicator = None

        # buffer cache for fsdp streaming
        self._buffer_cache: dict[tuple[str, int, str], torch.Tensor] = {}

    def check_server(self, total_timeout: float = 0.0, retry_interval: float = 2.0):
        url = f"{self.base_url}/health/"
        start_time = time.time()
        while True:
            try:
                response = requests.get(url)
            except requests.exceptions.RequestException as exc:
                elapsed_time = time.time() - start_time
                if elapsed_time >= total_timeout:
                    raise ConnectionError(
                        f"The vLLM server can't be reached at {self.base_url} after {total_timeout} seconds."
                    ) from exc
            else:
                if response.status_code == 200:
                    if "X-Forwarded-For" in response.headers:
                        self.host = response.headers["X-Forwarded-For"]
                    logger.info("Server is up!")
                    return None
            logger.info(f"Server is not up yet. Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)

    def generate(self, prompts: list[str], images: list | None = None, **kwargs) -> dict[str, list[list[int]]]:
        url = f"{self.base_url}/generate/"
        images = [pil_to_base64(img) for img in images] if images else None
        payload = {"prompts": prompts, "images": images}
        for k in [
            "n", "repetition_penalty", "temperature", "top_p", "top_k", "min_p",
            "max_tokens", "truncate_prompt_tokens", "guided_decoding_regex"
        ]:
            if k in kwargs:
                payload[k] = kwargs[k]
        if "generation_kwargs" in kwargs:
            payload["generation_kwargs"] = kwargs["generation_kwargs"]

        response = self.session.post(url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def chat(self, messages: list[list[dict]], **kwargs) -> dict[str, list[list[int]]]:
        url = f"{self.base_url}/chat/"
        msgs_copy = copy.deepcopy(messages)
        for ml in msgs_copy:
            for m in ml:
                if isinstance(m["content"], list):
                    for part in m["content"]:
                        if part.get("type") == "image_pil":
                            part["image_pil"] = pil_to_base64(part["image_pil"])
        payload = {"messages": msgs_copy}
        for k in [
            "n", "repetition_penalty", "temperature", "top_p", "top_k", "min_p",
            "max_tokens", "truncate_prompt_tokens", "guided_decoding_regex",
            "chat_template_kwargs"
        ]:
            if k in kwargs:
                payload[k] = kwargs[k]
        if "generation_kwargs" in kwargs:
            payload["generation_kwargs"] = kwargs["generation_kwargs"]

        response = self.session.post(url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def init_communicator(self, device: torch.device | str | int = 0):
        if dist.is_available() and dist.is_initialized():
            fsdp_rank = dist.get_rank()
            fsdp_world_size = dist.get_world_size()
        else:
            fsdp_rank = 0
            fsdp_world_size = 1
        self.fsdp_rank = fsdp_rank
        self.fsdp_world_size = fsdp_world_size

        url = f"{self.base_url}/get_world_size/"
        response = requests.get(url)
        if response.status_code == 200:
            vllm_world_size = response.json()["world_size"]
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        self.vllm_world_size = int(vllm_world_size)

        world_size = self.vllm_world_size + fsdp_world_size
        self.rank = self.vllm_world_size + fsdp_rank

        if fsdp_rank == 0:
            if is_torch_xpu_available():
                uuid = (
                    str(torch.xpu.get_device_properties(device).uuid)
                    if hasattr(torch.xpu.get_device_properties(device), "uuid")
                    else "42"
                )
            else:
                uuid = str(torch.cuda.get_device_properties(device).uuid)

            # IMPORTANT: use resolved host, not 0.0.0.0
            resp = self.session.post(
                f"{self.base_url}/init_communicator/",
                json={
                    "host": self.host,
                    "port": self.group_port,
                    "world_size": world_size,
                    "client_device_uuid": uuid,
                },
            )
            if resp.status_code != 200:
                raise Exception(f"Request failed: {resp.status_code}, {resp.text}")
            time.sleep(0.1)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        if is_torch_xpu_available():
            store = torch.distributed.TCPStore(
                host_name=self.host,
                port=self.group_port,
                world_size=world_size,
                is_master=(self.rank == 0),
            )
            pg = torch.distributed.ProcessGroupXCCL(
                store=torch.distributed.PrefixStore("client2server", store),
                rank=self.rank,
                size=world_size,
            )
            self.communicator = pg
            self.broadcast_device = None
        else:
            pg = StatelessProcessGroup.create(
                host=self.host,
                port=self.group_port,
                rank=self.rank,
                world_size=world_size,
            )
            self.communicator = PyNcclCommunicator(pg, device=device)
            self.broadcast_device = torch.device(device) if isinstance(device, (str, int)) else device

        atexit.register(self.close_communicator)

    def _move_to_broadcast_device(self, t: torch.Tensor) -> torch.Tensor:
        t = t.detach()
        if not is_torch_xpu_available() and self.broadcast_device is not None:
            if t.device != self.broadcast_device:
                t = t.to(self.broadcast_device)
        return t.contiguous()

    # NO per-shard barrier
    def _broadcast_flat(self, t: torch.Tensor, src_rank: int):
        if self.communicator is None:
            raise RuntimeError("Call init_communicator first.")
        if is_torch_xpu_available():
            self.communicator.broadcast(t, root=src_rank)
        else:
            self.communicator.broadcast(t, src=src_rank)

    def update_named_param(self, name: str, weights: torch.Tensor):
        if self.rank is None:
            raise RuntimeError("Call init_communicator first.")

        dtype, shape = str(weights.dtype), tuple(weights.shape)
        resp = self.session.post(
            f"{self.base_url}/update_named_param/",
            json={"name": name, "dtype": dtype, "shape": shape},
        )
        if resp.status_code != 200:
            raise Exception(f"Request failed: {resp.status_code}, {resp.text}")

        tensor = self._move_to_broadcast_device(weights)
        self._broadcast_flat(tensor, src_rank=int(self.rank))

    def _is_router_param_name(self, name: str) -> bool:
        lname = name.lower()
        if lname.endswith("mlp.gate.weight") or ".mlp.gate.weight" in lname:
            return True
        if "router" in lname and "router_logits" not in lname:
            return True
        if "routing" in lname:
            return True
        return False

    def _is_expert_param_name(self, name: str) -> bool:
        lname = name.lower()
        if ".experts." in lname or "ffn_experts." in lname or "expert_layers." in lname:
            return True
        return False

    def update_model_params(
        self,
        model: nn.Module,
        router_only: bool = False,
        expert_only: bool = False,
        trainable_only: bool = False,
    ):
        for name, param in model.named_parameters():
            if trainable_only and not param.requires_grad:
                continue
            if router_only and not self._is_router_param_name(name):
                continue
            if expert_only and not (self._is_expert_param_name(name) or self._is_router_param_name(name)):
                continue
            self.update_named_param(name, param.data)

    # --- 名前の正規化関数 ---
    def _clean_name(self, name: str) -> str:
        name = name.replace("_fsdp_wrapped_module.", "")
        name = name.replace("_checkpoint_wrapped_module.", "")
        return name

    # --- offset/stride helpers ---
    def _calc_offset_1d(self, offsets: list[int], full_shape: list[int]) -> int:
        if not offsets:
            return 0
        if not full_shape:
            return int(offsets[0]) if offsets else 0
        if len(full_shape) == 1:
            return int(offsets[0])

        # row-major strides
        strides = []
        prod = 1
        for d in reversed(full_shape[1:]):
            prod *= int(d)
            strides.append(prod)
        strides = list(reversed(strides))
        strides.append(1)

        off = 0
        for o, s in zip(offsets, strides):
            off += int(o) * int(s)
        return int(off)

    def _torch_dtype_from_str(self, dtype_str: str) -> torch.dtype:
        s = dtype_str.strip()
        if s.startswith("torch."):
            key = s.split("torch.", 1)[1]
        else:
            key = s
        if not hasattr(torch, key):
            raise ValueError(f"Unknown dtype string: {dtype_str}")
        return getattr(torch, key)

    def _get_full_shape_from_obj(self, obj: Any) -> list[int]:
        if hasattr(obj, "size") and callable(obj.size):
            try:
                return [int(x) for x in obj.size()]
            except Exception:
                pass
        if hasattr(obj, "metadata") and hasattr(obj.metadata, "tensor_sizes"):
            try:
                return [int(x) for x in obj.metadata.tensor_sizes]
            except Exception:
                pass
        if hasattr(obj, "shape"):
            try:
                return [int(x) for x in obj.shape]
            except Exception:
                pass
        return []

    # --- Batch metadata posting ---
    def _try_post_shard_metadata_batch(self, schedule: list[dict[str, Any]]) -> bool:
        """
        Try new batch endpoint.
        Expected server-side payload:
          { "shards": [ {name,dtype,full_shape,shard_numel,offset,src_rank,is_last}, ... ] }
        Returns True if succeeded, else False.
        """
        if int(self.fsdp_rank or 0) != 0:
            return False

        payload = {"shards": schedule}

        # primary endpoint
        resp = self.session.post(f"{self.base_url}/update_param_shards/", json=payload)

        if resp.status_code == 200:
            return True

        # optional alternative naming (if you prefer to implement this on server)
        if resp.status_code == 404:
            resp2 = self.session.post(f"{self.base_url}/update_param_shard_batch/", json=payload)
            if resp2.status_code == 200:
                return True

        logger.warning(
            f"Batch shard metadata API failed "
            f"(status={resp.status_code}). Falling back to legacy per-shard API."
        )
        return False

    def _get_or_make_buf(self, dtype: torch.dtype, numel: int, device: Any) -> torch.Tensor:
        key = (str(dtype), int(numel), str(device))
        buf = self._buffer_cache.get(key)
        if buf is None or buf.numel() != numel or buf.dtype != dtype or str(buf.device) != str(device):
            buf = torch.empty((numel,), dtype=dtype, device=device)
            self._buffer_cache[key] = buf
        return buf

    def update_model_params_streaming_fsdp(
        self,
        model: nn.Module,
        router_only: bool = False,
        expert_only: bool = False,
        trainable_only: bool = False,
    ):
        """
        B-plan safe FSDP streaming sync:
        - Build a shard schedule on FSDP rank0 using all_gather_object
        - Broadcast schedule to all FSDP ranks
        - All FSDP ranks call broadcast in identical order/count with identical src

        Perf:
        - NO per-shard barrier
        - NO CPU offload
        - Batch shard-metadata API (rank0)
        """
        if self.rank is None or self.vllm_world_size is None:
            raise RuntimeError("Call init_communicator first.")

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, ShardedStateDictConfig

        # 1) Build name filter (cleaned)
        name_filter: set[str] = set()
        for name, param in model.named_parameters():
            if trainable_only and not param.requires_grad:
                continue

            clean_name = self._clean_name(name)

            if router_only and not self._is_router_param_name(clean_name):
                continue
            if expert_only and not (self._is_expert_param_name(clean_name) or self._is_router_param_name(clean_name)):
                continue

            name_filter.add(clean_name)

        # 2) Get sharded state dict (NO CPU offload)
        sharded_cfg = ShardedStateDictConfig(offload_to_cpu=False)
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, state_dict_config=sharded_cfg):
            sd = model.state_dict()

        # 3) Build local shard map + infos
        local_map: dict[tuple[str, int], torch.Tensor] = {}
        local_infos: list[dict[str, Any]] = []

        for name, obj in sd.items():
            if name_filter and name not in name_filter:
                continue

            local_shards = getattr(obj, "local_shards", None)
            if callable(local_shards):
                local_shards = local_shards()

            full_shape = self._get_full_shape_from_obj(obj)

            if local_shards:
                for shard in local_shards:
                    st = shard.tensor
                    meta = shard.metadata
                    offsets = []
                    if hasattr(meta, "shard_offsets"):
                        offsets = list(meta.shard_offsets)
                    elif hasattr(meta, "offsets"):
                        offsets = list(meta.offsets)

                    offset_1d = self._calc_offset_1d(offsets, full_shape)
                    key = (name, int(offset_1d))
                    local_map[key] = st

                    local_infos.append({
                        "name": name,
                        "dtype": str(st.dtype),
                        "full_shape": [int(x) for x in full_shape],
                        "shard_numel": int(st.numel()),
                        "offset": int(offset_1d),
                        "owner_fsdp_rank": int(self.fsdp_rank or 0),
                    })

            else:
                # local_shards is None => treat as single unsharded tensor
                # owner fixed to fsdp rank0 to avoid duplicate schedule entries
                if local_shards is None and int(self.fsdp_rank or 0) == 0:
                    try:
                        t = obj
                        key = (name, 0)
                        local_map[key] = t
                        full_shape_t = self._get_full_shape_from_obj(t)

                        local_infos.append({
                            "name": name,
                            "dtype": str(t.dtype),
                            "full_shape": [int(x) for x in full_shape_t],
                            "shard_numel": int(t.numel()),
                            "offset": 0,
                            "owner_fsdp_rank": 0,
                        })
                    except Exception:
                        pass

        # 4) Gather infos to rank0
        if dist.is_available() and dist.is_initialized():
            gathered: list[Any] = [None for _ in range(int(self.fsdp_world_size or 1))]
            dist.all_gather_object(gathered, local_infos)
        else:
            gathered = [local_infos]

        # 5) Build schedule on rank0
        schedule: list[dict[str, Any]] = []
        if int(self.fsdp_rank or 0) == 0:
            all_infos: list[dict[str, Any]] = []
            for li in gathered:
                if li:
                    all_infos.extend(li)

            by_name: dict[str, list[dict[str, Any]]] = {}
            for it in all_infos:
                by_name.setdefault(it["name"], []).append(it)

            for pname, shards in by_name.items():
                shards = sorted(shards, key=lambda x: int(x.get("offset", 0)))
                for i, sh in enumerate(shards):
                    sh2 = dict(sh)
                    sh2["is_last"] = (i == len(shards) - 1)
                    # precompute global src rank for batch payload
                    sh2["src_rank"] = int(self.vllm_world_size) + int(sh2["owner_fsdp_rank"])
                    schedule.append(sh2)

            if not schedule:
                print(
                    f"[Client Error] No shards to sync! "
                    f"Filter count: {len(name_filter)} SD count: {len(sd)}",
                    flush=True,
                )
                if len(name_filter) > 0:
                    print(f"Sample Filter Name (Cleaned): {list(name_filter)[0]}", flush=True)
                if len(sd) > 0:
                    print(f"Sample SD Key: {list(sd.keys())[0]}", flush=True)

        # 6) Broadcast schedule to all ranks
        if dist.is_available() and dist.is_initialized():
            obj_list = [schedule]
            dist.broadcast_object_list(obj_list, src=0)
            schedule = obj_list[0]

        if not schedule:
            return

        # 6.5) Try batch metadata post on rank0, broadcast flag
        use_batch = False
        if int(self.fsdp_rank or 0) == 0:
            use_batch = self._try_post_shard_metadata_batch(schedule)

        if dist.is_available() and dist.is_initialized():
            flag_list = [use_batch]
            dist.broadcast_object_list(flag_list, src=0)
            use_batch = bool(flag_list[0])

        # 7) Execute schedule: all ranks must call identical broadcast with identical src
        last_name = None
        total = len(schedule)

        # choose buffer device
        buf_device: Any = "cpu"
        if (not is_torch_xpu_available()) and (self.broadcast_device is not None):
            buf_device = self.broadcast_device

        for i, sh in enumerate(schedule):
            name = sh["name"]
            dtype_str = sh["dtype"]
            full_shape = sh.get("full_shape", [])
            shard_numel = int(sh["shard_numel"])
            offset = int(sh["offset"])
            owner = int(sh.get("owner_fsdp_rank", 0))
            is_last = bool(sh.get("is_last", False))

            # debug trace per param name change (rank0 only)
            if int(self.fsdp_rank or 0) == 0 and name != last_name:
                print(f"[Client Trace] Syncing param: {name}", flush=True)
                last_name = name

            global_src = int(sh.get("src_rank", int(self.vllm_world_size) + owner))

            # Legacy per-shard meta only if batch is not available
            if (not use_batch) and int(self.fsdp_rank or 0) == 0:
                payload = {
                    "name": name,
                    "dtype": dtype_str,
                    "full_shape": [int(x) for x in full_shape],
                    "shard_numel": shard_numel,
                    "offset": offset,
                    "src_rank": global_src,
                    "is_last": is_last,
                }
                resp = self.session.post(f"{self.base_url}/update_param_shard/", json=payload)
                if resp.status_code != 200:
                    raise Exception(f"Failed /update_param_shard/: {resp.status_code}, {resp.text}")

            # Allocate identical buffer on all ranks (cached)
            torch_dtype = self._torch_dtype_from_str(dtype_str)
            buf = self._get_or_make_buf(torch_dtype, shard_numel, buf_device)

            # Owner copies real data, others keep dummy
            if int(self.fsdp_rank or 0) == owner:
                t = local_map.get((name, offset))
                if t is not None:
                    flat = self._move_to_broadcast_device(t.detach().reshape(-1))
                    if flat.numel() != shard_numel:
                        # Silent truncation here would corrupt the weights pushed
                        # to the vLLM server; fail loudly instead.
                        raise RuntimeError(
                            f"[VLLMClient] shard numel mismatch for param '{name}' "
                            f"(offset={offset}): local shard has {int(flat.numel())} "
                            f"elements but schedule expects {shard_numel}."
                        )
                    buf.copy_(flat)

            # The critical part: all ranks use the SAME src
            self._broadcast_flat(buf, src_rank=global_src)

            # occasional lightweight progress (rank0 only)
            if int(self.fsdp_rank or 0) == 0 and (i + 1) % 1000 == 0:
                print(f"[Client Trace] Shard progress: {i+1}/{total}", flush=True)

    def reset_prefix_cache(self):
        resp = self.session.post(f"{self.base_url}/reset_prefix_cache/")
        if resp.status_code != 200:
            raise Exception(f"Request failed: {resp.status_code}")

    def close_communicator(self):
        try:
            self.session.post(f"{self.base_url}/close_communicator/")
        except Exception:
            pass
