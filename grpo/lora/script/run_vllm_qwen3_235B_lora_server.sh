#!/bin/bash
#SBATCH --job-name=vllm_qwen3_235B_lora
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --nodelist=node-XX,node-XX,node-XX,node-XX
#SBATCH --chdir=${HOME}
#SBATCH --output=${HOME}/%x_%j.out
#SBATCH --error=${HOME}/%x_%j.err

set -eEo pipefail
echo "=== [START] $(date) host=$(hostname) pwd=$(pwd) jobid=${SLURM_JOB_ID:-} nnodes=${SLURM_NNODES:-} nodelist=${SLURM_NODELIST:-} ==="

if [[ "${1:-}" != "__worker" ]]; then
  ENDPOINTS_DIR="${ENDPOINTS_DIR:-${HOME}/vllm_endpoints_${SLURM_JOB_ID}}"
  mkdir -p "${ENDPOINTS_DIR}"
  rm -f "${ENDPOINTS_DIR}"/*.txt || true

  # ---- ポート戦略 ----
  PORT_BASE="${PORT_BASE:-8000}"
  PORT_PER_NODE="${PORT_PER_NODE:-0}"
  PORT_DEFAULT="${PORT:-8000}"

  # ---- READY/HEARTBEAT設定 ----
  READY_POLL_SEC="${READY_POLL_SEC:-2}"
  READY_MAX_TRIES="${READY_MAX_TRIES:-1800}"   # デフォルト60分
  HEARTBEAT_SEC="${HEARTBEAT_SEC:-600}"        # 10分
  HB_TIMEOUT_SEC="${HB_TIMEOUT_SEC:-2}"

  # ---- “偽OK”防止: /v1/models にこの model id が含まれるかで判定 ----
  EXPECTED_MODEL_ID="${SERVED_MODEL_NAME:-qwen3-235b-base}"

  # ---- LoRA 監視（ログのみ）----
  # vLLMログの "Loaded new LoRA adapter: name '...'" を集計
  LORA_NAME="${LORA_NAME:-train-lora}"

  echo "=== [COORD] ENDPOINTS_DIR=${ENDPOINTS_DIR} ==="
  echo "=== [COORD] PORT_PER_NODE=${PORT_PER_NODE} PORT_BASE=${PORT_BASE} PORT_DEFAULT=${PORT_DEFAULT} ==="
  echo "=== [COORD] READY: poll=${READY_POLL_SEC}s tries=${READY_MAX_TRIES} | HEARTBEAT: every=${HEARTBEAT_SEC}s timeout=${HB_TIMEOUT_SEC}s ==="
  echo "=== [COORD] EXPECTED_MODEL_ID=${EXPECTED_MODEL_ID} ==="
  echo "=== [COORD] LORA_NAME=${LORA_NAME} (LoRA sync is checked from logs) ==="

  SCRIPT_PATH="${HOME}/run_vllm_qwen3_235B_lora_server.sh"
  [[ -f "${SCRIPT_PATH}" ]] || { echo "FATAL: not found ${SCRIPT_PATH}"; exit 1; }

  mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_NODELIST}")
  echo "=== [COORD] Hosts: ${HOSTS[*]} ==="

  echo "=== [COORD] Launching workers via srun (background) ==="
  srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --exclusive \
       --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
       --gres=gpu:8 \
       --kill-on-bad-exit=1 \
       --output="${HOME}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_%N.out" \
       --error="${HOME}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_%N.err" \
       bash "${SCRIPT_PATH}" __worker &
  SRUN_PID=$!

  cleanup_coord() {
    echo "=== [COORD CLEANUP] $(date) ==="
    if kill -0 "${SRUN_PID}" 2>/dev/null; then
      echo "Killing background srun pid=${SRUN_PID}"
      kill -TERM "${SRUN_PID}" 2>/dev/null || true
      sleep 2 || true
      kill -KILL "${SRUN_PID}" 2>/dev/null || true
    fi
    echo "=== [COORD CLEANUP DONE] ==="
  }
  trap cleanup_coord EXIT

  # ---- READY待ち（proxy無効 + JSONで model id を確認）----
  ready_check() {
    local url="$1"
    python - <<PY
import time, sys, json
from urllib.request import build_opener, ProxyHandler

url = "${url}"
expected = "${EXPECTED_MODEL_ID}"
poll = float("${READY_POLL_SEC}")
tries = int("${READY_MAX_TRIES}")

opener = build_opener(ProxyHandler({}))  # proxy無効
last = None

for _ in range(tries):
    try:
        with opener.open(url, timeout=2) as r:
            data = json.load(r)
        ids = [m.get("id") for m in data.get("data", [])]
        if expected in ids:
            sys.exit(0)
        last = ("unexpected_models", ids[:5])
    except Exception as e:
        last = repr(e)
    time.sleep(poll)

print("NG", last)
sys.exit(1)
PY
  }

  set +e
  for i in "${!HOSTS[@]}"; do
    host="${HOSTS[$i]}"
    if [[ "${PORT_PER_NODE}" == "1" ]]; then
      port="$((PORT_BASE + i))"
    else
      port="${PORT_DEFAULT}"
    fi
    url="http://${host}:${port}/v1/models"
    echo "=== [COORD] Waiting READY: ${host}:${port} (${url}) ==="
    ready_check "${url}"
    rc=$?
    if [[ $rc -eq 0 ]]; then
      echo "=== [READY] ${host}:${port} ==="
    else
      echo "=== [WARN] READY timeout: ${host}:${port} (will keep heartbeat watching) ==="
    fi
  done
  set -e

  echo "=== [COORD] Endpoints (created after local READY on worker) ==="
  ls -1 "${ENDPOINTS_DIR}"/*.txt 2>/dev/null | sort | xargs -r cat || true

  # ---- HEARTBEAT（10分おき）: /v1/models で生存確認 ----
  check_one() {
    local host="$1" port="$2"
    python - <<PY
import sys, json
from urllib.request import build_opener, ProxyHandler

url="http://${host}:${port}/v1/models"
expected="${EXPECTED_MODEL_ID}"
opener = build_opener(ProxyHandler({}))  # proxy無効
try:
    with opener.open(url, timeout=${HB_TIMEOUT_SEC}) as r:
        data = json.load(r)
    ids = [m.get("id") for m in data.get("data", [])]
    sys.exit(0 if expected in ids else 1)
except Exception:
    sys.exit(1)
PY
  }

  # ---- LoRA（ログのみ）: 各ノードの “最新 step” を拾って揃ってるか判定 ----
  lora_latest_step_for_host() {
    local host="$1"
    local lf="${HOME}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${host}.out"
    [[ -f "${lf}" ]] || { echo "NOLOG"; return 0; }

    local line
    line="$(grep -E "Loaded new LoRA adapter: name '${LORA_NAME}'" "${lf}" | tail -n 1 || true)"
    [[ -n "${line}" ]] || { echo "NONE"; return 0; }

    # step_00000051 みたいなのを抜く
    echo "${line}" | sed -n 's/.*\(step_[0-9]\+\).*/\1/p'
  }

  echo "=== [COORD] Heartbeat loop started (every ${HEARTBEAT_SEC}s) ==="
  while kill -0 "${SRUN_PID}" 2>/dev/null; do
    ts="$(date '+%F %T')"

    # --- Server health ---
    line="=== [HEARTBEAT] ${ts} ==="
    ok_all=1
    for i in "${!HOSTS[@]}"; do
      host="${HOSTS[$i]}"
      if [[ "${PORT_PER_NODE}" == "1" ]]; then
        port="$((PORT_BASE + i))"
      else
        port="${PORT_DEFAULT}"
      fi

      if check_one "${host}" "${port}"; then
        line+=" ${host}:${port}=OK"
      else
        line+=" ${host}:${port}=NG"
        ok_all=0
      fi
    done
    echo "${line}"
    if [[ "${ok_all}" -ne 1 ]]; then
      echo "=== [WARN] Some endpoints unhealthy. See per-node logs: ${HOME}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_<node>.err ==="
    fi

    # --- LoRA sync from logs only ---
    echo -n "=== [LORA-HEARTBEAT] ${ts} ==="
    steps=()
    ok_lora=1
    for host in "${HOSTS[@]}"; do
      step="$(lora_latest_step_for_host "${host}")"
      [[ -n "${step}" ]] || step="NONE"
      steps+=("${step}")
      echo -n " ${host}=${step}"
      [[ "${step}" == "NONE" || "${step}" == "NOLOG" ]] && ok_lora=0
    done

    # 全一致判定（NONE/NOLOG が混ざったら WARN）
    first=""
    for s in "${steps[@]}"; do
      [[ "${s}" == "NONE" || "${s}" == "NOLOG" || -z "${s}" ]] && { first=""; break; }
      if [[ -z "${first}" ]]; then first="${s}"; fi
      [[ "${s}" != "${first}" ]] && { ok_lora=0; break; }
    done

    if [[ "${ok_lora}" -eq 1 ]]; then
      echo " (OK)"
    else
      echo " (WARN)"
    fi

    sleep "${HEARTBEAT_SEC}"
  done

  echo "=== [COORD] srun finished (workers exited). $(date) ==="
  wait "${SRUN_PID}"
  exit 0
fi

# =========================
# Worker
# =========================
NODEID="${SLURM_NODEID:-0}"
NODE_HOST="$(hostname -s)"
unset HOST 2>/dev/null || true  # conda が HOST を上書きするので無効化

module reset || true
module load cuda/12.1 || true
module load cudnn/8.9.7 || true
module load nccl/2.21.5 || true
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt || true
module list || true

export CUDNN_HOME=/home/appli/cudnn/8.9.7
export NCCL_HOME=/home/appli/nccl/2.21.5
export NCCL_SOCKET_IFNAME=enp25s0np0
export LD_LIBRARY_PATH=${CUDA_PATH:-/usr/local/cuda}/lib64:$CUDNN_HOME/lib:$NCCL_HOME/lib:${LD_LIBRARY_PATH:-}
export NVTE_FUSED_ATTN=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1

PORT_BASE="${PORT_BASE:-8000}"
PORT_PER_NODE="${PORT_PER_NODE:-0}"
if [[ "${PORT_PER_NODE}" == "1" ]]; then
  PORT="$((PORT_BASE + NODEID))"
else
  PORT="${PORT:-8000}"
fi

eval "$(${HOME}/miniconda3-py311/bin/conda shell.bash hook)"
conda activate qwen_lora_rl

echo "=== [WORKER] $(date) nodeid=${NODEID} host=${NODE_HOST} port=${PORT} cwd=$(pwd) ==="
python -V
nvidia-smi -L || true

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi

# ★ポート掃除（取り残し対策）
if command -v lsof >/dev/null 2>&1; then
  pids="$(lsof -ti TCP:${PORT} -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    echo "=== [WORKER] Port ${PORT} is busy. Killing: ${pids} ==="
    kill -TERM ${pids} 2>/dev/null || true
    sleep 2 || true
    kill -KILL ${pids} 2>/dev/null || true
  fi
fi
if command -v fuser >/dev/null 2>&1; then
  fuser -k -n tcp "${PORT}" 2>/dev/null || true
fi

MODEL_PATH="${MODEL_PATH:-/path/to/base_model}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-235b-base}"
TP_SIZE="${TP_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-6000}"
MAX_LORAS="${MAX_LORAS:-1}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"

echo "MODEL_PATH=${MODEL_PATH}"
echo "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "PORT=${PORT} TP_SIZE=${TP_SIZE} MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "MAX_LORAS=${MAX_LORAS} MAX_LORA_RANK=${MAX_LORA_RANK}"

VLLM_PID=""
cleanup() {
  echo "=== [CLEANUP] $(date) nodeid=${NODEID} host=${NODE_HOST} ==="
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill -TERM "${VLLM_PID}" 2>/dev/null || true
    sleep 2 || true
    kill -KILL "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --trust-remote-code \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --enable-lora \
  --max-loras "${MAX_LORAS}" \
  --max-lora-rank "${MAX_LORA_RANK}" \
  --gpu-memory-utilization 0.88 \
  --disable-custom-all-reduce \
  --max-num-seqs 64 \
  &
VLLM_PID=$!
echo "=== [WORKER] vLLM started pid=${VLLM_PID} host=${NODE_HOST} port=${PORT} ==="

# ローカルREADY後に endpoints を書く（proxy無効＆モデルID確認）
READY_POLL_SEC="${READY_POLL_SEC:-2}"
READY_MAX_TRIES="${READY_MAX_TRIES:-1800}"
python - <<PY
import time, sys, json
from urllib.request import build_opener, ProxyHandler

url = "http://127.0.0.1:${PORT}/v1/models"
expected = "${SERVED_MODEL_NAME}"
poll = float("${READY_POLL_SEC}")
tries = int("${READY_MAX_TRIES}")

opener = build_opener(ProxyHandler({}))
last = None

for _ in range(tries):
    try:
        with opener.open(url, timeout=2) as r:
            data = json.load(r)
        ids = [m.get("id") for m in data.get("data", [])]
        if expected in ids:
            sys.exit(0)
        last = ("unexpected_models", ids[:5])
    except Exception as e:
        last = repr(e)
    time.sleep(poll)

print("READY_CHECK_FAILED", last)
sys.exit(1)
PY

ENDPOINTS_DIR="${ENDPOINTS_DIR:-${HOME}/vllm_endpoints_${SLURM_JOB_ID}}"
mkdir -p "${ENDPOINTS_DIR}"
echo "${NODE_HOST}:${PORT}" > "${ENDPOINTS_DIR}/${NODEID}_${NODE_HOST}.txt"
echo "=== [WORKER READY] ${NODE_HOST}:${PORT} ==="

wait "${VLLM_PID}"
echo "=== [END] vLLM server finished $(date) nodeid=${NODEID} host=${NODE_HOST} ==="
