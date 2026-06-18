#!/bin/bash
#SBATCH --job-name=vllm_qwen3_235B
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodelist=node-XX,node-XX
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err
# 必要なら（環境によっては不要/禁止のこともある）
#SBATCH --mem=0

set -euo pipefail

echo "=== vLLM Qwen3-235B サーバ起動ジョブ開始 ==="
echo "開始時刻: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "NODELIST   : ${SLURM_NODELIST}"
echo "NNODES     : ${SLURM_NNODES}"
echo

mkdir -p outputs/logs outputs/vllm

module reset
module load cuda/12.1
module load cudnn/8.9.7
module load nccl/2.21.5
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt

export CUDNN_PATH=/home/appli/cudnn/8.9.7
export CUDNN_HOME=/home/appli/cudnn/8.9.7
export NCCL_HOME=/home/appli/nccl/2.21.5
export NCCL_PATH=/home/appli/nccl/2.21.5
export NCCL_ROOT=/home/appli/nccl/2.21.5
export NCCL_SOCKET_IFNAME=enp25s0np0
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$CUDNN_HOME/lib:$NCCL_HOME/lib:$LD_LIBRARY_PATH
export CUDNN_INCLUDE_DIR=$CUDNN_HOME/include
export NVTE_FUSED_ATTN=0

export PORT=8000
export IFNAME=enp25s0np0

# READY 集約（共有FS前提）
export READY_FILE="outputs/vllm/ready_${SLURM_JOB_ID}.tsv"
export READY_LOCK="outputs/vllm/ready_${SLURM_JOB_ID}.lock"
: > "${READY_FILE}" || true

echo "PORT        : ${PORT}"
echo "IFNAME      : ${IFNAME}"
echo "READY_FILE  : ${READY_FILE}"
echo
echo "=== vLLM サーバ起動開始（各ノード1台） ==="

# LoRA版と同じ：各ノード1回。ログは sbatch本体ログに集約（tailが楽）
srun -l --kill-on-bad-exit=1 \
  --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  bash -c '
set -euo pipefail

PORT="${PORT:-8000}"
IFNAME="${IFNAME:-enp25s0np0}"
READY_FILE="${READY_FILE:?}"
READY_LOCK="${READY_LOCK:?}"

cleanup() {
  echo "[vllm ${HOSTNAME}] Cleaning up..."
  pkill -f "trl vllm-serve" || true
  if command -v lsof >/dev/null 2>&1; then
    lsof -i :"${PORT}" | awk "NR>1 {print \$2}" | xargs -r kill -9 || true
  else
    fuser -k "${PORT}"/tcp || true
  fi
  echo "[vllm ${HOSTNAME}] Cleanup done."
}
trap cleanup EXIT

# /etc/bashrc の nounset 地雷回避
set +u
export BASHRCSOURCED=1
source ~/.bashrc
conda activate qwen_expert_rl
set -u

MODEL_NAME="/path/to/base_model"
TP_SIZE=8
MAX_MODEL_LEN=8192

IP_ADDR=$(ip -4 addr show "${IFNAME}" | grep -oP "(?<=inet\\s)\\d+(\\.\\d+){3}" | head -n 1 || true)
URL="http://${IP_ADDR}:${PORT}"

echo "[vllm ${HOSTNAME}] IF=${IFNAME} IP=${IP_ADDR} PORT=${PORT}"
echo "[vllm ${HOSTNAME}] MODEL_NAME=${MODEL_NAME}"
echo "[vllm ${HOSTNAME}] TP_SIZE=${TP_SIZE} MAX_MODEL_LEN=${MAX_MODEL_LEN}"

HEALTH_URL_1="http://127.0.0.1:${PORT}/health"
HEALTH_URL_2="http://127.0.0.1:${PORT}/v1/models"

# 235Bロードは長いので長め
max_tries=600   # 50分
sleep_sec=5

# task0 が全ノード READY を待って表示（LoRA版の「起動確認」寄せ）
if [ "${SLURM_PROCID:-999999}" -eq 0 ]; then
  (
    echo "[vllm ${HOSTNAME}] (task0) Waiting ALL READY... expect=${SLURM_NNODES}"
    for j in $(seq 1 1200); do
      # cut/grep/sort で集計（$1 を使わない＝nounset安全）
      cnt=$(cut -f1 "${READY_FILE}" 2>/dev/null | grep -v "^$" | sort -u | wc -l | tr -d " ")
      if [ "${cnt}" -ge "${SLURM_NNODES}" ]; then
        echo "[vllm ${HOSTNAME}] (task0) ALL READY ✅ (${cnt}/${SLURM_NNODES})"
        echo "----- READY endpoints -----"
        cut -f1,4 "${READY_FILE}" | sort -u | tr "\t" " "
        urls=$(cut -f4 "${READY_FILE}" | sort -u | paste -sd, -)
        echo "export VLLM_BASE_URLS=\"${urls}\""
        echo "---------------------------"
        exit 0
      fi
      if (( j % 12 == 0 )); then
        echo "[vllm ${HOSTNAME}] (task0) READY so far: ${cnt}/${SLURM_NNODES}"
      fi
      sleep 5
    done
    echo "[vllm ${HOSTNAME}] (task0) WARNING: ALL READY wait timeout (still running servers)"
  ) &
fi

# READY検知して共有ファイルに書く（裏で動かす）
(
  echo "[vllm ${HOSTNAME}] Waiting for READY check... (${URL})"
  for i in $(seq 1 ${max_tries}); do
    if curl -fsSL --max-time 2 "${HEALTH_URL_1}" >/dev/null 2>&1 || \
       curl -fsSL --max-time 2 "${HEALTH_URL_2}" >/dev/null 2>&1; then
      echo "[vllm ${HOSTNAME}] READY ✅ ${URL}"
      {
        flock -x 9
        # 既に書いてたら追加しない（$1を使わず grep で判定）
        if ! grep -q "^${HOSTNAME}\t" "${READY_FILE}" 2>/dev/null; then
          printf "%s\t%s\t%s\t%s\n" "${HOSTNAME}" "${IP_ADDR}" "${PORT}" "${URL}" >> "${READY_FILE}"
        fi
      } 9>"${READY_LOCK}" || true
      exit 0
    fi
    if (( i % 12 == 0 )); then
      echo "[vllm ${HOSTNAME}] Still loading... (try ${i}/${max_tries})"
    fi
    sleep ${sleep_sec}
  done
  echo "[vllm ${HOSTNAME}] WARNING: READY check timeout (server may still come up later)"
) &

# ---- vLLM 起動（設定は変更しない・単発版と同じ）----
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
trl vllm-serve \
  --model "$MODEL_NAME" \
  --tensor-parallel-size $TP_SIZE \
  --max_model_len $MAX_MODEL_LEN \
  --host 0.0.0.0 \
  --port $PORT

echo "[vllm ${HOSTNAME}] vLLM server finished."
'
