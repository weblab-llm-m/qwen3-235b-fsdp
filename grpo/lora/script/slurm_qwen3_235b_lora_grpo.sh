#!/bin/bash
#SBATCH --job-name=qwen3_235b_lora_grpo
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodelist=node-XX,node-XX,node-XX,node-XX,node-XX,node-XX,node-XX,node-XX
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=240
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -euo pipefail

echo "=== Qwen3-235B LoRA GRPO ジョブ開始 ==="
echo "開始時刻: $(date)"

mkdir -p outputs/logs outputs/ckpts

# === ネットワーク設定の要 ===
# 通信に使いたい高速インターフェース名（環境に合わせて確認済みと想定）
export TARGET_IFNAME="enp25s0np0"

module load nccl/2.22.3
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="${TARGET_IFNAME}"
export NCCL_BUFFSIZE=2097152

# 初期化通信(GLOO/TCPStore)も同じIFに強制する
export GLOO_SOCKET_IFNAME="${TARGET_IFNAME}"
export TP_SOCKET_IFNAME="${TARGET_IFNAME}"

export OMP_NUM_THREADS=2
ulimit -n 262144 || true

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export TORCHDYNAMO_DISABLE=1

eval "$(conda shell.bash hook)"
set +u
conda activate qwen_lora_rl
set -u

# ====== 設定値 ======
export BASE_MODEL="/path/to/base_model"
export SPECIALIST_CHATML_PATH="/path/to/train.jsonl"
export VLLM_BASE_URLS="${VLLM_BASE_URLS:-http://host1:8000,http://host2:8000}"

# === 重要: マスターノードのIP取得ロジック ===
MASTER_HOSTNAME="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n 1)"

# マスターノード上で、指定IF(enp25s0np0)に割り当たっているIPを直接取得する
# これにより DNS解決(eth0等が返る)による不整合を防ぐ
HOST_ADDR=$(srun --nodes=1 --ntasks=1 -w "${MASTER_HOSTNAME}" ip -4 addr show "${TARGET_IFNAME}" | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n 1)

# 取得失敗時のフォールバック
if [ -z "${HOST_ADDR}" ]; then
    echo "WARNING: ${TARGET_IFNAME} のIP取得に失敗。ホスト名を使用します。"
    HOST_ADDR="${MASTER_HOSTNAME}"
fi

# === ポート自動設定（OSに空きポートを探させる） ===
export MAIN_PROCESS_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export HOST_ADDR

# タイムアウト延長
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=7200

echo "MASTER IP (Rank0)   : ${HOST_ADDR} (on ${TARGET_IFNAME})"
echo "MAIN_PROCESS_PORT : ${MAIN_PROCESS_PORT} (Automatically assigned)"

echo "=== LoRA GRPO 学習開始 ==="

# 各ノード 1 回ずつ accelerate を起動
srun -l --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash -c '
set -euo pipefail

GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"
if [ -z "${GPUS_PER_NODE}" ]; then GPUS_PER_NODE=8; fi
WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))
export WORLD_SIZE

# NCCL_DEBUG=INFO で詳細ログを出す
NCCL_DEBUG=WARN accelerate launch \
  --num_machines "${SLURM_NNODES}" \
  --machine_rank "${SLURM_NODEID}" \
  --main_process_ip "${HOST_ADDR}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --num_processes "${WORLD_SIZE}" \
  --mixed_precision bf16 \
  run_lora_grpo.py \
    --model "${BASE_MODEL}" \
    --dataset "${SPECIALIST_CHATML_PATH}" \
    --epochs 1 \
    --batch_size 2 \
    --grpo_group_size 4 \
    --gen_max_new_tokens 3000 \
    --gen_temperature 0.7 \
    --gen_top_p 0.9 \
    --gen_repetition_penalty 1.05 \
    --reward_scale 1.0 \
    --reward_clip 1.0 \
    --judge_backend openrouter \
    --judge_model_name "qwen/qwen3-235b-a22b-2507" \
    --judge_max_new_tokens 4096 \
    --judge_temperature 0.0 \
    --judge_max_retries 10000 \
    --use_vllm \
    --vllm_base_urls "${VLLM_BASE_URLS}" \
    --vllm_model_name qwen3-235b-base \
    --vllm_lora_name train-lora \
    --vllm_connection_timeout 240 \
    --vllm_force_n1 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --log_every 1 \
    --save_dir "outputs/ckpts_qwen3_235b_lora_grpo" \
    --save_updated_only \
    --wandb \
    --debug_print
'
exit $?
