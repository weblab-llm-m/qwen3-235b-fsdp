#!/bin/bash
#SBATCH --job-name=qwen3_235b_expert_grpo
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodelist=node-XX,node-XX,node-XX,node-XX
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=240
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -euo pipefail

echo "=== Qwen3-235B Expert Full-Param GRPO ジョブ開始 ==="
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
conda activate qwen_expert_rl
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

echo "=== Expert Full-Param GRPO 学習開始 ==="

# 各ノード 1 回ずつ accelerate を起動
srun -l --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash -c '
set -euo pipefail

GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"
if [ -z "${GPUS_PER_NODE}" ]; then GPUS_PER_NODE=8; fi
WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))
export WORLD_SIZE

NCCL_DEBUG=WARN accelerate launch \
  --num_machines "${SLURM_NNODES}" \
  --machine_rank "${SLURM_NODEID}" \
  --main_process_ip "${HOST_ADDR}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --num_processes "${WORLD_SIZE}" \
  --mixed_precision bf16 \
  run_expert_fullparam_grpo.py \
    --model "${BASE_MODEL}" \
    --dataset "${SPECIALIST_CHATML_PATH}" \
    --layers 94 \
    --experts 128 \
    --mask_json "/path/to/selected_experts.json" \
    --epochs 5 \
    --batch_size 1 \
    --grpo_group_size 8 \
    --seq_len 2048 \
    --lr 5e-5 \
    --weight_decay 0.0 \
    --grad_clip 1.0 \
    --gen_max_new_tokens 2048 \
    --gen_temperature 0.7 \
    --gen_top_p 0.9 \
    --gen_repetition_penalty 1.05 \
    --reward_mode gspo \
    --reward_scale 1.0 \
    --reward_clip 1.0 \
    --judge_backend cerebras+openrouter \
    --judge_model_name qwen-3-235b-a22b-instruct-2507 \
    --judge_fallback_model_name qwen/qwen3-235b-a22b \
    --smart_fallback_model_name qwen/qwen3-235b-a22b \
    --judge_max_new_tokens 4096 \
    --judge_temperature 0.0 \
    --judge_max_retries 10000 \
    --use_vllm \
    --vllm_base_urls "${VLLM_BASE_URLS}" \
    --vllm_connection_timeout 240 \
    --centralize_actor_judge \
    --log_every 1 \
    --save_dir "outputs/ckpts_qwen3_235b_expert_gspo" \
    --save_updated_only \
    --wandb \
    --wandb_project expert_fullparam_grpo \
    --debug_print \
    --eval_dataset /path/to/eval.jsonl
'
exit $?
