#!/bin/bash
#SBATCH --job-name=qwen3_235b_expert_dpo_attn70
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodelist=node-XX,node-XX,node-XX,node-XX,node-XX,node-XX
#SBATCH --nodes=6
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=240
#SBATCH --output=outputs/logs/qwen3_235b_expert_dpo_attn70-%j.out
#SBATCH --error=outputs/logs/qwen3_235b_expert_dpo_attn70-%j.err

echo "=== Qwen3-235B Expert Full-Param DPO (attn_layers=70-93) ジョブ開始 ==="
echo "ジョブ名      : qwen3_235b_expert_dpo_attn70"
echo "SLURM_JOB_ID  : ${SLURM_JOB_ID}"
echo "開始時刻      : $(date)"

echo "=== 環境セットアップ ==="

module load nccl/2.22.3
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="enp25s0np0"
export NCCL_BUFFSIZE=2097152

export OMP_NUM_THREADS=2

ulimit -m unlimited
ulimit -v unlimited
ulimit -d unlimited
ulimit -s unlimited
ulimit -n 262144

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export TORCHDYNAMO_DISABLE=1

mkdir -p outputs/logs

eval "$(conda shell.bash hook)"
conda activate qwen_expert_rl

echo "Python path      : $(which python)"
echo "Python version   : $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "#GPUs (local)    : $(nvidia-smi -L | wc -l)"

# ============================================================
# モデル / データ設定
# ============================================================
export BASE_MODEL="/path/to/base_model"
echo "BASE_MODEL       : $BASE_MODEL"

# DPO データセット (prompt / chosen / rejected の JSONL)
export DPO_DATASET="/path/to/train_dpo.jsonl"
echo "DPO_DATASET      : $DPO_DATASET"

# ============================================================
# DPO ハイパーパラメータ
# ============================================================
BETA="${BETA:-0.5}"
LOSS_TYPE="${LOSS_TYPE:-sigmoid}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.0}"
SEQ_LEN="${SEQ_LEN:-4096}"
LR="${LR:-5e-7}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"

echo "BETA              : $BETA"
echo "LOSS_TYPE         : $LOSS_TYPE"
echo "SEQ_LEN           : $SEQ_LEN"
echo "LR                : $LR"
echo "EPOCHS            : $EPOCHS"
echo "BATCH_SIZE        : $BATCH_SIZE"
echo "GRAD_ACCUM        : $GRAD_ACCUM"
TOTAL_GPUS=$((SLURM_NNODES * 8))
echo "Effective batch   : $((BATCH_SIZE * GRAD_ACCUM * TOTAL_GPUS)) pairs"

# ============================================================
# attention 設定 (SFT と同じ設定)
# ============================================================
export ATTN_LAYERS="70-93"
export ATTN_PROJ="q_proj,k_proj,v_proj,o_proj"
echo "ATTN_LAYERS       : $ATTN_LAYERS"
echo "ATTN_PROJ         : $ATTN_PROJ"

# ============================================================
# 通信設定
# ============================================================
HOST_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export HOST_ADDR
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-$((29500 + (${SLURM_JOB_ID:-0} % 500)))}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200

echo "SLURM_NNODES      : $SLURM_NNODES"
echo "HOST_ADDR (rank0) : $HOST_ADDR"
echo "MAIN_PROCESS_PORT : $MAIN_PROCESS_PORT"

# ============================================================
# 保存先
# ============================================================
SAVE_DIR_BASE="outputs/ckpts_qwen3_235b_expert_dpo_attn70"
SAVE_DIR="${SAVE_DIR_BASE}/${SLURM_JOB_ID}"
mkdir -p "$SAVE_DIR"
echo "SAVE_DIR          : $SAVE_DIR"

# W&B
WANDB_PROJECT="expert_fullparam_dpo"
WANDB_RUN_NAME="235b-attn70-beta${BETA}-${SLURM_JOB_ID}"

echo "=== Expert Full-Param DPO (attn) 学習開始 (Accelerate + FSDP) ==="

# ============================================================
# メモリ見積もり (6 nodes x 8 GPU = 48 GPU, FSDP FULL_SHARD):
#   attn 70-93 のみ学習 (~8.5B params):
#     Frozen params (sharded): ~235B/48 ≈ 5GB
#     Trainable params: ~8.5B/48 ≈ 0.4B → weights+grads+Adam ≈ 1.8GB/GPU
#     Activations (DPO concat fwd, grad ckpt): ~4-6 GB
#     Total: ~10-15 GB/GPU (H100 80GB → 余裕)
# ============================================================

srun -l --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c '
set -eux

echo "==== Node $(hostname) / SLURM_NODEID=${SLURM_NODEID} ===="

GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-8}
WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))

echo "GPUS_PER_NODE      = $GPUS_PER_NODE"
echo "SLURM_NNODES       = $SLURM_NNODES"
echo "WORLD_SIZE         = $WORLD_SIZE"
echo "This machine_rank  = $SLURM_NODEID"

NCCL_DEBUG=WARN accelerate launch \
  --num_machines "$SLURM_NNODES" \
  --machine_rank "$SLURM_NODEID" \
  --main_process_ip "$HOST_ADDR" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  --num_processes "$WORLD_SIZE" \
  --mixed_precision bf16 \
  run_expert_fullparam_dpo.py \
    --model "$BASE_MODEL" \
    --dataset "$DPO_DATASET" \
    --attn_layers "$ATTN_LAYERS" \
    --attn_proj "$ATTN_PROJ" \
    --seq_len '"${SEQ_LEN}"' \
    --batch_size '"${BATCH_SIZE}"' \
    --gradient_accumulation_steps '"${GRAD_ACCUM}"' \
    --epochs '"${EPOCHS}"' \
    --steps 0 \
    --lr '"${LR}"' \
    --grad_clip 1.0 \
    --beta '"${BETA}"' \
    --loss_type '"${LOSS_TYPE}"' \
    --label_smoothing '"${LABEL_SMOOTHING}"' \
    --log_every 1 \
    --save_every 50 \
    --wandb_project '"${WANDB_PROJECT}"' \
    --wandb_run_name '"${WANDB_RUN_NAME}"' \
    --save_dir "'"$SAVE_DIR"'"
'

EXIT_CODE=$?

echo "=== Expert Full-Param DPO (attn) 学習完了 ==="
echo "終了コード: $EXIT_CODE"
echo "終了時刻: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Expert+Attn full-param DPO 学習が正常に完了しました！"
else
    echo "✗ Expert+Attn full-param DPO 学習でエラーが発生しました。終了コード: $EXIT_CODE"
fi

exit $EXIT_CODE
