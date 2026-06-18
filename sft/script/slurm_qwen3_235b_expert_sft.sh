#!/bin/bash
#SBATCH --job-name=qwen3_235b_expert_sft_attn70
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodelist=node-XX,node-XX,node-XX,node-XX,node-XX,node-XX
#SBATCH --nodes=6
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=6              # 6ノードなので 6タスク
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=240
#SBATCH --output=outputs/logs/qwen3_235b_expert_sft_attn70-%j.out
#SBATCH --error=outputs/logs/qwen3_235b_expert_sft_attn70-%j.err

echo "=== Qwen3-235B Expert Full-Param SFT (attn_layers=70-93) ジョブ開始 ==="
echo "ジョブ名      : qwen3_235b_expert_sft_attn70"
echo "SLURM_JOB_ID  : ${SLURM_JOB_ID}"
echo "開始時刻      : $(date)"

echo "=== 環境セットアップ ==="

module load nccl/2.22.3
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="enp25s0np0"
export NCCL_BUFFSIZE=2097152

# CPU スレッド
export OMP_NUM_THREADS=2

# ulimit 緩める
ulimit -m unlimited
ulimit -v unlimited
ulimit -d unlimited
ulimit -s unlimited
ulimit -n 262144

# PyTorch / CUDA メモリ周辺
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export TORCHDYNAMO_DISABLE=1

mkdir -p outputs/logs

# Conda 環境
eval "$(conda shell.bash hook)"
conda activate qwen_expert_rl

echo "Python path      : $(which python)"
echo "Python version   : $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "#GPUs (local)    : $(nvidia-smi -L | wc -l)"

# 本番 HF モデルパス
export BASE_MODEL="/path/to/base_model"
echo "BASE_MODEL       : $BASE_MODEL"

# データ
export SPECIALIST_CHATML_PATH="/path/to/train.jsonl"

# シーケンス長（ここだけ変えれば pretokenize ファイル名と --seq_len が連動）
export SEQ_LEN="${SEQ_LEN:-8192}"
echo "SEQ_LEN           : $SEQ_LEN"

# 高速化設定（pretokenize + bucketing + dataloader workers）
export TOKENIZED_DATASET="${TOKENIZED_DATASET:-outputs/data/train.tok.seqlen${SEQ_LEN}.jsonl}"
mkdir -p outputs/data
export BUCKET_BUFFER="${BUCKET_BUFFER:-512}"
export NUM_WORKERS="${NUM_WORKERS:-4}"

echo "TOKENIZED_DATASET : $TOKENIZED_DATASET"
echo "BUCKET_BUFFER     : $BUCKET_BUFFER"
echo "NUM_WORKERS       : $NUM_WORKERS"

# rank0 ノードの IP とポート（Accelerate に渡す）
HOST_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export HOST_ADDR

# main_process_port をジョブごとに変える（衝突回避）
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-$((29500 + (${SLURM_JOB_ID:-0} % 500)))}

# 1時間 (3600秒) まで待てるようにする
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

echo "SLURM_NNODES      : $SLURM_NNODES"
echo "HOST_ADDR (rank0) : $HOST_ADDR"
echo "MAIN_PROCESS_PORT : $MAIN_PROCESS_PORT"

# ckpt出力を run-id(SLURM_JOB_ID) で分離
SAVE_DIR_BASE="outputs/ckpts_qwen3_235b_expert_sft_attn70"
SAVE_DIR="${SAVE_DIR_BASE}/${SLURM_JOB_ID}"
mkdir -p "$SAVE_DIR"
echo "SAVE_DIR          : $SAVE_DIR"

# attention 設定
# 235B は 94層 (0-93)。70層目以降 (70-93) の self_attn q/k/v/o_proj を学習。
# expert mask と同時指定しているので OR で両方学習される。
export ATTN_LAYERS="70-93"
export ATTN_PROJ="q_proj,k_proj,v_proj,o_proj"
echo "ATTN_LAYERS       : $ATTN_LAYERS"
echo "ATTN_PROJ         : $ATTN_PROJ"

echo "=== Expert Full-Param SFT (attn) 学習開始 (Accelerate + FSDP) ==="

srun -l --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c '
set -eux

echo "==== Node $(hostname) / SLURM_NODEID=${SLURM_NODEID} ===="

GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-8}
WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))

echo "GPUS_PER_NODE      = $GPUS_PER_NODE"
echo "SLURM_NNODES       = $SLURM_NNODES"
echo "WORLD_SIZE         = $WORLD_SIZE"
echo "This machine_rank  = $SLURM_NODEID"

# tokenized dataset が無い場合は raw fallback（ただし遅い）
DATA_ARGS=()
if [ -f "$TOKENIZED_DATASET" ]; then
  echo "[DATA] Using tokenized dataset: $TOKENIZED_DATASET"
  DATA_ARGS+=( --tokenized_dataset "$TOKENIZED_DATASET" )
else
  echo "[WARN][DATA] tokenized dataset not found: $TOKENIZED_DATASET"
  echo "[WARN][DATA] Falling back to raw dataset (on-the-fly tokenize): $SPECIALIST_CHATML_PATH"
  DATA_ARGS+=( --dataset "$SPECIALIST_CHATML_PATH" )
fi

NCCL_DEBUG=WARN accelerate launch \
  --num_machines "$SLURM_NNODES" \
  --machine_rank "$SLURM_NODEID" \
  --main_process_ip "$HOST_ADDR" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  --num_processes "$WORLD_SIZE" \
  --mixed_precision bf16 \
  run_expert_fullparam_sft.py \
    --model "$BASE_MODEL" \
    --attn_layers "$ATTN_LAYERS" \
    --attn_proj "$ATTN_PROJ" \
    --seq_len "$SEQ_LEN" \
    "${DATA_ARGS[@]}" \
    --bucket_buffer "$BUCKET_BUFFER" \
    --num_workers "$NUM_WORKERS" \
    --epochs 1 \
    --batch_size 1 \
    --lr 2e-5 \
    --log_every 1 \
    --packing \
    --save_dir "'"$SAVE_DIR"'" \
    --wandb
'

EXIT_CODE=$?

echo "=== Expert Full-Param SFT (attn) 学習完了 ==="
echo "終了コード: $EXIT_CODE"
echo "終了時刻: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Expert+Attn full-param SFT 学習が正常に完了しました！"
else
    echo "✗ Expert+Attn full-param SFT 学習でエラーが発生しました。終了コード: $EXIT_CODE"
fi

exit $EXIT_CODE
