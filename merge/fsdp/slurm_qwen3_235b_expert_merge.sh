#!/bin/bash
#SBATCH --job-name=qwen3_235b_expert_merge
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodelist=node-XX
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1          
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=240
#SBATCH --output=outputs/logs/qwen3_235b_expert_merge-%j.out
#SBATCH --error=outputs/logs/qwen3_235b_expert_merge-%j.err

echo "=== Qwen3-235B Expert Merge & Bake (DCP) ジョブ開始 ==="
echo "開始時刻      : $(date)"

# ログを即 flush させる
export PYTHONUNBUFFERED=1

module load nccl/2.22.3
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt

# GPUを完全に無効化してCPUだけ使う
export CUDA_VISIBLE_DEVICES=""

# CPUスレッド
export OMP_NUM_THREADS=8

# ulimit (失敗しても無視)
ulimit -m unlimited  || true
ulimit -v unlimited  || true
ulimit -d unlimited  || true
ulimit -s unlimited  || true
ulimit -n 262144     || true

mkdir -p outputs/logs

# Conda 環境
eval "$(conda shell.bash hook)"
conda activate qwen_expert_rl

echo "Python path      : $(which python)"
echo "Python version   : $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# ベースモデル & チェックポイント & 出力パス
export BASE_MODEL="/path/to/base_model"
export CKPT_DIR="outputs/ckpts_qwen3_235b_expert_grpo"   # accelerator.save_state の保存先
export MASK_JSON="/path/to/selected_experts.json"
export MERGED_MODEL_DIR="/path/to/merged_model"

echo "BASE_MODEL       : $BASE_MODEL"
echo "CKPT_DIR         : $CKPT_DIR"
echo "MASK_JSON        : $MASK_JSON"
echo "MERGED_MODEL_DIR : $MERGED_MODEL_DIR"
echo "SLURM_NNODES     : $SLURM_NNODES"

echo "=== FSDP DCP から selected experts をロードしてベースに焼き込み開始 (1ノード, CPU) ==="

python merge_experts_and_bake_base.py \
  --checkpoint_dir "$CKPT_DIR" \
  --base_model "$BASE_MODEL" \
  --mask_json "$MASK_JSON" \
  --layers 94 \
  --experts 128 \
  --out_dir "$MERGED_MODEL_DIR" \
  --dtype bf16

EXIT_CODE=$?

echo "=== Qwen3-235B Expert Merge & Bake 完了 ==="
echo "終了コード: $EXIT_CODE"
echo "終了時刻: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 統合 / 焼き込みが正常に完了しました！"
else
    echo "✗ 統合 / 焼き込みでエラーが発生しました。終了コード: $EXIT_CODE"
fi

if [ -x /path/to/cleanup_ram.sh ]; then
  /path/to/cleanup_ram.sh || true
fi

exit $EXIT_CODE
