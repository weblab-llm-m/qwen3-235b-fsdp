#!/bin/bash
#SBATCH --job-name=qwen3_235b_merge
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodelist=node-XX
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0        # GPUは不要（CPUメモリを使うため）
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=240      # メモリ確保のため多めに（ノードの仕様に合わせて調整）
#SBATCH --output=outputs/logs/qwen3_235b_merge-%j.out
#SBATCH --error=outputs/logs/qwen3_235b_merge-%j.err

echo "=== Qwen3-235B Sharded Checkpoint Merge ジョブ開始 ==="
echo "開始時刻      : $(date)"

# ログを即 flush させる
export PYTHONUNBUFFERED=1

# GPUを無効化してCPUのみ使用（メモリ節約と誤動作防止）
export CUDA_VISIBLE_DEVICES=""

# CPUスレッド制御
export OMP_NUM_THREADS=16

# ulimit
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

# --- 設定: パス ---
# ベースモデル
export BASE_MODEL="/path/to/base_model"

# 入力: 保存に成功した分割チェックポイントのディレクトリ
# (model_sharded.rank*.pt がある場所)
export CKPT_DIR="/path/to/sharded_checkpoint"

# 出力: マージ済みモデルの保存先
export MERGED_MODEL_DIR="/path/to/merged_model"

echo "BASE_MODEL       : $BASE_MODEL"
echo "CKPT_DIR         : $CKPT_DIR"
echo "MERGED_MODEL_DIR : $MERGED_MODEL_DIR"

echo "=== 分割チェックポイントのロードとベースモデルへの統合 (CPU) ==="

# 新しいマージスクリプトを実行
# メモリ要件: 235B(BF16)のロードに約470GB + 作業領域が必要
export OMP_NUM_THREADS=1
torchrun --standalone --nproc_per_node=64 fsdp_sharded_to_full_torchrun.py \
  --checkpoint_dir "$CKPT_DIR" \
  --base_model "$BASE_MODEL" \
  --out_dir "$MERGED_MODEL_DIR" \
  --dtype bf16 \
  --comm_dtype auto \
  --strict_dim0_only 1 \
  --require_all_named_parameters 0 \
  --send_and_free 1 \
  --torch_threads 1 \
  --log_every 2000


EXIT_CODE=$?

echo "=== Qwen3-235B Merge 完了 ==="
echo "終了コード: $EXIT_CODE"
echo "終了時刻: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ マージと保存が正常に完了しました！"
    echo "出力先: $MERGED_MODEL_DIR"
else
    echo "✗ マージ処理でエラーが発生しました。終了コード: $EXIT_CODE"
fi

if [ -x /path/to/cleanup_ram.sh ]; then
  /path/to/cleanup_ram.sh || true
fi

exit $EXIT_CODE
