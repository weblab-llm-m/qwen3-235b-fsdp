#!/bin/bash
#SBATCH --job-name=qwen235b_merge_loras
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --nodelist=node-XX
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -eEo pipefail

mkdir -p outputs/logs

WORKDIR="${WORKDIR_OVERRIDE:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${WORKDIR}"
echo "=== [START] $(date) host=$(hostname) jobid=${SLURM_JOB_ID:-} pwd=$(pwd) ==="

module reset || true
module load cuda/12.1 || true
module load cudnn/8.9.7 || true
module load nccl/2.21.5 || true
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt || true

export CUDNN_HOME=/home/appli/cudnn/8.9.7
export NCCL_HOME=/home/appli/nccl/2.21.5
export NCCL_SOCKET_IFNAME=enp25s0np0
export LD_LIBRARY_PATH=${CUDA_PATH:-/usr/local/cuda}/lib64:$CUDNN_HOME/lib:$NCCL_HOME/lib:${LD_LIBRARY_PATH:-}
export NVTE_FUSED_ATTN=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export PYTORCH_ALLOC_CONF=expandable_segments:True

CONDA_BIN="${CONDA_BIN:-conda}"
eval "$(${CONDA_BIN} shell.bash hook)"

MERGE_ENV="${MERGE_ENV:-qwen_lora_rl}"

# ----------------
# Paths - 特定ステップを直接指定
# ----------------
BASE_MODEL="${BASE_MODEL:-/path/to/base_model}"

declare -a LORA_DIRS=(
  "outputs/ckpts_qwen3_235b_lora_grpo/adapters/train-lora/step_00000100"
  "outputs/ckpts_qwen3_235b_lora_gspo/adapters/train-lora/step_00000075"
)

declare -a OUT_DIRS=(
  "/path/to/merged_model_grpo_step100"
  "/path/to/merged_model_gspo_step75"
)

echo "=== CONFIG ==="
echo "BASE_MODEL=${BASE_MODEL}"
for i in "${!LORA_DIRS[@]}"; do
  echo "LORA[$i]=${LORA_DIRS[$i]} -> OUT[$i]=${OUT_DIRS[$i]}"
done

# sanity check
if [ ! -f "${BASE_MODEL}/config.json" ]; then
  echo "ERROR: BASE_MODEL missing config.json: ${BASE_MODEL}"
  exit 1
fi

merge_lora_to_out() {
  local lora_path="$1"
  local out_dir="$2"

  # 指定パスにadapter_config.jsonがあるか確認
  if [ ! -f "${lora_path}/adapter_config.json" ]; then
    echo "ERROR: adapter_config.json not found in: ${lora_path}"
    exit 2
  fi

  echo "=== [MERGE] ==="
  echo "base=${BASE_MODEL}"
  echo "lora=${lora_path}"
  echo "out =${out_dir}"

  if [ -e "${out_dir}" ] && [ "${OVERWRITE:-0}" != "1" ]; then
    echo "ERROR: out_dir already exists: ${out_dir}"
    echo "       set OVERWRITE=1 to overwrite."
    exit 3
  fi

  [ "${OVERWRITE:-0}" = "1" ] && rm -rf "${out_dir}"
  mkdir -p "$(dirname "${out_dir}")"

  conda activate "${MERGE_ENV}"

  python - <<PY
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "${BASE_MODEL}"
lora = "${lora_path}"
out  = "${out_dir}"

os.makedirs(out, exist_ok=True)

tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
dtype = torch.bfloat16

model = AutoModelForCausalLM.from_pretrained(
    base,
    trust_remote_code=True,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    device_map="auto",
)

try:
    peft = PeftModel.from_pretrained(model, lora, is_trainable=False, ensure_weight_tying=True)
except TypeError:
    peft = PeftModel.from_pretrained(model, lora, is_trainable=False)

merged = peft.merge_and_unload()

try:
    merged.tie_weights()
except Exception:
    pass

merged.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
tok.save_pretrained(out)

print("[MERGE] saved to", out, flush=True)
PY

  conda deactivate || true
}

# ----------------
# main
# ----------------
if [ "${#LORA_DIRS[@]}" -ne "${#OUT_DIRS[@]}" ]; then
  echo "ERROR: LORA_DIRS and OUT_DIRS must have same length."
  exit 1
fi

for i in "${!LORA_DIRS[@]}"; do
  merge_lora_to_out "${LORA_DIRS[$i]}" "${OUT_DIRS[$i]}"
done

echo "=== [DONE] merged models saved ==="
for o in "${OUT_DIRS[@]}"; do
  echo " - ${o}"
done
