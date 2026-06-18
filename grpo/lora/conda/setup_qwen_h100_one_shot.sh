#!/usr/bin/env bash
set -eo pipefail   # -u は付けない（/etc/bashrc 問題を避ける）

ENV_NAME=${ENV_NAME:-qwen_lora_rl}
PY_VER=${PY_VER:-3.10}
QWEN_DIR="${HOME}/Qwen"

# 0) conda をロード（~/.bashrcを読まない安全ルート）
eval "$($(conda info --base)/bin/conda shell.bash hook)"

# 1) 同名環境があれば削除
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] Removing existing env: ${ENV_NAME}"
  conda deactivate || true
  conda remove -y -n "${ENV_NAME}" --all
fi

# 2) 新規作成＆有効化
echo "[INFO] Creating env: ${ENV_NAME} (python=${PY_VER})"
conda create -y -n "${ENV_NAME}" python="${PY_VER}"
conda activate "${ENV_NAME}"

# 3) PyTorch (CUDA 12.4, H100 OK)
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch torchvision torchaudio

# 4) CUDA Toolkit を conda で導入（nvcc を確実に入れる）
conda install -y -c nvidia cuda-toolkit=12.4

# 5) CUDA_HOME をこのenvに固定（永続化）
export CUDA_HOME="$CONDA_PREFIX"
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/cuda_home.sh" <<'EOS'
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
EOS
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/cuda_home.sh" <<'EOS'
unset CUDA_HOME
EOS

echo "[INFO] CUDA_HOME set to $CUDA_HOME"
if [ -x "$CUDA_HOME/bin/nvcc" ]; then
  "$CUDA_HOME/bin/nvcc" --version || true
else
  echo "[ERROR] nvcc not found in $CUDA_HOME/bin" >&2
  exit 1
fi

# 6) 依存（ビルドに必要なもの含む）
pip install -U pip setuptools wheel packaging ninja

# 7) DeepSpeed をビルドインストール（CUDAあり）
pip install deepspeed==0.17.5 --no-build-isolation

# 8) そのほか必要ライブラリ
pip install -U transformers datasets peft accelerate trl[vllm] hf_transfer
pip install "bitsandbytes>=0.43.3"
pip install -U wandb

# Flash-Attention：ビルド環境が整っていれば有効。不要ならコメントのままでOK
pip install flash-attn --no-build-isolation

# 9) Qwen 公式リポジトリ（finetune.py を使うため）
if [ -d "${QWEN_DIR}/.git" ]; then
  echo "[INFO] Updating ${QWEN_DIR}"
  git -C "${QWEN_DIR}" pull --ff-only || true
else
  echo "[INFO] Cloning Qwen repo to ${QWEN_DIR}"
  git clone https://github.com/QwenLM/Qwen.git "${QWEN_DIR}"
fi

# 10) 検証：torch/DS/CUDA が見えるか
python - <<'PY'
import os, torch, deepspeed, subprocess
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("DeepSpeed:", deepspeed.__version__)
print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
nvcc = os.path.join(os.environ.get("CUDA_HOME",""), "bin", "nvcc")
print("Has nvcc:", os.path.exists(nvcc))
try:
    if os.path.exists(nvcc):
        subprocess.run([nvcc, "--version"], check=False)
except Exception as e:
    print("nvcc check error:", e)
PY

echo "[OK] Environment '${ENV_NAME}' is ready."
echo "Next: conda activate ${ENV_NAME}"
