#!/bin/bash
# =============================================================================
# AI City View - 一键安装脚本 (Ubuntu 24.04 + NVIDIA GPU)
# 适用于: GCP, Vast.ai, RunPod, Lambda, 或任何有 GPU 的 Ubuntu 24 机器
#
# 用法:
#   chmod +x install.sh && ./install.sh
#
# 前提:
#   - Ubuntu 24.04
#   - NVIDIA 驱动已安装 (nvidia-smi 可用)
#   - 有网络访问
# =============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "============================================"
echo " AI City View - 环境安装"
echo " 目录: $SCRIPT_DIR"
echo "============================================"

# ---------------------------------------------------------------------------
# 1. 系统依赖
# ---------------------------------------------------------------------------
echo ""
echo "[1/5] 安装系统依赖..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-venv python3-pip python3-dev \
    libgl1 libglib2.0-0 \
    git wget curl

# ---------------------------------------------------------------------------
# 2. 检查 GPU
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] 检查 GPU..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "错误: nvidia-smi 不可用，请先安装 NVIDIA 驱动"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "GPU 检查通过"

# ---------------------------------------------------------------------------
# 3. 创建虚拟环境
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] 创建 Python 虚拟环境..."
if [ -d "$VENV_DIR" ]; then
    echo "虚拟环境已存在，跳过创建"
else
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q

# ---------------------------------------------------------------------------
# 4. 安装 Python 依赖
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] 安装 Python 依赖..."

# PyTorch (CUDA 12.x)
pip install  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 核心依赖
pip install  \
    opencv-python>=4.8.0 \
    numpy>=1.24.0 \
    Pillow>=10.0.0 \
    transformers>=4.30.0 \
    accelerate>=0.20.0 \
    tqdm>=4.65.0

# LangSAM (语义分割)
pip install  git+https://github.com/luca-medeiros/lang-segment-anything.git

# 云存储 SDK
pip install  azure-storage-blob google-cloud-storage

# ---------------------------------------------------------------------------
# 5. 验证安装
# ---------------------------------------------------------------------------
echo ""
echo "[5/5] 验证安装..."

python3 -c "
import torch
import cv2
import numpy as np
from PIL import Image

print(f'  Python:     {__import__(\"sys\").version.split()[0]}')
print(f'  PyTorch:    {torch.__version__}')
print(f'  CUDA:       {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"})')
print(f'  OpenCV:     {cv2.__version__}')
print(f'  NumPy:      {np.__version__}')

assert torch.cuda.is_available(), 'CUDA 不可用!'
print()
print('  所有检查通过!')
"

echo ""
echo "============================================"
echo " 安装完成!"
echo ""
echo " 激活环境:  source $VENV_DIR/bin/activate"
echo ""
echo " 测试运行:"
echo "   python cloud_batch_run.py \\"
echo "     --azure-conn-str \"...\" \\"
echo "     --azure-container CONTAINER \\"
echo "     --gcs-bucket BUCKET \\"
echo "     --limit 10"
echo "============================================"
