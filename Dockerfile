# =============================================================================
# AI City View - Vision API
# Base: NVIDIA CUDA 12.4 runtime + cuDNN (Ubuntu 22.04)
#
# Build:
#   docker build -t scenerx/vision-api .
# Run (requires NVIDIA Container Toolkit):
#   docker run --gpus all -p 8000:8000 \
#     -e VISION_DEPTH_MODEL=DA3METRIC-LARGE \
#     -v hf-cache:/root/.cache/huggingface \
#     scenerx/vision-api
# =============================================================================

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

# System dependencies + Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
        python3-pip \
        git \
        libgl1 libglib2.0-0 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python/python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# PyTorch with CUDA 12.4 — install first so it's a separate cached layer
RUN pip install \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124

# Core dependencies (mirrors requirements.txt + install.sh)
RUN pip install \
        opencv-python>=4.8.0 \
        "numpy>=1.24.0,<2" \
        Pillow>=10.0.0 \
        transformers>=4.30.0 \
        accelerate>=0.20.0 \
        scikit-learn>=1.3.0 \
        scipy>=1.10.0 \
        fastapi>=0.100.0 \
        "uvicorn[standard]>=0.23.0" \
        python-multipart>=0.0.6 \
        tqdm>=4.65.0

# Depth Anything V3 (ByteDance fork) — installed from GitHub.
# `--ignore-installed` skirts the Ubuntu 22.04 distutils-installed
# `python3-blinker 1.4` which pip can't uninstall cleanly when DA3's
# transitive deps want a newer blinker. Without this flag the build
# fails with `uninstall-distutils-installed-package`.
RUN pip install --ignore-installed blinker \
    && pip install git+https://github.com/ByteDance-Seed/depth-anything-3.git

# Application code
COPY . .

# Persistent caches and outputs
VOLUME ["/root/.cache/huggingface", "/app/outputs"]

EXPOSE 8000

# Use uvicorn directly so signal handling works cleanly in containers
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
