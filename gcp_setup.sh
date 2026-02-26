#!/bin/bash
# =============================================================================
# GCP Spot VM 启动/恢复脚本
# 功能: 首次部署安装依赖，抢占恢复后自动续跑
#
# 设置方法 (创建 VM 时传入 metadata):
#   gcloud compute instances create ai-city-runner \
#     --metadata-from-file startup-script=gcp_setup.sh \
#     --metadata=azure-sas-url="https://...",gcs-bucket=my-bucket,run-user=sky
# =============================================================================

set -eo pipefail

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
# 从 VM metadata 获取参数 (startup script 以 root 运行，$USER=root)
META_URL="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
META_HEADER="Metadata-Flavor: Google"

get_meta() { curl -sf -H "$META_HEADER" "$META_URL/$1" 2>/dev/null || echo ""; }

RUN_USER=$(get_meta "run-user")
RUN_USER=${RUN_USER:-"sky"}  # 默认用户名

PROJECT_DIR="/home/$RUN_USER/AI_City_View"
VENV_DIR="$PROJECT_DIR/.venv"
LOG_FILE="$PROJECT_DIR/cloud_batch_run.log"
PID_FILE="$PROJECT_DIR/.runner_pid"

AZURE_SAS_URL=$(get_meta "azure-sas-url")
AZURE_CONN_STR=$(get_meta "azure-conn-str")
AZURE_CONTAINER=$(get_meta "azure-container")
GCS_BUCKET=$(get_meta "gcs-bucket")
GCS_PREFIX=$(get_meta "gcs-prefix")
GCS_PREFIX=${GCS_PREFIX:-"output"}

DEPTH_RES=$(get_meta "depth-res")
DEPTH_RES=${DEPTH_RES:-672}
PNG_COMPRESSION=$(get_meta "png-compression")
PNG_COMPRESSION=${PNG_COMPRESSION:-6}

LOCAL_BUFFER="/tmp/ai_city_buffer"

# ---------------------------------------------------------------------------
# 0. 防止重复启动
# ---------------------------------------------------------------------------
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "处理已在运行 (PID $OLD_PID)，跳过启动"
        exit 0
    fi
    rm -f "$PID_FILE"
fi

# ---------------------------------------------------------------------------
# 1. 安装系统依赖 (仅首次)
# ---------------------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo "=== 首次部署：安装依赖 ==="

    apt-get update -qq
    apt-get install -y -qq python3-venv python3-pip libgl1 libglib2.0-0 git

    if [ ! -d "$PROJECT_DIR" ]; then
        echo "错误: 项目代码不存在于 $PROJECT_DIR，请先部署"
        exit 1
    fi

    cd "$PROJECT_DIR"
    sudo -u "$RUN_USER" python3 -m venv "$VENV_DIR"
    sudo -u "$RUN_USER" bash -c "source $VENV_DIR/bin/activate && pip install --upgrade pip && pip install -r requirements.txt && pip install azure-storage-blob google-cloud-storage"

    echo "=== 依赖安装完成 ==="
fi

# ---------------------------------------------------------------------------
# 2. 检查 GPU
# ---------------------------------------------------------------------------
echo "=== GPU 状态 ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || {
    echo "错误: GPU 不可用，中止启动"
    exit 1
}

sudo -u "$RUN_USER" bash -c "source $VENV_DIR/bin/activate && python3 -c \"import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'PyTorch CUDA OK: {torch.cuda.get_device_name(0)}')\"" || {
    echo "错误: PyTorch CUDA 不可用，中止启动"
    exit 1
}

# ---------------------------------------------------------------------------
# 3. 参数校验
# ---------------------------------------------------------------------------
if [ -z "$GCS_BUCKET" ]; then
    echo "错误: 缺少 gcs-bucket metadata"
    exit 1
fi

if [ -z "$AZURE_SAS_URL" ] && { [ -z "$AZURE_CONN_STR" ] || [ -z "$AZURE_CONTAINER" ]; }; then
    echo "错误: 缺少 Azure 连接信息 (azure-sas-url 或 azure-conn-str + azure-container)"
    exit 1
fi

# ---------------------------------------------------------------------------
# 4. 构建命令并启动 (用 exec 确保 SIGTERM 直达 Python 进程)
# ---------------------------------------------------------------------------
echo "=== 启动/恢复处理 ==="

# 使用数组构建命令，避免 SAS URL 中的特殊字符被 shell 解析
CMD_ARGS=(
    "$VENV_DIR/bin/python3" "$PROJECT_DIR/cloud_batch_run.py"
    --gcs-bucket "$GCS_BUCKET"
    --gcs-prefix "$GCS_PREFIX"
    --local-buffer "$LOCAL_BUFFER"
    --depth-res "$DEPTH_RES"
    --png-compression "$PNG_COMPRESSION"
)

if [ -n "$AZURE_SAS_URL" ]; then
    CMD_ARGS+=(--azure-sas-url "$AZURE_SAS_URL")
elif [ -n "$AZURE_CONN_STR" ]; then
    CMD_ARGS+=(--azure-conn-str "$AZURE_CONN_STR" --azure-container "$AZURE_CONTAINER")
fi

echo "日志: $LOG_FILE"
echo "用户: $RUN_USER"

cd "$PROJECT_DIR"
# 以目标用户运行，用 exec 替换 shell 进程保证 SIGTERM 直达 Python
sudo -u "$RUN_USER" nohup "${CMD_ARGS[@]}" >> "$LOG_FILE" 2>&1 &
PROC_PID=$!
echo "$PROC_PID" > "$PID_FILE"

echo "进程 PID: $PROC_PID"
echo "=== 启动完成，处理在后台运行 ==="
echo "查看进度: tail -f $LOG_FILE"
echo "安全停止: kill -TERM $PROC_PID"
