# AI City View

城市街景全景图视觉分析 API。输入一张全景图，自动裁剪为 3 个视角（左 / 中 / 右），对每个视角执行语义分割、深度估计、前中背景分层等分析，生成 25 张分析图片。

通过 FastAPI 提供 HTTP API，供 [GreenSVC](../greensvc) 平台集成调用。

---

## 系统要求

| 项目 | 最低要求 |
|------|---------|
| GPU | NVIDIA，显存 >= 8 GB（RTX 3060 / 4060 及以上） |
| 内存 | 16 GB |
| Python | 3.10 |

---

## 安装

```bash
git clone https://github.com/ZhenGtai123/AI_City_View.git
cd AI_City_View

conda create -n cityview python=3.10 -y
conda activate cityview

# 1. 先装 PyTorch CUDA 版（必须在 requirements.txt 之前！）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2. 验证 GPU（必须显示 True 再继续）
python -c "import torch; print(torch.cuda.is_available())"

# 3. 装项目依赖（不会覆盖已装的 PyTorch）
pip install -r requirements.txt

# 4. 装 Depth Anything V3（从 GitHub）
pip install git+https://github.com/ByteDance-Seed/depth-anything-3.git

# 5. 可选加速
pip install xformers
```

> 第一次启动会自动下载 AI 模型（OneFormer ~1.2GB + DA3 ~1.4GB），之后会缓存到本地。

---

## 运行

### FastAPI Server（主要方式）

```bash
conda activate cityview
python server.py
```

- API 地址：**http://localhost:8000**
- 交互文档：**http://localhost:8000/docs**
- 首次启动会预加载模型（约 30–60 秒）

自定义端口：

```bash
PORT=8001 python server.py
```

> 确保 GreenSVC 后端 `.env` 中设置 `VISION_API_URL=http://127.0.0.1:8000`。

### 命令行（单张处理）

```bash
python main.py <图片路径> <输出目录>
python main.py full1.jpg output
```

### 批量处理（本地）

```bash
python batch_run.py /data/input /data/output --workers=2
```

### 云批量处理（Azure Blob → GPU VM → GCS）

适合50万张级别的大规模处理。支持断点续跑、Spot 抢占优雅退出、下载/处理/上传三阶段流水线并行。

```bash
# 额外依赖
pip install azure-storage-blob google-cloud-storage

# 运行
python cloud_batch_run.py \
  --azure-sas-url "https://account.blob.core.windows.net/container?sv=..." \
  --gcs-bucket my-output-bucket \
  --workers=4 \
  --gpu-concurrency=2
```

---

## API 文档

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/analyze` | 分析单张图片，返回 hex 编码图像 + 统计数据 |
| `POST` | `/analyze/panorama` | 全景图模式：自动裁剪 3 个视角并分析 |
| `GET` | `/health` | 健康检查（GPU 状态、已加载模型） |
| `GET` | `/config` | 返回语义类配置 (`Semantic_configuration.json`) |
| `GET` | `/outputs/{job_id}/download` | 下载某次分析的全部输出（ZIP） |
| `GET` | `/outputs/{job_id}/{filename}` | 下载单个输出文件 |

### POST /analyze 请求格式

`multipart/form-data`：
- `file` — 图片文件
- `request_data` — JSON 字符串：

```json
{
  "semantic_classes": ["Sky", "Trees", "Building", "Road"],
  "semantic_countability": [0, 1, 0, 0],
  "openness_list": [1, 0, 0, 0],
  "enable_hole_filling": true,
  "image_id": "optional_custom_id"
}
```

### 响应格式

```json
{
  "status": "success",
  "job_id": "img_a1b2c3d4_1708900000",
  "images": {
    "semantic_map": "<hex-encoded PNG>",
    "depth_map": "<hex-encoded PNG>",
    "openness_map": "...",
    "fmb_map": "...",
    "foreground_map": "...",
    "middleground_map": "...",
    "background_map": "...",
    "sky_mask": "...",
    "semantic_raw": "..."
  },
  "class_statistics": {
    "sky": { "pixel_count": 50000, "percentage": 25.5 },
    "tree": { "pixel_count": 30000, "percentage": 15.3 }
  },
  "fmb_statistics": { ... },
  "download_url": "/outputs/img_a1b2c3d4_1708900000/download",
  "processing_time": 6.2
}
```

---

## 处理流程

```
全景图输入
    │
    ▼
Stage 1: 预处理 ─── 等距柱状投影裁剪 → left / front / right (90° FOV)
    │
    ▼
Stage 2: AI 推理 ─── OneFormer (语义分割, ADE20K-150) + Depth Anything V3 (度量深度)
    │
    ▼
Stage 3: 后处理 ─── 语义图清洗、噪点去除
    │
    ▼
Stage 4: FMB 分层 ── 前景 (0-10m) / 中景 (10-50m) / 背景 (>50m) / 天空
    │
    ▼
Stage 5: 开放度 ─── 基于语义类的空间开放度计算
    │
    ▼
Stage 6: 生成图片 ── 23 张分析图
    │
    ▼
Stage 7: 保存输出 ── 23 PNG + sky_mask + semantic_raw + metadata.json + depth_metric.npy
```

---

## 输出结构

每张全景图生成 3 个视角文件夹，每个文件夹 25 个文件：

```
output/
├── {图片名}_left/
│   ├── semantic_map.png
│   ├── depth_map.png
│   ├── openness_map.png
│   ├── fmb_map.png
│   ├── original.png
│   ├── foreground_map.png / middleground_map.png / background_map.png
│   ├── semantic_foreground.png / semantic_middleground.png / semantic_background.png
│   ├── depth_foreground.png / depth_middleground.png / depth_background.png
│   ├── openness_foreground.png / openness_middleground.png / openness_background.png
│   ├── original_foreground.png / original_middleground.png / original_background.png
│   ├── fmb_foreground.png / fmb_middleground.png / fmb_background.png
│   ├── sky_mask.png
│   ├── semantic_raw.png
│   ├── depth_metric.npy
│   └── metadata.json
├── {图片名}_front/
└── {图片名}_right/
```

---

## 配置

### 深度估计模型

在 `server.py` 的 `get_default_config()` 中配置：

| 模型 | 参数量 | 输出 | 显存需求 |
|------|--------|------|---------|
| `DA3METRIC-LARGE`（默认） | 0.35B | 规范化深度 → 米数 | 8 GB |
| `DA3NESTED-GIANT-LARGE-1.1` | 1.4B | 真实米数 + 天空检测 | 16 GB+ |
| `DA3MONO-LARGE` | 0.35B | 相对深度（无米数） | 8 GB |

### Semantic_configuration.json

语义类定义文件：

```json
{
  "name": "Trees",
  "color": "#00FF00",
  "countable": 1,
  "openness": 0
}
```

---

## 项目结构

```
AI_City_View/
├── server.py              # FastAPI API 入口
├── main.py                # 全景图处理核心逻辑
├── batch_run.py           # 本地批量处理脚本
├── cloud_batch_run.py     # 云批量处理 (Azure Blob → GCS)
├── pipeline/
│   ├── stage1_preprocess.py        # 等距柱状投影裁剪
│   ├── stage2_ai_inference.py      # OneFormer + DA3 (GPU)
│   ├── stage3_postprocess.py       # 语义图后处理
│   ├── stage4_intelligent_fmb.py   # 智能 FMB 分层
│   ├── stage4_depth_layering.py    # 深度分层 (备选)
│   ├── stage5_openness.py          # 开放度计算
│   ├── stage6_generate_images.py   # 生成分析图
│   ├── stage7_save_outputs.py      # 保存输出
│   └── gpu_utils.py                # GPU 工具函数
├── Semantic_configuration.json     # 语义类配置
├── requirements.txt                # Python 依赖
└── output/                         # 默认输出目录
```
