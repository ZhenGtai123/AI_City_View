# AI City View

城市街景全景图视觉分析工具。输入一张全景图，自动裁剪为 3 个视角（左 / 中 / 右），对每个视角执行语义分割、深度估计、前中背景分层等分析，生成 25 张分析图片。

提供两种运行方式：

- **Gradio UI** — 本地网页界面，拖拽上传，批量处理
- **FastAPI Server** — HTTP API，供 [GreenSVC](../greensvc) 平台集成调用

---

## 目录

- [系统要求](#系统要求)
- [安装](#安装)
- [运行](#运行)
  - [方式 A：Gradio UI](#方式-a-gradio-ui)
  - [方式 B：FastAPI Server（与 GreenSVC 集成）](#方式-b-fastapi-server-与-greensvc-集成)
  - [方式 C：命令行](#方式-c-命令行)
- [API 文档](#api-文档)
- [处理流程](#处理流程)
- [输出结构](#输出结构)
- [配置说明](#配置说明)
- [常见问题](#常见问题)
- [没有显卡？用 Google Colab](#没有显卡用-google-colab)

---

## 系统要求

| 项目 | 最低要求 |
|------|---------|
| 操作系统 | Windows 10 / 11 |
| GPU | NVIDIA，显存 >= 8 GB（RTX 3060 / 4060 及以上） |
| 内存 | 16 GB |
| 硬盘 | 10 GB 可用空间 |
| Python | 3.10（通过 Miniconda 管理） |

> 查看显卡型号：`Ctrl+Shift+Esc` → 性能 → GPU。
> 没有 NVIDIA 显卡或显存不足？跳到 [Google Colab](#没有显卡用-google-colab) 部分。

---

## 安装

整个过程约 20–30 分钟，取决于网速。

### 1. 安装 Miniconda

1. 下载 [Miniconda3 Windows 64-bit](https://docs.conda.io/en/latest/miniconda.html)
2. 安装时勾选 **Add Miniconda3 to my PATH**
3. 安装完成后，打开 **开始菜单** → 搜索 **Anaconda Prompt**

验证：

```
conda --version
```

> 以下所有步骤均在 Anaconda Prompt 中操作。

### 2. 创建 Python 环境

```bash
conda create -n cityview python=3.10 -y
conda activate cityview
```

窗口前缀从 `(base)` 变为 `(cityview)` 即成功。

### 3. 安装 PyTorch（CUDA 12.4）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> 下载约 2–3 GB，耐心等待。

验证 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

输出 `True` 则成功。输出 `False` 见 [常见问题](#常见问题)。

### 4. 安装项目依赖

```bash
cd D:\green-svc\AI_City_View
pip install -r requirements.txt
```

### 5. 安装加速组件（推荐）

```bash
pip install xformers
```

安装完成。以后每次使用只需要重复 "运行" 步骤。

> **注意：第一次处理图片时，程序会自动下载 AI 模型（约 2-3 GB），需要联网，可能要等几分钟。下载一次后会缓存到本地，以后不需要重复下载。**

---

## 运行

每次使用前先激活环境：

```bash
conda activate cityview
cd D:\green-svc\AI_City_View
```

### 方式 A: Gradio UI

适合独立使用，拖拽上传图片，实时查看进度。

```bash
python app.py
```

打开浏览器访问 **http://localhost:7860**

操作步骤：
1. 拖拽全景图到上传区域（支持多选 JPG/PNG）
2. 输出目录保持默认 `output`
3. 点击 **开始处理**
4. 每张约 30–40 秒，日志区显示进度
5. 结果保存在 `AI_City_View/output/` 中

### 方式 B: FastAPI Server（与 GreenSVC 集成）

GreenSVC 平台的 Vision API 后端。启动后 GreenSVC 后端会通过 HTTP 调用此服务。

```bash
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
```

或者直接：

```bash
python server.py
```

服务启动后：
- API 地址：**http://localhost:8000**
- 交互文档：**http://localhost:8000/docs**
- 首次启动会预加载模型（约 30–60 秒）

> 确保 GreenSVC 后端 `.env` 中设置 `VISION_API_URL=http://127.0.0.1:8000`。

### 方式 C: 命令行

直接处理单张全景图：

```bash
python main.py <图片路径> <输出目录>

# 示例
python main.py full1.jpg output
```

---

## API 文档

FastAPI Server 提供以下端点：

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
- `request_data` — JSON 字符串，包含以下字段：

```json
{
  "semantic_classes": ["Sky", "Trees", "Building", "Road"],
  "semantic_countability": [0, 1, 0, 0],
  "openness_list": [1, 0, 0, 0],
  "encoder": "vitb",
  "detection_threshold": 0.3,
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
  "detected_classes": 12,
  "total_classes": 15,
  "class_statistics": {
    "Sky": { "pixel_count": 50000, "percentage": 25.5 },
    "Trees": { "pixel_count": 30000, "percentage": 15.3 }
  },
  "fmb_statistics": { ... },
  "download_url": "/outputs/img_a1b2c3d4_1708900000/download",
  "processing_time": 12.5
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
Stage 2: AI 推理 ─── 语义分割 (OneFormer ADE20K) + 深度估计 (Depth Anything V3)
    │
    ▼
Stage 3: 后处理 ─── 语义图清洗、噪点去除
    │
    ▼
Stage 4: FMB 分层 ── 基于度量深度的前景 / 中景 / 背景分离
    │
    ▼
Stage 5: 开放度 ─── 基于语义类的空间开放度计算
    │
    ▼
Stage 6: 生成图片 ── 23 张分析图 (5 基础图 × 3 FMB 层 + 8 基础/掩码图)
    │
    ▼
Stage 7: 保存输出 ── 23 PNG + sky_mask + semantic_raw + metadata.json + depth_metric.npy
```

---

## 输出结构

每张全景图生成 3 个视角文件夹，每个文件夹约 25 个文件：

```
output/
├── {图片名}_left/
│   ├── semantic_map.png          # 语义分割图
│   ├── depth_map.png             # 深度估计图
│   ├── openness_map.png          # 开放度图
│   ├── fmb_map.png               # 前中背景分层总图
│   ├── original.png              # 原图
│   ├── foreground_map.png        # 前景掩码
│   ├── middleground_map.png      # 中景掩码
│   ├── background_map.png        # 背景掩码
│   ├── semantic_foreground.png   # 语义 × 前景
│   ├── semantic_middleground.png # 语义 × 中景
│   ├── semantic_background.png   # 语义 × 背景
│   ├── depth_foreground.png      # 深度 × 前景
│   ├── depth_middleground.png    # 深度 × 中景
│   ├── depth_background.png      # 深度 × 背景
│   ├── openness_foreground.png   # 开放度 × 前景
│   ├── openness_middleground.png # 开放度 × 中景
│   ├── openness_background.png   # 开放度 × 背景
│   ├── original_foreground.png   # 原图 × 前景
│   ├── original_middleground.png # 原图 × 中景
│   ├── original_background.png   # 原图 × 背景
│   ├── fmb_foreground.png        # FMB × 前景
│   ├── fmb_middleground.png      # FMB × 中景
│   ├── fmb_background.png        # FMB × 背景
│   ├── sky_mask.png              # 天空掩码
│   ├── semantic_raw.png          # 原始语义 class ID
│   ├── depth_metric.npy          # 度量深度数据 (float32, 米)
│   └── metadata.json             # 处理元数据 + FMB 统计
├── {图片名}_front/
│   └── (同上)
└── {图片名}_right/
    └── (同上)
```

---

## 配置说明

### Semantic_configuration.json

语义类定义文件，位于项目根目录。每个条目包含：

```json
{
  "name": "Trees",
  "color": "#00FF00",
  "countable": 1,
  "openness": 0
}
```

- `name` — 语义类名称
- `color` — 可视化颜色（HEX）
- `countable` — 是否可计数（0/1）
- `openness` — 开放度贡献（0/1）

### 深度估计模型

在 `main.py` 的 `get_default_config()` 中配置：

| 模型 | 大小 | 输出 | 显存需求 |
|------|------|------|---------|
| `DA3METRIC-LARGE` | 0.35B | 规范化深度（需焦距转换为米） | 8 GB（推荐） |
| `DA3NESTED-GIANT-LARGE-1.1` | 1.4B | 真实米数 + 天空检测 | 16 GB+ |
| `DA3MONO-LARGE` | 0.35B | 相对深度（无米数） | 8 GB |

---

## 常见问题

### 找不到 Anaconda Prompt？

安装 Miniconda 后重启电脑，再搜索 "Anaconda Prompt"。

### `torch.cuda.is_available()` 返回 False？

1. 确认电脑有 NVIDIA 独立显卡（任务管理器 → 性能 → GPU）
2. 更新显卡驱动：[NVIDIA 驱动下载](https://www.nvidia.cn/drivers/)
3. 重新打开 Anaconda Prompt，激活环境后重试

### 安装依赖报错？

- 确认窗口前缀是 `(cityview)` 而不是 `(base)`，否则先 `conda activate cityview`
- 升级 pip：`python -m pip install --upgrade pip`

### 处理速度？

每张全景图约 30–40 秒（RTX 3060–3070）。GPU 推理（Stage 2）是瓶颈，后续阶段在 CPU 上并行处理。

### 端口被占用？

```bash
# Gradio
python app.py              # 默认 7860

# FastAPI — 可通过环境变量或参数改端口
python -m uvicorn server:app --port 8001
```

---

## 没有显卡？用 Google Colab

Google Colab 提供免费 GPU（T4），无需本地安装。

1. 打开 [Google Colab](https://colab.research.google.com)（需 Google 账号）
2. 新建笔记本 → **运行时 → 更改运行时类型** → 选 **T4 GPU** → 保存
3. 逐段粘贴以下代码，按 `Shift+Enter` 运行：

**安装环境：**

```python
!git clone https://github.com/ZhenGtai123/AI_City_View.git
%cd AI_City_View
!pip install -r requirements.txt
!pip install xformers
```

**上传图片并处理：**

```python
from google.colab import files
uploaded = files.upload()

from main import process_panorama
for name in uploaded:
    process_panorama(name, "output")
```

**下载结果：**

```python
!zip -r output.zip output/
files.download("output.zip")
```

---

## 项目结构

```
AI_City_View/
├── app.py                 # Gradio UI 入口
├── server.py              # FastAPI Vision API 入口
├── main.py                # 全景图处理核心逻辑
├── pipeline/
│   ├── stage1_preprocess.py        # 等距柱状投影裁剪
│   ├── stage2_ai_inference.py      # 语义分割 + 深度估计 (GPU)
│   ├── stage3_postprocess.py       # 语义图后处理
│   ├── stage4_intelligent_fmb.py   # 智能 FMB 分层
│   ├── stage4_depth_layering.py    # 深度分层 (备选)
│   ├── stage5_openness.py          # 开放度计算
│   ├── stage6_generate_images.py   # 生成 23 张分析图
│   ├── stage7_save_outputs.py      # 保存输出
│   ├── ade20k_palette.py           # ADE20K 调色板
│   └── gpu_utils.py                # GPU 工具函数
├── Semantic_configuration.json     # 语义类配置
├── requirements.txt                # Python 依赖
└── output/                         # 默认输出目录
```
