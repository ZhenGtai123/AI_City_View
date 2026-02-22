# AI City View — 全景图分析工具

输入一张全景图，自动裁剪为 left / front / right 三个视角，每个视角输出 25 个分析文件（语义分割、深度估计、前中背景分层等）。

---

## 系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10 / 11 |
| 显卡 | NVIDIA，至少 **8 GB 显存**（RTX 3060 / 3070 / 3080 / 4060 / 4070 / 4080 等） |
| 内存 | 16 GB 以上 |
| 磁盘空间 | 约 10 GB（模型权重 + conda 环境） |

---

## 安装步骤（一步一步来）

### 第 1 步：安装 Miniconda

1. 打开浏览器，访问 https://docs.conda.io/en/latest/miniconda.html
2. 下载 **Windows 64-bit** 安装包
3. 双击运行安装程序，一路点"下一步"即可（建议勾选"Add to PATH"）
4. 安装完成后，打开 **Anaconda Prompt**（在开始菜单中搜索），输入以下命令验证：

```bash
conda --version
```

如果显示版本号（例如 `conda 24.x.x`），说明安装成功。

### 第 2 步：创建 Python 环境

在 Anaconda Prompt 中依次输入：

```bash
conda create -n cityview python=3.10 -y
conda activate cityview
```

> 以后每次使用本工具前，都需要先运行 `conda activate cityview`。

### 第 3 步：安装 PyTorch（GPU 版）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

安装完成后验证 GPU 是否可用：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

如果输出 `True`，说明 GPU 版 PyTorch 安装成功。如果输出 `False`，请参考下方"常见问题"。

### 第 4 步：下载项目代码

```bash
git clone <本项目的仓库地址>
cd AI_City_View
```

### 第 5 步：安装项目依赖

```bash
pip install -r requirements.txt
```

### 第 6 步：安装 xformers（可选，加速推理）

```bash
pip install xformers
```

> xformers 可以让深度估计模型运行更快、占用更少显存。推荐安装。

---

## 使用方法

### 方式一：图形界面（推荐零基础用户）

```bash
conda activate cityview
python app.py
```

运行后会显示类似 `Running on local URL: http://0.0.0.0:7860` 的信息。
打开浏览器，访问 **http://localhost:7860** ，然后：

1. **拖拽或点击上传**全景图（支持多选）
2. 设置输出目录（默认 `output`）
3. 点击 **🚀 开始处理**
4. 等待处理完成，查看日志中的进度和结果

### 方式二：命令行（单张图片）

```bash
conda activate cityview
python main.py 图片路径.jpg output/
```

示例：

```bash
python main.py input/full1.jpg output
```

### 方式三：命令行（批量处理）

```bash
conda activate cityview
python batch_run.py input_folder/ output/ --workers 1
```

> `--workers 1` 表示每次处理 1 张图（GPU 是瓶颈，并行不会更快）。

---

## 输出说明

每张全景图会生成 **3 个视角**（left / front / right），每个视角 **25 个文件**：

| 类别 | 文件数 | 说明 |
|------|--------|------|
| 语义分割图 | 若干 | 各语义类别的掩码和彩色叠加图 |
| 深度估计图 | 若干 | 深度热力图、度量深度彩色图 |
| 前中背景分层 | 3 | 前景 / 中景 / 背景掩码 |
| 组合分层图 | 若干 | 前中背景 × 原图 / 语义 / 深度的叠加 |
| 天空掩码 | 1 | 白色 = 天空，黑色 = 非天空 |
| 开放度图 | 1 | 场景开放度热力图 |
| 原始数据 | 2 | `depth_metric.npy`（深度矩阵）+ `metadata.json` |

输出目录结构示例：

```
output/
├── full1_left/       # 25 个文件
├── full1_front/      # 25 个文件
└── full1_right/      # 25 个文件
```

---

## 常见问题

### `torch.cuda.is_available()` 返回 False

- 确认电脑有 NVIDIA 独立显卡
- 更新显卡驱动：访问 https://www.nvidia.cn/drivers/ 下载最新驱动
- 确认安装的是 GPU 版 PyTorch（第 3 步的命令带 `--index-url ...cu124`）

### 内存不足 / 显存不足

- 关闭其他占用显存的程序（如游戏、其他 AI 工具）
- 每次只处理一张图片

### 处理速度慢

- 正常速度：每张全景图约 30 - 40 秒（取决于显卡性能）
- RTX 3070 Laptop 实测约 35 秒/张

### 安装依赖报错

- 确认已激活 cityview 环境：`conda activate cityview`
- 尝试升级 pip：`python -m pip install --upgrade pip`
- 如果 `lang-segment-anything` 安装失败，需要先安装 git：https://git-scm.com/download/win

---

## 项目结构

```
AI_City_View/
├── app.py                          # Gradio 图形界面
├── main.py                         # 单图处理入口
├── batch_run.py                    # 批量处理脚本
├── requirements.txt                # Python 依赖
├── Semantic_configuration.json     # 语义类别配置
└── pipeline/                       # 处理流水线
    ├── stage1_preprocess.py        # 全景图裁剪
    ├── stage2_ai_inference.py      # AI 推理（深度 + 语义）
    ├── stage3_postprocess.py       # 语义后处理
    ├── stage4_intelligent_fmb.py   # 前中背景分层
    ├── stage5_openness.py          # 开放度计算
    ├── stage6_generate_images.py   # 生成分析图片
    └── stage7_save_outputs.py      # 保存输出
```
