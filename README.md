# AI City View — 全景图分析工具

把一张全景图自动切成 3 个视角（左 / 中 / 右），每个视角生成 25 张分析图片（语义分割、深度估计、前中背景分层等）。

---

## 你的电脑需要什么？

- **Windows 10 或 11**
- **NVIDIA 独立显卡**，显存至少 8 GB（比如 RTX 3060、3070、4060、4070 等）
  - 不确定自己显卡型号？按 `Ctrl+Shift+Esc` 打开任务管理器 → 点"性能"→ 左边找"GPU"就能看到
- **内存** 16 GB 以上
- **硬盘空间** 10 GB 以上

> 如果你的电脑没有 NVIDIA 显卡，或者显存不够，请直接跳到最后的 **"没有显卡？用 Google Colab"** 部分。

---

## 安装教程（照着做就行）

整个过程大约 20-30 分钟，取决于网速。

---

### 第 1 步：安装 Miniconda（一个 Python 管理工具）

1. 打开浏览器，访问：https://docs.conda.io/en/latest/miniconda.html
2. 找到 **Windows** 那一栏，点击 **Miniconda3 Windows 64-bit** 下载
3. 双击下载好的文件，一路点 **Next**
   - 中间会看到一个 **"Add Miniconda3 to my PATH"** 的勾选框，**建议勾上**
4. 点 **Install**，等安装完成，点 **Finish**

**验证是否安装成功：**

5. 点电脑左下角的 **开始菜单**（Windows 图标）
6. 搜索 **"Anaconda Prompt"**，点击打开（会弹出一个黑色窗口）
7. 在黑色窗口里输入下面这行，然后按回车：

```
conda --version
```

如果看到类似 `conda 24.x.x` 这样的数字，就说明安装成功了。

> **接下来所有步骤都在这个"Anaconda Prompt"黑色窗口里操作。不要关掉它。**

---

### 第 2 步：创建专用环境

在黑色窗口里输入下面这行，按回车：

```
conda create -n cityview python=3.10 -y
```

等它跑完（大约 1-2 分钟），然后输入：

```
conda activate cityview
```

你会看到窗口最左边从 `(base)` 变成了 `(cityview)`，说明切换成功了。

---

### 第 3 步：安装 PyTorch（AI 计算引擎）

复制下面这整行，粘贴到窗口里，按回车：

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> 这一步会下载约 2-3 GB 文件，耐心等待。

安装完后，输入下面这行来验证显卡是否能用：

```
python -c "import torch; print(torch.cuda.is_available())"
```

- 如果显示 **True** → 成功，继续下一步
- 如果显示 **False** → 显卡驱动可能需要更新，见下方"常见问题"

---

### 第 4 步：进入项目文件夹

假设你把收到的 `AI_City_View` 文件夹放在了桌面上，输入：

```
cd %USERPROFILE%\Desktop\AI_City_View
```

> 如果你放在了其他位置，把上面的路径改成实际路径即可。比如放在 D 盘根目录就输入 `cd D:\AI_City_View`。

---

### 第 5 步：安装项目依赖

```
pip install -r requirements.txt
```

> 这一步也会下载不少东西，大约 5-10 分钟。

---

### 第 6 步：安装加速组件（推荐）

```
pip install xformers
```

---

### 安装完成！

到这里所有安装就做完了。以后每次使用只需要重复下面的"启动"步骤就行。

---

## 如何使用

### 每次使用前

1. 打开 **Anaconda Prompt**（开始菜单搜索）
2. 输入：
```
conda activate cityview
cd %USERPROFILE%\Desktop\AI_City_View
python app.py
```

3. 等几秒钟，窗口里会出现类似这样的文字：
```
Running on local URL: http://0.0.0.0:7860
```

4. 打开浏览器（Chrome、Edge 都行），在地址栏输入：**http://localhost:7860** 然后回车

### 处理图片

1. 在打开的网页里，**拖拽全景图到上传区域**（也可以点击选择文件，支持一次选多张）
2. 输出目录保持默认 `output` 就行
3. 点击 **开始处理**
4. 等待处理完成（每张大约 30-40 秒），日志区域会显示进度
5. 处理完后，到 `AI_City_View\output` 文件夹里查看结果

### 结果在哪？

处理完后打开 `AI_City_View` 文件夹下的 `output` 文件夹，里面每张全景图会生成 3 个子文件夹：

```
output/
├── 图片名_left/       ← 左视角，25 个文件
├── 图片名_front/      ← 中视角，25 个文件
└── 图片名_right/      ← 右视角，25 个文件
```

---

## 常见问题

### 找不到 Anaconda Prompt？

安装 Miniconda 之后，点开始菜单搜索 "Anaconda Prompt"。如果搜不到，试试重启电脑。

### `torch.cuda.is_available()` 显示 False？

说明 PyTorch 没检测到你的显卡。解决方法：

1. 确认你的电脑有 NVIDIA 独立显卡（任务管理器 → 性能 → GPU 查看）
2. 更新显卡驱动：打开 https://www.nvidia.cn/drivers/ ，选择你的显卡型号，下载安装最新驱动
3. 重新打开 Anaconda Prompt，激活环境后再试一次

### 安装依赖报错？

- 确认窗口左边显示的是 `(cityview)` 而不是 `(base)`。如果是 `(base)`，先输入 `conda activate cityview`
- 试试升级 pip：`python -m pip install --upgrade pip`，然后重新运行安装命令

### 处理速度很慢？

这是正常的。每张全景图大约需要 30-40 秒（取决于显卡性能）。

### 黑色窗口里出现一堆红色报错？

把报错内容截图发给开发者。

---

## 没有显卡？用 Google Colab（免费）

如果你的电脑没有 NVIDIA 显卡，或者显存不够 8 GB，可以用 Google 提供的免费云端 GPU：

1. 打开浏览器，访问 https://colab.research.google.com （需要 Google 账号）
2. 点左上角 **文件 → 新建笔记本**
3. 点菜单栏 **运行时 → 更改运行时类型** → 选 **T4 GPU** → 点保存
4. 在页面中间的输入框里，**逐段复制粘贴**下面的代码，每段粘贴后按 **Shift+回车** 运行：

**第一段：安装环境**
```python
!git clone https://github.com/ZhenGtai123/AI_City_View.git
%cd AI_City_View
!pip install -r requirements.txt
!pip install xformers
```

**第二段：上传图片并处理**
```python
from google.colab import files
uploaded = files.upload()  # 会弹出选择文件的按钮，点击后选择你的全景图

from main import process_panorama
for name in uploaded:
    process_panorama(name, "output")
```

**第三段：下载结果**
```python
!zip -r output.zip output/
files.download("output.zip")  # 浏览器会自动下载一个 zip 文件
```

5. 解压下载的 `output.zip` 即可查看结果
