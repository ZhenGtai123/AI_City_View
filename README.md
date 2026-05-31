# AI City View

Urban street-view panorama analysis API. Feed it a single panorama, it auto-crops to three 90° views (left / front / right) and runs semantic segmentation, depth estimation, and foreground/middleground/background layering on each — producing 25 analysis images per view.

Exposed via FastAPI as an HTTP service so [SceneRx](../greensvc) and other platforms can call it.

## Quickstart (Docker)

Requires an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
git clone https://github.com/ZhenGtai123/AI_City_View.git
cd AI_City_View
docker compose up -d
```

Open **http://localhost:8000/docs** for the interactive API console, or `curl http://localhost:8000/health` to verify.

- First build: ~10 min (installs PyTorch CUDA + Depth Anything + xformers).
- First request: ~2 min extra (downloads OneFormer ~1.2 GB + DA3 ~1.4 GB into the `hf_cache` volume; cached after that).
- Switch depth model: `VISION_DEPTH_MODEL=DA3NESTED-GIANT-LARGE-1.1 docker compose up -d`.

To wire this into SceneRx, open the SceneRx Settings page and set `VISION_API_URL=http://127.0.0.1:8000` (or `http://host.docker.internal:8000` when both stacks run on the same Docker host).

---

## System requirements

| Item   | Minimum |
|--------|---------|
| GPU    | NVIDIA, ≥ 8 GB VRAM (RTX 3060 / 4060 or newer) |
| Memory | 16 GB |
| Python | 3.10 (only for the local-Python development mode) |

---

## Running

### FastAPI Server (local Python, development only)

Use this only when you're editing model code and don't want to rebuild the image. Follow the **Installation** section below to set up the conda env first.

```bash
conda activate cityview
python server.py
```

- API: **http://localhost:8000**
- Docs: **http://localhost:8000/docs**
- First start preloads models (~30–60 s).

Custom port:

```bash
PORT=8001 python server.py
```

> Wire into SceneRx by setting `VISION_API_URL=http://127.0.0.1:8000` in its Settings page.

### Single-image CLI

```bash
python main.py <image_path> <output_dir>
python main.py full1.jpg output
```

### Local batch processing

```bash
python batch_run.py /data/input /data/output --workers=2
```

### Cloud batch processing (Azure Blob → GPU VM → GCS)

Designed for 500 K+ image runs. Supports resume-after-interrupt, graceful Spot preemption, and parallel download/process/upload stages.

```bash
# Extra deps
pip install azure-storage-blob google-cloud-storage

# Run
python cloud_batch_run.py \
  --azure-sas-url "https://account.blob.core.windows.net/container?sv=..." \
  --gcs-bucket my-output-bucket \
  --workers=4 \
  --gpu-concurrency=2
```

---

## Installation (local Python, development only)

Skip this entirely if you're using Docker — the image bakes all of these steps.

```bash
git clone https://github.com/ZhenGtai123/AI_City_View.git
cd AI_City_View

conda create -n cityview python=3.10 -y
conda activate cityview

# 1. Project deps
pip install -r requirements.txt

# 2. Depth Anything V3 (from GitHub)
pip install git+https://github.com/ByteDance-Seed/depth-anything-3.git

# 3. PyTorch CUDA build (must come BEFORE xformers; steps 1-2 install the CPU
#    wheel so we force-overwrite here)
pip install --force-reinstall --no-deps torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

# 4. Verify GPU (must print True before continuing)
python -c "import torch; print(torch.cuda.is_available())"

# 5. Optional accelerator (install AFTER step 3 so versions match)
pip install xformers --index-url https://download.pytorch.org/whl/cu124
```

> Steps 1–2 pull a CPU build of PyTorch; step 3 overwrites it with the CUDA build. xformers must come last. If `torch.cuda.is_available()` returns `False`, re-run step 3.

---

## API reference

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/analyze`                       | Analyze one image, return hex-encoded outputs + stats |
| `POST` | `/analyze/panorama`              | Panorama mode: auto-crop 3 views and analyze each |
| `GET`  | `/health`                        | Health check (GPU status, models loaded) |
| `GET`  | `/config`                        | Returns the current `Semantic_configuration.json` |
| `GET`  | `/outputs/{job_id}/download`     | Download a single job's outputs as a ZIP |
| `GET`  | `/outputs/{job_id}/{filename}`   | Download one file from a job |

### `POST /analyze` request

`multipart/form-data`:

- `file` — image file
- `request_data` — JSON string:

```json
{
  "semantic_classes": ["Sky", "Trees", "Building", "Road"],
  "semantic_countability": [0, 1, 0, 0],
  "openness_list": [1, 0, 0, 0],
  "enable_hole_filling": true,
  "image_id": "optional_custom_id"
}
```

### Response

```json
{
  "status": "success",
  "job_id": "img_a1b2c3d4_1708900000",
  "images": {
    "semantic_map":     "<hex-encoded PNG>",
    "depth_map":        "<hex-encoded PNG>",
    "openness_map":     "...",
    "fmb_map":          "...",
    "foreground_map":   "...",
    "middleground_map": "...",
    "background_map":   "...",
    "sky_mask":         "...",
    "semantic_raw":     "..."
  },
  "class_statistics": {
    "sky":  { "pixel_count": 50000, "percentage": 25.5 },
    "tree": { "pixel_count": 30000, "percentage": 15.3 }
  },
  "fmb_statistics": { ... },
  "download_url": "/outputs/img_a1b2c3d4_1708900000/download",
  "processing_time": 6.2
}
```

---

## Pipeline

```
panorama in
    │
    ▼
Stage 1: preprocess  ── equirectangular crop → left / front / right (90° FOV)
    │
    ▼
Stage 2: AI inference ── OneFormer (ADE20K-150 segmentation) + Depth Anything V3 (metric depth)
    │
    ▼
Stage 3: postprocess ── semantic map cleanup, denoising
    │
    ▼
Stage 4: FMB layering ─ foreground (0-10 m) / middleground (10-50 m) / background (>50 m) / sky
    │
    ▼
Stage 5: openness    ── semantic-class-aware spatial openness map
    │
    ▼
Stage 6: render      ── 23 analysis images
    │
    ▼
Stage 7: save        ── 23 PNG + sky_mask + semantic_raw + metadata.json + depth_metric.npy
```

---

## Output layout

Each panorama produces three view folders; each view folder contains 25 files:

```
output/
├── {image_name}_left/
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
├── {image_name}_front/
└── {image_name}_right/
```

---

## Configuration

### Depth estimation model

Picked at startup via `VISION_DEPTH_MODEL`, or change the default in `server.py` → `get_default_config()`.

| Model | Params | Output | VRAM |
|-------|--------|--------|------|
| `DA3METRIC-LARGE` (default) | 0.35B | canonical → metric | 8 GB |
| `DA3NESTED-GIANT-LARGE-1.1` | 1.4B  | native metric + sky detection | 16 GB+ |
| `DA3MONO-LARGE`             | 0.35B | relative depth (no metric) | 8 GB |

### `Semantic_configuration.json`

Per-class definition:

```json
{
  "name": "Trees",
  "color": "#00FF00",
  "countable": 1,
  "openness": 0
}
```

---

## Project layout

```
AI_City_View/
├── server.py              # FastAPI entry point
├── main.py                # panorama pipeline core
├── batch_run.py           # local batch script
├── cloud_batch_run.py     # cloud batch (Azure Blob → GCS)
├── pipeline/
│   ├── stage1_preprocess.py        # equirectangular crop
│   ├── stage2_ai_inference.py      # OneFormer + DA3 (GPU)
│   ├── stage3_postprocess.py       # semantic map cleanup
│   ├── stage4_intelligent_fmb.py   # smart FMB layering
│   ├── stage4_depth_layering.py    # depth layering (fallback)
│   ├── stage5_openness.py          # openness computation
│   ├── stage6_generate_images.py   # render analysis images
│   ├── stage7_save_outputs.py      # save to disk
│   └── gpu_utils.py                # GPU helpers
├── Semantic_configuration.json     # class definitions
├── requirements.txt                # Python deps
└── output/                         # default output dir
```
