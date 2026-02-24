"""
AI_City_View Vision API Server
FastAPI server exposing the computer vision pipeline (semantic segmentation +
depth estimation + FMB layering) over HTTP.

Endpoints:
    POST /analyze          – Single image analysis → JSON (hex-encoded images)
    POST /analyze/panorama – Panorama mode: crop 3 views, analyze each
    GET  /health           – Health check with GPU / model info
    GET  /config           – Returns Semantic_configuration.json
    GET  /outputs/{job_id}/download   – ZIP download of all output images
    GET  /outputs/{job_id}/{filename} – Individual file download

Usage:
    pip install fastapi uvicorn python-multipart
    python -m uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import shutil
import sys
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so pipeline imports work
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.stage1_preprocess import crop_panorama_three_views
from pipeline.stage2_ai_inference import stage2_ai_inference, clear_model_cache
from pipeline.stage3_postprocess import stage3_postprocess
from pipeline.stage4_intelligent_fmb import stage4_intelligent_fmb, stage4_metric_fmb
from pipeline.stage4_depth_layering import stage4_depth_layering
from pipeline.stage5_openness import stage5_openness
from pipeline.stage6_generate_images import stage6_generate_images
from pipeline.stage7_save_outputs import stage7_save_outputs

logger = logging.getLogger("vision_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUTS_DIR = _PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
OUTPUT_RETENTION_SECONDS = 3600  # 1 hour
CLEANUP_INTERVAL_SECONDS = 600  # 10 minutes

# The 9 key images to hex-encode in JSON responses
HEX_IMAGE_KEYS = [
    "semantic_map",
    "depth_map",
    "openness_map",
    "fmb_map",
    "foreground_map",
    "middleground_map",
    "background_map",
    "sky_mask",
    "semantic_raw",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hex_to_bgr(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


def _generate_job_id(prefix: str = "img") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}_{int(time.time())}"


def _encode_image_hex(img: np.ndarray) -> str:
    """Encode a numpy image (BGR/BGRA/grayscale) to PNG bytes then hex string."""
    if img is None:
        return ""
    # Ensure uint8
    if img.dtype != np.uint8:
        if img.dtype == bool:
            img = img.astype(np.uint8) * 255
        else:
            img = img.astype(np.uint8)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return ""
    return buf.tobytes().hex()


def _load_semantic_config() -> list[dict]:
    """Load Semantic_configuration.json from project root."""
    config_path = _PROJECT_ROOT / "Semantic_configuration.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_default_config() -> Dict[str, Any]:
    """Build the default pipeline config (mirrors main.py logic)."""
    semantic_config = _load_semantic_config()
    for item in semantic_config:
        if "color" in item and "bgr" not in item:
            item["bgr"] = _hex_to_bgr(item["color"])

    return {
        "split_method": "percentile",
        "semantic_items": semantic_config,
        "enable_semantic": True,
        "depth_backend": "v3",
        "depth_model_id_v3": "depth-anything/DA3METRIC-LARGE",
        "depth_focal_length": 300,
        "depth_process_res": 672,
        "depth_invert_v3": False,
    }


def build_config_from_request(
    request_data: Dict[str, Any],
    base_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Overlay GreenSVC request fields onto the default pipeline config.

    GreenSVC sends:
        semantic_classes:      ["Sky", "Trees", ...]
        semantic_countability: [0, 1, ...]
        openness_list:         [0, 1, ...]
        semantic_colors:       {"0": [r,g,b], "1": [r,g,b], ...}
        enable_hole_filling:   bool
    """
    config = dict(base_config)

    classes = request_data.get("semantic_classes")
    countability = request_data.get("semantic_countability")
    openness = request_data.get("openness_list")
    colors = request_data.get("semantic_colors")

    if classes:
        # Build semantic_items list expected by the pipeline
        items = []
        for i, name in enumerate(classes):
            item: Dict[str, Any] = {"name": name}
            if countability and i < len(countability):
                item["countable"] = countability[i]
            if openness and i < len(openness):
                item["openness"] = openness[i]
            # Color from request (RGB list) → hex + BGR
            if colors and str(i) in colors:
                rgb = colors[str(i)]
                item["color"] = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
                item["bgr"] = (rgb[2], rgb[1], rgb[0])
            items.append(item)
        config["semantic_items"] = items

    # Forward hole filling / median blur toggles
    if "enable_hole_filling" in request_data:
        config["enable_hole_filling"] = bool(request_data["enable_hole_filling"])
    if "enable_median_blur" in request_data:
        config["enable_median_blur"] = bool(request_data["enable_median_blur"])

    return config


def _compute_class_statistics(
    semantic_map: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute per-class pixel counts and percentages from the semantic map."""
    total_pixels = int(semantic_map.size)
    unique, counts = np.unique(semantic_map, return_counts=True)

    items = config.get("semantic_items", [])
    stats: Dict[str, Any] = {}
    for cls_id, count in zip(unique, counts):
        cls_id = int(cls_id)
        if cls_id < len(items):
            name = items[cls_id].get("name", f"class_{cls_id}")
        else:
            name = f"class_{cls_id}"
        stats[name] = {
            "pixel_count": int(count),
            "percentage": round(float(count) / total_pixels * 100, 2),
        }
    return stats


def _run_pipeline(
    image: np.ndarray,
    config: Dict[str, Any],
    job_id: str,
) -> Dict[str, Any]:
    """Run stages 2-7 synchronously. Called from thread pool."""
    t0 = time.time()
    h, w = image.shape[:2]
    original_copy = image.copy()

    # Stage 2: AI inference
    logger.info("[%s] Stage 2: AI inference ...", job_id)
    stage2 = stage2_ai_inference(image, config)
    depth_map = stage2["depth_map"]
    semantic_map = stage2["semantic_map"]
    depth_metric = stage2.get("depth_metric")
    sky_mask = stage2.get("sky_mask")

    # Stage 3: Post-processing
    logger.info("[%s] Stage 3: Post-processing ...", job_id)
    stage3 = stage3_postprocess(semantic_map, config)
    semantic_processed = stage3["semantic_map_processed"]

    # Stage 4: FMB layering
    logger.info("[%s] Stage 4: FMB layering ...", job_id)
    if depth_metric is not None:
        stage4 = stage4_metric_fmb(
            depth_metric, config,
            semantic_map=semantic_map,
            sky_mask=sky_mask,
        )
    else:
        fmb_method = str(config.get("fmb_method", "intelligent")).lower()
        if fmb_method == "intelligent":
            stage4 = stage4_intelligent_fmb(depth_map, config, semantic_map=semantic_map)
        else:
            stage4 = stage4_depth_layering(depth_map, config, semantic_map=semantic_map)

    fg_mask = stage4["foreground_mask"]
    mg_mask = stage4["middleground_mask"]
    bg_mask = stage4["background_mask"]

    # Stage 5: Openness
    logger.info("[%s] Stage 5: Openness ...", job_id)
    stage5 = stage5_openness(semantic_processed, config)

    # Stage 6: Generate images
    logger.info("[%s] Stage 6: Generate images ...", job_id)
    stage6 = stage6_generate_images(
        original_copy, semantic_processed, depth_map,
        stage5["openness_map"],
        fg_mask, mg_mask, bg_mask,
        config,
        depth_metric=depth_metric,
    )

    output_images = stage6["images"]
    # Attach sky_mask and raw semantic
    if sky_mask is not None:
        output_images["sky_mask"] = sky_mask.astype(np.uint8) * 255
    output_images["semantic_raw"] = semantic_map

    # Build metadata
    metadata = {
        "basename": job_id,
        "width": w,
        "height": h,
    }
    if "depth_stats" in stage4:
        metadata["depth_stats"] = stage4["depth_stats"]
    if "depth_thresholds" in stage4:
        metadata["fmb_thresholds"] = stage4["depth_thresholds"]
    if "layer_stats" in stage4:
        metadata["fmb_layer_stats"] = stage4["layer_stats"]

    # Stage 7: Save outputs to disk for download
    logger.info("[%s] Stage 7: Save outputs ...", job_id)
    output_dir = OUTPUTS_DIR / job_id
    stage7 = stage7_save_outputs(
        output_images, str(output_dir), job_id, metadata, depth_metric=depth_metric
    )

    # Hex-encode the 9 key images for JSON response
    hex_images: Dict[str, str] = {}
    for key in HEX_IMAGE_KEYS:
        if key in output_images:
            hex_images[key] = _encode_image_hex(output_images[key])

    # FMB statistics
    fmb_stats = stage4.get("layer_stats", {})

    # Class statistics
    class_stats = _compute_class_statistics(semantic_processed, config)

    elapsed = time.time() - t0
    logger.info("[%s] Pipeline complete in %.1fs", job_id, elapsed)

    return {
        "status": "success",
        "job_id": job_id,
        "images": hex_images,
        "detected_classes": len(class_stats),
        "total_classes": len(config.get("semantic_items", [])),
        "class_statistics": class_stats,
        "fmb_statistics": fmb_stats,
        "instances": [],
        "segmentation_mode": "single_label",
        "hole_filling_enabled": bool(config.get("enable_hole_filling", True)),
        "download_url": f"/outputs/{job_id}/download",
        "processing_time": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Background cleanup task
# ---------------------------------------------------------------------------

async def _cleanup_old_outputs():
    """Periodically remove output directories older than retention period."""
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        try:
            now = time.time()
            if not OUTPUTS_DIR.exists():
                continue
            for entry in OUTPUTS_DIR.iterdir():
                if entry.is_dir():
                    age = now - entry.stat().st_mtime
                    if age > OUTPUT_RETENTION_SECONDS:
                        shutil.rmtree(entry, ignore_errors=True)
                        logger.info("Cleaned up old output: %s (age=%.0fs)", entry.name, age)
        except Exception as e:
            logger.warning("Cleanup error: %s", e)


# ---------------------------------------------------------------------------
# Lifespan: preload models at startup
# ---------------------------------------------------------------------------

_default_config: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _default_config
    _default_config = get_default_config()
    logger.info("Default config loaded (%d semantic items)", len(_default_config.get("semantic_items", [])))

    # Preload models by running a tiny dummy inference in the thread pool
    logger.info("Preloading models (this may take 30-60s on first run) ...")
    try:
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, stage2_ai_inference, dummy, _default_config)
        logger.info("Models preloaded successfully")
    except Exception as e:
        logger.warning("Model preload failed (will retry on first request): %s", e)

    # Start cleanup task
    cleanup_task = asyncio.create_task(_cleanup_old_outputs())

    yield

    # Shutdown
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    clear_model_cache()
    logger.info("Server shutdown, GPU memory released")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI City View - Vision API",
    version="1.0.0",
    description="Computer vision pipeline: semantic segmentation + depth estimation + FMB layering",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    import torch

    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    gpu_memory = None
    if gpu_available:
        total = torch.cuda.get_device_properties(0).total_mem
        gpu_memory = f"{total / (1024**3):.1f} GB"

    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "gpu_memory": gpu_memory,
        "models_loaded": True,
        "semantic_classes": len(_default_config.get("semantic_items", [])),
        "depth_backend": _default_config.get("depth_backend", "unknown"),
        "depth_model": _default_config.get("depth_model_id_v3", "unknown"),
    }


# ---------------------------------------------------------------------------
# GET /config
# ---------------------------------------------------------------------------

@app.get("/config")
async def get_config():
    """Return the semantic configuration."""
    return _load_semantic_config()


# ---------------------------------------------------------------------------
# POST /analyze
# ---------------------------------------------------------------------------

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    request_data: str = Form(default="{}"),
):
    """Analyze a single image. Returns JSON with hex-encoded images + statistics."""
    # Parse request data
    try:
        req = json.loads(request_data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid request_data JSON: {e}")

    # Read image
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    arr = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # Build config
    config = build_config_from_request(req, _default_config)

    # Generate job ID
    image_id = req.get("image_id", "")
    job_id = image_id if image_id else _generate_job_id()

    # Run pipeline in thread pool to avoid blocking the event loop
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, _run_pipeline, image, config, job_id)
    except Exception as e:
        logger.error("Pipeline error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# POST /analyze/panorama
# ---------------------------------------------------------------------------

@app.post("/analyze/panorama")
async def analyze_panorama(
    file: UploadFile = File(...),
    request_data: str = Form(default="{}"),
):
    """Analyze a panorama image: crop into 3 views (left/front/right), analyze each."""
    try:
        req = json.loads(request_data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid request_data JSON: {e}")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    arr = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    config = build_config_from_request(req, _default_config)

    # Crop panorama into 3 views
    views = crop_panorama_three_views(image)
    base_id = req.get("image_id", "") or _generate_job_id("pano")

    # Process each view sequentially (GPU lock inside stage2 makes parallelism
    # less effective and risks OOM; sequential is safer for single-GPU setups)
    loop = asyncio.get_running_loop()
    results: Dict[str, Any] = {}

    for view_name, view_img in views.items():
        job_id = f"{base_id}_{view_name}"
        try:
            result = await loop.run_in_executor(None, _run_pipeline, view_img, config, job_id)
            results[view_name] = result
        except Exception as e:
            logger.error("Panorama view %s error: %s", view_name, e, exc_info=True)
            results[view_name] = {"status": "error", "error": str(e)}

    return JSONResponse(content={
        "status": "success",
        "panorama_id": base_id,
        "views": results,
    })


# ---------------------------------------------------------------------------
# GET /outputs/{job_id}/download
# ---------------------------------------------------------------------------

@app.get("/outputs/{job_id}/download")
async def download_outputs(job_id: str):
    """Download all output images for a job as a ZIP archive."""
    output_dir = OUTPUTS_DIR / job_id
    if not output_dir.exists() or not output_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Job outputs not found: {job_id}")

    # Build ZIP in memory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in sorted(output_dir.iterdir()):
            if fpath.is_file():
                zf.write(fpath, fpath.name)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.zip"'},
    )


# ---------------------------------------------------------------------------
# GET /outputs/{job_id}/{filename}
# ---------------------------------------------------------------------------

@app.get("/outputs/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    """Download an individual output file."""
    filepath = OUTPUTS_DIR / job_id / filename
    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {job_id}/{filename}")

    # Prevent path traversal
    try:
        filepath.resolve().relative_to(OUTPUTS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    return FileResponse(filepath, filename=filename)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    logger.info("Starting Vision API server on port %d ...", port)
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        log_level="info",
    )
