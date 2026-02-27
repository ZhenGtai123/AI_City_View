"""
阶段2: AI模型推理
功能: 语义分割 (SAM 2.1 + LangSAM) + 深度估计 (Depth Anything V2)
精度优先方案
"""

from __future__ import annotations

import time
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import threading

# 延迟导入，避免未安装时直接报错
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch未安装，AI推理功能将不可用")


# 全局GPU推理信号量: 控制同时进行GPU推理的线程数
# Semaphore(1) 等价于 Lock(), Semaphore(2) 允许2个并发推理
_GPU_INFERENCE_LOCK = threading.Semaphore(1)

# 模型加载锁: 防止多线程同时加载模型（特别是大模型会导致OOM）
_MODEL_LOAD_LOCK = threading.Lock()


def set_gpu_concurrency(n: int):
    """设置GPU并发推理数 (默认1, L4/24GB建议2-3)"""
    global _GPU_INFERENCE_LOCK
    _GPU_INFERENCE_LOCK = threading.Semaphore(max(1, n))


def stage2_ai_inference(image: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    阶段2: AI模型推理

    参数:
        image: np.ndarray - 来自阶段1的 original (H, W, 3) BGR格式
        config: dict - 配置参数

    返回:
        dict - 包含以下键:
            - semantic_map: np.ndarray - 语义分割图 (H, W) uint8 (若禁用则全0)
            - depth_map: np.ndarray - 深度图 (H, W) uint8
    """
    H, W = image.shape[:2]

    # 语义分割（可禁用，DA3 NESTED 已内置天空分割）
    if config.get('enable_semantic', True):
        semantic_map = _semantic_segmentation(image, config)
    else:
        semantic_map = np.zeros((H, W), dtype=np.uint8)

    # 深度估计 (返回 uint8 可视化 + float32 原始米数 + sky_mask)
    depth_result = _depth_estimation_with_metric(image, config)
    depth_map = depth_result['depth_map']
    depth_metric = depth_result['depth_metric']
    sky_mask = depth_result['sky_mask']

    # 天空处理:
    # - DA3NESTED: depth提供sky_mask → 修正semantic_map
    # - DA3METRIC/MONO: 无depth sky_mask → 从OneFormer semantic_map==2推导
    if sky_mask is not None and sky_mask.any():
        # DA3NESTED的天空掩码修正OneFormer的语义分割 (ADE20K sky=2)
        semantic_map[sky_mask] = 2
    elif depth_metric is not None:
        # DA3METRIC: 从OneFormer推导天空掩码, 设置depth_metric为inf
        semantic_sky = (semantic_map == 2)
        if semantic_sky.any():
            sky_mask = semantic_sky
            depth_metric[sky_mask] = np.inf
            sky_pct = sky_mask.sum() / sky_mask.size * 100
            print(f"  [SKY] Sky mask (from OneFormer): {sky_pct:.1f}% pixels")

    # 天空缝隙修补: 树缝/建筑缝隙间的天空 OneFormer 容易漏掉
    # 策略: 深度 > p95 (非天空) + 靠近已知天空区域 (膨胀掩码) → 补充为天空
    if (config.get('sky_refine_gaps', True)
            and sky_mask is not None and sky_mask.any()
            and depth_metric is not None):
        sky_mask, gap_count = _refine_sky_gaps(
            depth_metric, sky_mask,
            image_bgr=image,
            semantic_map=semantic_map,
        )
        if gap_count > 0:
            semantic_map[sky_mask] = 2
            depth_metric[sky_mask] = np.inf
            print(f"  [SKY] Sky gap refinement: +{gap_count} pixels "
                  f"(total {sky_mask.sum()/sky_mask.size*100:.1f}%)")

    return {
        'semantic_map': semantic_map,
        'depth_map': depth_map,
        'depth_metric': depth_metric,   # float32, 单位: 米
        'sky_mask': sky_mask,           # bool (H,W) or None
    }


def _semantic_segmentation(image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """
    语义分割 - 使用 SAM 2.1 + LangSAM
    
    参数:
        image: (H, W, 3) BGR uint8
        config: dict - 配置参数
    
    返回:
        semantic_map: (H, W) uint8, 值范围 [0, N]
            - 0: 背景/未分类
            - 1-N: 语义类别ID (N = len(classes))
    """
    H, W = image.shape[:2]
    backend = str(config.get('semantic_backend', 'oneformer_ade20k')).strip().lower()

    # 初始化语义分割图
    semantic_map = np.zeros((H, W), dtype=np.uint8)

    if backend.startswith('oneformer'):
        try:
            profile = bool(config.get('profile', False))
            t0 = time.perf_counter()

            model, processor = get_semantic_model(config)
            if model is None or processor is None:
                print("  ⚠️  警告: OneFormer 未就绪，语义分割使用占位实现")
                return semantic_map

            _maybe_apply_semantic_items_mapping_for_ade20k(model, config)

            if profile:
                print(f"  ⏱️  OneFormer ready: {time.perf_counter() - t0:.3f}s")

            device = _get_torch_device(config)
            use_fp16 = bool(config.get('semantic_use_fp16', True))

            # OpenCV(BGR)->RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # OneFormer semantic segmentation
            t1 = time.perf_counter()
            with _GPU_INFERENCE_LOCK:
                with torch.inference_mode():
                    inputs = processor(images=rgb, task_inputs=["semantic"], return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    if use_fp16 and device.type == 'cuda':
                        inputs = {k: v.half() if torch.is_floating_point(v) else v for k, v in inputs.items()}
                    outputs = model(**inputs)
                    pred = processor.post_process_semantic_segmentation(outputs, target_sizes=[(H, W)])[0]
                semantic_map = pred.detach().to('cpu').numpy().astype(np.uint8)
            if profile:
                print(f"  ⏱️  OneFormer inference+post: {time.perf_counter() - t1:.3f}s")
            return semantic_map

        except Exception as e:
            print(f"  ❌ OneFormer 语义分割出错: {e}")
            import traceback
            traceback.print_exc()
            return semantic_map

    # 兼容旧实现：LangSAM 文本提示分割
    classes = list(config.get('classes', []) or [])
    semantic_cfg = config.get('semantic', {}) or {}

    if len(classes) == 0:
        print("  警告: 未指定类别，返回全0语义图")
        return semantic_map

    try:
        profile = bool(config.get('profile', False))
        t0 = time.perf_counter()

        model, _processor = get_semantic_model({**config, 'semantic_backend': 'langsam'})
        if model is None:
            print("  ⚠️  警告: LangSAM 未就绪，语义分割使用占位实现")
            _generate_placeholder_semantic_map(semantic_map, classes, H, W)
            return semantic_map

        if profile:
            print(f"  ⏱️  LangSAM ready: {time.perf_counter() - t0:.3f}s")

        # OpenCV(BGR)->RGB PIL
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        for class_id, class_name in enumerate(classes, start=1):
            if profile:
                t_cls = time.perf_counter()
            masks = _langsam_predict_masks(model, pil_img, class_name, semantic_cfg)

            if masks is None:
                continue
            if masks.ndim == 2:
                combined = masks.astype(bool)
            else:
                combined = np.any(masks.astype(bool), axis=0)

            if combined.any():
                semantic_map[combined] = class_id

            if profile:
                print(f"  ⏱️  semantic '{class_name}': {time.perf_counter() - t_cls:.3f}s")

    except Exception as e:
        print(f"  ❌ LangSAM 语义分割出错: {e}")
        import traceback
        traceback.print_exc()
        return semantic_map

    return semantic_map


def _normalize_label(s: str) -> str:
    import re

    s = s.lower().strip()
    # Normalize separators and punctuation.
    s = s.replace('&', ' and ')
    s = re.sub(r"[\[\]\(\)\{\}]+", " ", s)
    s = re.sub(r"[;,:/]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _maybe_apply_semantic_items_mapping_for_ade20k(model, config: Dict[str, Any]) -> None:
    """Build colors/openness for OneFormer ADE20K ids from user-provided semantic_items.

    - OneFormer outputs class ids 0..149 (ADE20K-150)
    - User semantic config contains a small set of human labels with colors/openness.
    This function tries to match those labels to ADE20K id2label and fills:
      - config['classes'] as id-aligned label list (for stats)
      - config['colors'] mapping for matched ids (others handled by stage6 fallback)
      - config['openness_config'] list length max_id+1 with matched ids set
    """
    if config.get('_ade20k_mapped_from_semantic_items'):
        return
    backend = str(config.get('semantic_backend', 'oneformer_ade20k')).strip().lower()
    if not backend.startswith('oneformer'):
        return

    items = config.get('semantic_items', None)
    if not items:
        return

    id2label = getattr(getattr(model, 'config', None), 'id2label', None)
    if not isinstance(id2label, dict) or not id2label:
        return

    # Build id-aligned label list
    max_id = max(int(k) for k in id2label.keys() if str(k).isdigit()) if any(str(k).isdigit() for k in id2label.keys()) else 149
    labels_by_id = []
    for i in range(0, max_id + 1):
        labels_by_id.append(str(id2label.get(i, f"class_{i}")))

    # Build normalized label->id index
    norm_to_id = {}
    for i, lab in enumerate(labels_by_id):
        norm_to_id[_normalize_label(lab)] = i

    # 注意: 不再默认把id=0设为黑色，因为ADE20K中0是wall
    colors = {}
    openness_config = [0] * (max_id + 1)

    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get('name', '')).strip()
        if not name:
            continue
        openness = int(item.get('openness', 0) or 0)
        bgr = item.get('bgr', None)
        if bgr is None:
            continue

        # Try exact normalized match; then try synonyms split by ';'
        candidates = [name] + [p.strip() for p in name.split(';') if p.strip()]
        matched_id = None
        for cand in candidates:
            nid = norm_to_id.get(_normalize_label(cand), None)
            if nid is not None:
                matched_id = int(nid)
                break

        # Fallback: substring match (avoid ambiguous matches)
        if matched_id is None:
            n = _normalize_label(name)
            hits = [i for k, i in norm_to_id.items() if (n == k or n in k or k in n)]
            if len(hits) == 1:
                matched_id = int(hits[0])

        # 注意: ADE20K中class 0是wall，所以不能排除matched_id == 0
        if matched_id is None or matched_id < 0 or matched_id > max_id:
            continue

        colors[matched_id] = tuple(int(x) for x in bgr)
        openness_config[matched_id] = 1 if openness == 1 else 0

    config['classes'] = labels_by_id
    config['colors'] = colors
    config['openness_config'] = openness_config
    config['_ade20k_mapped_from_semantic_items'] = True


def _depth_estimation(image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """
    深度估计 - 支持 Depth Pro, Depth Anything V2/V3 (仅返回 uint8)
    保留向后兼容，内部调用 _depth_estimation_with_metric
    """
    result = _depth_estimation_with_metric(image, config)
    return result['depth_map']


def _depth_estimation_with_metric(image: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度估计 - 返回 uint8 可视化 + float32 度量深度(米) + 天空掩码

    返回:
        dict:
            - depth_map: (H, W) uint8, 0=近, 255=远
            - depth_metric: (H, W) float32, 单位米 (None if not available)
            - sky_mask: (H, W) bool (None if not available)
    """
    H, W = image.shape[:2]
    depth_backend = str(config.get('depth_backend', 'depth_pro')).lower().strip()

    if depth_backend == 'depth_pro':
        return _depth_estimation_depth_pro(image, config)
    elif depth_backend == 'v3':
        return _depth_estimation_v3(image, config)
    else:
        result = _depth_estimation_v2(image, config)
        # V2 没有度量深度和天空掩码
        if isinstance(result, np.ndarray):
            return {'depth_map': result, 'depth_metric': None, 'sky_mask': None}
        return result


def _depth_estimation_depth_pro(image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """
    深度估计 - 使用 Apple Depth Pro
    边缘锐利，度量深度精确，适合街景
    """
    H, W = image.shape[:2]

    try:
        profile = bool(config.get('profile', False))
        t0 = time.perf_counter()

        depth_model, transform = get_depth_model_depth_pro(config)
        if depth_model is None:
            print(f"  ⚠️  警告: Depth Pro 未就绪，回退到 Depth Anything V3")
            return _depth_estimation_v3(image, config)

        if profile:
            print(f"  ⏱️  Depth Pro model ready: {time.perf_counter() - t0:.3f}s")

        # OpenCV(BGR)->RGB, 然后转PIL
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Depth Pro 推理
        t1 = time.perf_counter()
        device = _get_torch_device(config)
        with _GPU_INFERENCE_LOCK:
            with torch.inference_mode():
                # 预处理并移到 GPU
                image_tensor = transform(pil_img)
                if device is not None and device.type == 'cuda':
                    image_tensor = image_tensor.to(device)

                # 推理 - Depth Pro 会自动估计焦距
                prediction = depth_model.infer(image_tensor, f_px=None)
                pred = prediction["depth"]  # 度量深度 (米)
                pred = pred.squeeze().cpu().numpy()

        if profile:
            print(f"  ⏱️  Depth Pro inference: {time.perf_counter() - t1:.3f}s")

        # 将预测结果 resize 回原图大小
        t2 = time.perf_counter()
        if pred.shape != (H, W):
            pred_resized = cv2.resize(pred, (W, H), interpolation=cv2.INTER_CUBIC)
        else:
            pred_resized = pred

        # Depth Pro 输出度量深度: 高值=远景(天空), 低值=近景(地面)
        # 保留原始米数用于 FMB
        depth_metric = pred_resized.astype(np.float32)

        # 可视化用 uint8: 0=近景, 255=远景
        depth_map = _normalize_depth_to_uint8(pred_resized, invert=bool(config.get('depth_invert_depth_pro', False)))

        if profile:
            print(f"  [TIME] Depth Pro postprocess: {time.perf_counter() - t2:.3f}s")
            print(f"  [DEPTH] Depth range: {float(depth_metric.min()):.1f}m - {float(depth_metric.max()):.1f}m")

    except Exception as e:
        print(f"  ❌ Depth Pro 出错: {e}")
        import traceback
        traceback.print_exc()
        print(f"  ⚠️  回退到 Depth Anything V3...")
        return _depth_estimation_v3(image, config)

    return {
        'depth_map': depth_map,
        'depth_metric': depth_metric,
        'sky_mask': None,  # Depth Pro 没有内置天空检测
    }


def _detect_da3_model_type(config: Dict[str, Any]) -> str:
    """检测DA3模型类型: 'nested', 'metric', 'mono'"""
    model_id = str(config.get('depth_model_id_v3', '')).upper()
    if 'NESTED' in model_id:
        return 'nested'
    elif 'METRIC' in model_id:
        return 'metric'
    else:
        return 'mono'


def _depth_estimation_v3(image: np.ndarray, config: Dict[str, Any]):
    """
    深度估计 - 使用 Depth Anything V3
    支持三种模型:
      - DA3NESTED: 真实度量深度(米) + 内置天空检测 (需16GB+ VRAM)
      - DA3METRIC: 规范化深度 → 通过焦距转换为米数 (推荐, 8GB VRAM)
      - DA3MONO: 相对深度 (无米数输出)

    返回:
        dict: depth_map (uint8), depth_metric (float32 米 or None), sky_mask (bool or None)
    """
    H, W = image.shape[:2]
    model_type = _detect_da3_model_type(config)

    try:
        profile = bool(config.get('profile', False))
        t0 = time.perf_counter()

        depth_model = get_depth_model_v3(config)
        if depth_model is None:
            print(f"  ⚠️  警告: Depth Anything V3 未就绪，回退到 V2")
            v2_result = _depth_estimation_v2(image, config)
            if isinstance(v2_result, np.ndarray):
                return {'depth_map': v2_result, 'depth_metric': None, 'sky_mask': None}
            return v2_result

        if profile:
            print(f"  ⏱️  Depth V3 model ready: {time.perf_counter() - t0:.3f}s")

        # 安装 sky mask hook (仅 DA3Nested 模型支持)
        if model_type == 'nested':
            _install_sky_hook(depth_model)

        # OpenCV(BGR)->RGB, 然后转PIL
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # 推理参数
        process_res = int(config.get('depth_process_res', 672))

        t1 = time.perf_counter()
        with _GPU_INFERENCE_LOCK:
            with torch.inference_mode():
                prediction = depth_model.inference(
                    [pil_img],
                    export_dir=None,
                    export_format='mini_npz',
                    process_res=process_res,
                    process_res_method='upper_bound_resize',
                )

                # prediction.depth: [N, H, W] float32
                pred = prediction.depth[0]
                pred = pred.cpu().numpy() if hasattr(pred, 'cpu') else np.array(pred)

        if profile:
            print(f"  ⏱️  Depth V3 inference ({model_type}): {time.perf_counter() - t1:.3f}s")

        # 以下全部CPU操作，不持GPU锁，其他线程可立即开始GPU推理
        t2 = time.perf_counter()
        if pred.shape != (H, W):
            pred_resized = cv2.resize(pred, (W, H), interpolation=cv2.INTER_CUBIC)
        else:
            pred_resized = pred

        # 根据模型类型处理深度值
        depth_metric = None
        sky_mask = None

        if model_type == 'nested':
            # DA3NESTED: 输出已经是米数
            depth_metric = pred_resized.astype(np.float32)

            # 提取天空掩码 (由 hook 捕获)
            sky_mask_small = _get_sky_mask(depth_model)
            if sky_mask_small is not None:
                sky_mask = cv2.resize(
                    sky_mask_small.astype(np.uint8), (W, H),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                sky_pct = sky_mask.sum() / sky_mask.size * 100
                depth_metric[sky_mask] = np.inf
                if profile:
                    print(f"  [SKY] Sky mask (nested): {sky_pct:.1f}% pixels")

        elif model_type == 'metric':
            # DA3METRIC: 输出canonical depth at focal=300
            # 转换: meters = canonical * (actual_focal / 300.0)
            focal_length = float(config.get('depth_focal_length', 300))
            scale = focal_length / 300.0
            depth_metric = (pred_resized * scale).astype(np.float32)
            if profile:
                print(f"  [FOCAL] Focal conversion: canonical * {scale:.3f} (focal={focal_length})")
            # sky_mask 由 stage2_ai_inference 从 semantic_map 推导

        else:
            # DA3MONO: 相对深度，无米数
            depth_metric = None
            if profile:
                print(f"  [INFO] DA3MONO: relative depth only, no metric output")

        # V2-Style 视差归一化 (天空自动=255, 对比度增强)
        depth_map = _normalize_depth_v2style(
            pred_resized, sky_mask=sky_mask,
            contrast_boost=float(config.get('depth_contrast_boost', 2.0)),
        )

        if profile:
            if depth_metric is not None:
                non_sky = depth_metric[np.isfinite(depth_metric)]
                if len(non_sky) > 0:
                    print(f"  [DEPTH] Depth range: {float(non_sky.min()):.1f}m - {float(non_sky.max()):.1f}m")
            print(f"  [TIME] Depth V3 postprocess: {time.perf_counter() - t2:.3f}s")

    except Exception as e:
        print(f"  ❌ Depth Anything V3 出错: {e}")
        import traceback
        traceback.print_exc()
        print(f"  ⚠️  回退到 Depth Anything V2...")
        v2_result = _depth_estimation_v2(image, config)
        if isinstance(v2_result, np.ndarray):
            return {'depth_map': v2_result, 'depth_metric': None, 'sky_mask': None}
        return v2_result

    return {
        'depth_map': depth_map,
        'depth_metric': depth_metric,
        'sky_mask': sky_mask,
    }


def _refine_sky_gaps(
    depth_metric: np.ndarray,
    sky_mask: np.ndarray,
    depth_percentile: float = 85,
    dilate_kernel_size: int = 21,
    dilate_iterations: int = 2,
    max_rounds: int = 5,
    image_bgr: np.ndarray | None = None,
    semantic_map: np.ndarray | None = None,
) -> tuple:
    """修补天空缝隙: 树缝/建筑间隙中的天空像素 (迭代扩展 + 颜色/语义守卫)。

    保留原有的激进膨胀策略（21px/5轮）以深入树缝，
    但增加颜色和语义守卫，防止把绿色树冠、暗色物体、建筑等误标为天空。

    守卫规则:
      - 排除绿色像素 (H=30-80, S>=40): 树冠/植被
      - 排除暗色像素 (V<80): 阴影/深色物体
      - 排除硬性语义类: building, road, car 等永远不应是天空的类别

    参数:
        depth_metric: (H, W) float32, 天空=inf
        sky_mask: (H, W) bool, 已知天空
        depth_percentile: 深度阈值百分位 (默认 90)
        dilate_kernel_size: 膨胀核大小 (默认 21px)
        dilate_iterations: 膨胀迭代次数 (默认 2)
        max_rounds: 最大迭代轮数 (默认 5)
        image_bgr: (H, W, 3) uint8 BGR原图, 用于颜色守卫
        semantic_map: (H, W) uint8 ADE20K class ids, 用于语义守卫

    返回:
        (refined_sky_mask, total_gap_pixel_count)
    """
    H, W = depth_metric.shape[:2]
    finite_non_sky = depth_metric[np.isfinite(depth_metric) & ~sky_mask]
    if len(finite_non_sky) < 100:
        return sky_mask, 0

    depth_thresh = float(np.percentile(finite_non_sky, depth_percentile))

    # ---- Precompute HSV channels (reused by guard + color detection) ----
    h = s = v = None
    if image_bgr is not None and image_bgr.ndim == 3:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # ---- Build guard mask: pixels that should NEVER become sky ----
    guard = np.zeros((H, W), dtype=bool)

    # Color guard: reject green vegetation and dark pixels
    if h is not None:
        green_veg = (h >= 30) & (h <= 80) & (s >= 40)
        dark = (v < 80)
        guard |= green_veg | dark

    # Semantic guard: hard classes that are never sky
    if semantic_map is not None:
        _NEVER_SKY_IDS = {
            0, 1, 3, 5, 6, 9, 11, 12, 13, 20,  # wall, building, floor, ceiling, road, grass, sidewalk, person, earth, car
            33, 53, 72, 80, 83, 90, 116, 127,    # fence, stairs, signboard, truck, bus, van, pole, bicycle
        }
        for cls_id in _NEVER_SKY_IDS:
            guard |= (semantic_map == cls_id)

    # ---- Pass 1: Color+depth direct detection (no dilation needed) ----
    # Catches isolated sky gaps surrounded by green canopy that dilation
    # can never reach through. Uses stricter depth threshold (p95).
    current_sky = sky_mask.copy()
    total_added = 0

    if h is not None:
        p95 = float(np.percentile(finite_non_sky, 95))

        # Sky-like colors: blue, cyan, gray/overcast, bright white
        blue_sky = (h >= 90) & (h <= 135) & (s >= 20) & (s <= 200) & (v >= 100)
        cyan_sky = (h >= 80) & (h < 90) & (s >= 20) & (s <= 150) & (v >= 140)
        gray_sky = (s <= 25) & (v >= 150) & (v <= 240)
        bright_white = (s <= 30) & (v >= 200)
        sky_color = blue_sky | cyan_sky | gray_sky | bright_white

        # Upper 60% of image only
        upper_mask = np.zeros((H, W), dtype=bool)
        upper_mask[:int(H * 0.6), :] = True

        color_sky = (
            sky_color
            & upper_mask
            & (depth_metric > p95)
            & ~current_sky
            & ~guard
        )
        color_added = int(color_sky.sum())
        if color_added > 0:
            current_sky |= color_sky
            total_added += color_added

    # ---- Pass 2: Iterative dilation (aggressive reach into gaps) ----
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))

    for _round in range(max_rounds):
        sky_nearby = cv2.dilate(
            current_sky.astype(np.uint8) * 255, kernel,
            iterations=dilate_iterations,
        ) > 0

        gap_pixels = (
            (depth_metric > depth_thresh)
            & sky_nearby
            & ~current_sky
            & ~guard
        )
        added = int(gap_pixels.sum())
        if added == 0:
            break

        current_sky = current_sky | gap_pixels
        total_added += added

    return current_sky, total_added


def _install_sky_hook(depth_model):
    """
    给 DA3Nested 模型安装 hook，捕获 sky mask。
    DA3Nested._handle_sky_regions 内部计算了天空掩码但没有暴露，
    我们通过 monkey-patch 把它存下来。
    """
    inner_model = getattr(depth_model, 'model', None)
    if inner_model is None:
        return

    # 只处理 DA3Nested 类型
    cls_name = type(inner_model).__name__
    if 'Nested' not in cls_name:
        return

    # 如果已经 patch 过就跳过
    if getattr(inner_model, '_sky_hook_installed', False):
        return

    original_handle_sky = inner_model._handle_sky_regions

    def _patched_handle_sky(output, metric_output, sky_depth_def=200.0):
        from depth_anything_3.utils.alignment import compute_sky_mask
        non_sky_mask = compute_sky_mask(metric_output.sky, threshold=0.3)
        # 存储天空掩码: True = 天空
        sky_mask = ~non_sky_mask  # (B, S, H, W) bool tensor
        inner_model._captured_sky_mask = sky_mask.squeeze().cpu().numpy()
        return original_handle_sky(output, metric_output, sky_depth_def)

    inner_model._handle_sky_regions = _patched_handle_sky
    inner_model._sky_hook_installed = True
    inner_model._captured_sky_mask = None


def _get_sky_mask(depth_model) -> np.ndarray | None:
    """从 hook 中获取天空掩码"""
    inner_model = getattr(depth_model, 'model', None)
    if inner_model is None:
        return None
    mask = getattr(inner_model, '_captured_sky_mask', None)
    return mask


def _depth_estimation_v2(image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """
    深度估计 - 使用 Depth Anything V2 (原有实现)
    """
    H, W = image.shape[:2]
    
    try:
        profile = bool(config.get('profile', False))
        t0 = time.perf_counter()

        depth_model, depth_processor = get_depth_model(config)
        if depth_model is None or depth_processor is None:
            print(f"  ⚠️  警告: Depth Anything V2 未就绪，使用占位深度图")
            return _generate_placeholder_depth_map(H, W)

        if profile:
            print(f"  ⏱️  Depth model ready: {time.perf_counter() - t0:.3f}s")

        device = _get_torch_device(config)
        use_fp16 = bool(config.get('depth_use_fp16', True))

        # OpenCV(BGR)->RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 推理
        t1 = time.perf_counter()
        with _GPU_INFERENCE_LOCK:
            with torch.inference_mode():
                inputs = depth_processor(images=rgb, return_tensors='pt')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                if use_fp16 and device.type == 'cuda':
                    inputs = {k: v.half() if torch.is_floating_point(v) else v for k, v in inputs.items()}

                outputs = depth_model(**inputs)

                # transformers 的深度模型一般有 predicted_depth
                if hasattr(outputs, 'predicted_depth'):
                    pred = outputs.predicted_depth
                elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                    pred = outputs[0]
                else:
                    pred = getattr(outputs, 'logits', None)

                if pred is None:
                    raise RuntimeError("Depth Anything V2 输出不包含 predicted_depth/logits")

                # pred: (B, H', W')
                pred = pred.squeeze(0)
                pred = pred.detach().float().cpu().numpy()

        if profile:
            print(f"  ⏱️  Depth inference: {time.perf_counter() - t1:.3f}s")

        # 将预测结果 resize 回原图大小
        t2 = time.perf_counter()
        pred_resized = cv2.resize(pred, (W, H), interpolation=cv2.INTER_CUBIC)

        depth_map = _normalize_depth_to_uint8(pred_resized, invert=bool(config.get('depth_invert', True)))

        if profile:
            print(f"  ⏱️  Depth postprocess (resize+norm): {time.perf_counter() - t2:.3f}s")
        
    except ImportError as e:
        print(f"  ❌ 错误: 缺少必要的库，无法执行深度估计")
        print(f"  请安装: pip install transformers accelerate")
        print(f"  错误详情: {e}")
        # 返回简单梯度图，不中断流程
        return _generate_placeholder_depth_map(H, W)
    
    except Exception as e:
        print(f"  ❌ 深度估计出错: {e}")
        import traceback
        traceback.print_exc()
        # 返回简单梯度图，不中断流程
        return _generate_placeholder_depth_map(H, W)
    
    return depth_map


def _normalize_depth_to_uint8(depth_raw: np.ndarray, invert: bool = True) -> np.ndarray:
    """把 float 深度/逆深度图归一化到 uint8 [0,255]。

    约定输出：0=近，255=远。
    Depth Anything V2 的输出通常更像"逆深度"（近处更大），因此默认 invert=True。
    """
    depth_min = float(np.min(depth_raw))
    depth_max = float(np.max(depth_raw))
    if not np.isfinite(depth_min) or not np.isfinite(depth_max) or depth_max == depth_min:
        return np.full(depth_raw.shape, 128, dtype=np.uint8)

    norm = (depth_raw - depth_min) / (depth_max - depth_min)
    norm = (norm * 255.0).clip(0, 255).astype(np.uint8)
    if invert:
        norm = (255 - norm).astype(np.uint8)
    return norm


def _normalize_depth_v2style(depth_raw: np.ndarray, sky_mask: np.ndarray | None = None,
                              contrast_boost: float = 2.0) -> np.ndarray:
    """V2-Style 视差归一化 (参考 ComfyUI-DepthAnythingV3)。

    纯numpy/cv2实现，避免GPU传输开销。
    对于<1M像素的图像，numpy在7800X3D V-cache上比GPU+PCIe传输更快。

    流程:
        1. depth → disparity (1/depth)，天空自然变为~0
        2. 用 sky_mask 排除天空区域，仅对内容区域归一化
        3. 百分位裁剪 (1%-99%) 提升对比度
        4. power transform 增强近景细节
        5. 天空边缘抗锯齿
        6. 反转输出使 0=近, 255=远

    返回:
        depth_map: (H, W) uint8, 0=近, 255=远 (天空=255)
    """
    epsilon = 1e-6

    disparity = 1.0 / (depth_raw.astype(np.float64) + epsilon)

    if sky_mask is not None and sky_mask.any():
        content_mask = (~sky_mask).astype(np.float32)
    else:
        content_mask = np.ones_like(disparity, dtype=np.float32)

    content_pixels = disparity[content_mask > 0.5]
    if content_pixels.size > 100:
        p1 = np.percentile(content_pixels, 1)
        p99 = np.percentile(content_pixels, 99)
        if p99 - p1 < epsilon:
            p1 = content_pixels.min()
            p99 = content_pixels.max()
    else:
        p1 = disparity.min()
        p99 = disparity.max()

    disp_norm = (disparity - p1) / (p99 - p1 + epsilon)
    disp_norm = np.clip(disp_norm, 0.0, 1.0)
    disp_contrast = np.power(disp_norm, 1.0 / contrast_boost)

    if sky_mask is not None and sky_mask.any():
        content_smooth = cv2.blur(content_mask, (3, 3))
        disp_contrast = disp_contrast * content_smooth

    result = 1.0 - disp_contrast
    return (result * 255.0).clip(0, 255).astype(np.uint8)


def _generate_placeholder_semantic_map(semantic_map: np.ndarray, classes: List[str], H: int, W: int) -> None:
    """
    生成占位语义图 (仅用于测试)
    简单的垂直分块模式
    """
    num_classes = len(classes)
    if num_classes == 0:
        return
    
    # 简单的垂直分块
    block_height = H // num_classes
    for i, class_id in enumerate(range(1, num_classes + 1)):
        y_start = i * block_height
        y_end = (i + 1) * block_height if i < num_classes - 1 else H
        semantic_map[y_start:y_end, :] = class_id


def _generate_placeholder_depth_map(H: int, W: int) -> np.ndarray:
    """
    生成占位深度图 (仅用于测试)
    简单的垂直梯度
    """
    depth_map = np.zeros((H, W), dtype=np.uint8)
    for y in range(H):
        depth_map[y, :] = int((y / H) * 255)
    return depth_map


# ============================================
# 模型缓存管理 (用于优化性能)
# ============================================

_model_cache = {
    'semantic_model': None,
    'semantic_processor': None,
    'semantic_backend': None,
    'semantic_device': None,
    'depth_model': None,
    'depth_processor': None,
    'depth_model_v3': None,  # Depth Anything V3 模型
    'depth_model_depth_pro': None,  # Apple Depth Pro 模型
    'depth_transform_depth_pro': None,  # Depth Pro 预处理 transform
}


def get_semantic_model(config: Dict[str, Any]):
    """
    获取语义分割模型 (带缓存, 线程安全)
    只初始化一次，后续复用
    """
    if not TORCH_AVAILABLE:
        return None, None

    backend = str(config.get('semantic_backend', 'oneformer_ade20k')).strip().lower()

    # 快速路径: 已缓存且backend相同
    if (_model_cache['semantic_model'] is not None
            and _model_cache.get('semantic_backend') == backend):
        return _model_cache['semantic_model'], _model_cache['semantic_processor']

    # 慢路径: 加锁加载
    with _MODEL_LOAD_LOCK:
        # Double-check
        if (_model_cache['semantic_model'] is not None
                and _model_cache.get('semantic_backend') == backend):
            return _model_cache['semantic_model'], _model_cache['semantic_processor']

        device = _get_torch_device(config)

        # If backend changes between calls, reset semantic cache.
        if _model_cache.get('semantic_backend') != backend:
            _model_cache['semantic_model'] = None
            _model_cache['semantic_processor'] = None
            _model_cache['semantic_backend'] = backend
            _model_cache['semantic_device'] = None

        if backend.startswith('oneformer'):
            try:
                profile = bool(config.get('profile', False))
                t_import = time.perf_counter()
                print("  初始化 OneFormer（首次导入/下载权重可能较慢）...")
                from transformers import AutoProcessor, OneFormerForUniversalSegmentation
                if profile:
                    print(f"  ⏱️  import transformers(oneformer): {time.perf_counter() - t_import:.3f}s")
            except Exception as e:
                print(f"  ❌ 无法导入 OneFormer 相关依赖: {e}")
                return None, None

            model_id = str(config.get('oneformer_model_id', 'shi-labs/oneformer_ade20k_swin_large'))
            use_fp16 = bool(config.get('semantic_use_fp16', True))
            print(f"  加载 OneFormer(ADE20K-150): {model_id} (device={device}, fp16={use_fp16})")

            t_load = time.perf_counter()
            processor = AutoProcessor.from_pretrained(model_id)
            model = OneFormerForUniversalSegmentation.from_pretrained(model_id)
            model.eval()

            if device.type == 'cuda':
                model.to(device)
                if use_fp16:
                    model.half()

            if bool(config.get('profile', False)):
                print(f"  ⏱️  load processor+model: {time.perf_counter() - t_load:.3f}s")

            _model_cache['semantic_model'] = model
            _model_cache['semantic_processor'] = processor
            _model_cache['semantic_backend'] = backend
            _model_cache['semantic_device'] = str(device)
            return model, processor

        # LangSAM backend
        try:
            profile = bool(config.get('profile', False))
            t_import = time.perf_counter()
            print("  初始化 LangSAM（首次导入/下载权重可能较慢）...")
            from lang_sam import LangSAM
            if profile:
                print(f"  ⏱️  import lang_sam: {time.perf_counter() - t_import:.3f}s")
        except Exception as e:
            print(f"  ❌ 无法导入 lang_sam: {e}")
            return None, None

        sam_type = str(config.get('sam_type', '')).strip()
        if not sam_type:
            encoder = str(config.get('encoder', 'vitb')).lower()
            if encoder in ('vitb', 'vit_b', 'b'):
                sam_type = 'sam2.1_hiera_base_plus'
            elif encoder in ('vits', 'vit_s', 's'):
                sam_type = 'sam2.1_hiera_small'
            else:
                sam_type = 'sam2.1_hiera_small'

        print(f"  加载 LangSAM (sam_type={sam_type}, device={device})")
        model = LangSAM(sam_type=sam_type, device=device)

        _model_cache['semantic_model'] = model
        _model_cache['semantic_processor'] = None
        _model_cache['semantic_backend'] = backend
        _model_cache['semantic_device'] = str(device)
        return model, None


def _langsam_predict_masks(model, pil_img: Image.Image, text_prompt: str, config: Dict[str, Any] | None = None) -> Optional[np.ndarray]:
    """调用 LangSAM 并返回 masks ndarray。

    兼容不同版本返回格式：可能是 (masks, boxes, phrases, logits) 或 dict。
    返回:
      - None: 没有检测到
      - np.ndarray: (N,H,W) 或 (H,W)
    """
    cfg = config or {}
    box_threshold = float(cfg.get('box_threshold', 0.3))
    text_threshold = float(cfg.get('text_threshold', 0.25))

    # LangSAM/SAM2 predictor is not thread-safe; serialize predict calls.
    with _GPU_INFERENCE_LOCK:
        # LangSAM.predict expects lists and returns list[dict] (one per image)
        results = model.predict([pil_img], [text_prompt], box_threshold=box_threshold, text_threshold=text_threshold)
    if not results:
        return None
    first = results[0]
    if not isinstance(first, dict):
        return None

    masks = first.get('masks', None)
    if masks is None:
        return None

    masks = np.asarray(masks)
    if masks.size == 0:
        return None
    return masks


def get_depth_model(config: Dict[str, Any]):
    """
    获取深度估计模型 (带缓存)
    只初始化一次，后续复用
    """
    if _model_cache['depth_model'] is None:
        if not TORCH_AVAILABLE:
            return None, None

        try:
            # 这一段在首次运行时可能会“卡住”，通常是 transformers/torch 的大导入 + CUDA 初始化，并非 bug。
            profile = bool(config.get('profile', False))
            t_import = time.perf_counter()
            print("  初始化 Depth Anything V2（导入 transformers，首次可能较慢）...")
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            if profile:
                print(f"  ⏱️  import transformers: {time.perf_counter() - t_import:.3f}s")
        except Exception as e:
            print(f"  ❌ 无法导入 transformers 深度模型: {e}")
            return None, None

        model_id = str(config.get('depth_model_id', 'depth-anything/Depth-Anything-V2-Base-hf'))
        device = _get_torch_device(config)
        use_fp16 = bool(config.get('depth_use_fp16', True))

        print(f"  加载 Depth Anything V2: {model_id} (device={device}, fp16={use_fp16})")

        t_load = time.perf_counter()
        # transformers 会提示 future default use_fast=True；这里尽量显式指定。
        try:
            processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
        except TypeError:
            processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForDepthEstimation.from_pretrained(model_id)
        if profile:
            print(f"  ⏱️  load processor+model: {time.perf_counter() - t_load:.3f}s")
        model.eval()

        if device.type == 'cuda':
            t_to = time.perf_counter()
            model.to(device)
            if use_fp16:
                model.half()
            if profile:
                print(f"  ⏱️  model.to(device)/half: {time.perf_counter() - t_to:.3f}s")

        _model_cache['depth_model'] = model
        _model_cache['depth_processor'] = processor

    return _model_cache['depth_model'], _model_cache['depth_processor']


def get_depth_model_v3(config: Dict[str, Any]):
    """
    获取 Depth Anything V3 模型 (带缓存, 线程安全)
    """
    # 快速路径: 已缓存则直接返回
    if _model_cache['depth_model_v3'] is not None:
        return _model_cache['depth_model_v3']

    # 慢路径: 加锁加载，防止多线程同时加载大模型导致OOM
    with _MODEL_LOAD_LOCK:
        # Double-check: 另一个线程可能已经加载完成
        if _model_cache['depth_model_v3'] is not None:
            return _model_cache['depth_model_v3']

        if not TORCH_AVAILABLE:
            return None

        try:
            profile = bool(config.get('profile', False))
            t_import = time.perf_counter()
            print("  初始化 Depth Anything V3...")
            from depth_anything_3.api import DepthAnything3
            if profile:
                print(f"  ⏱️  import depth_anything_3: {time.perf_counter() - t_import:.3f}s")
        except ImportError as e:
            print(f"  ❌ 无法导入 depth_anything_3: {e}")
            print(f"  请安装: pip install git+https://github.com/ByteDance-Seed/depth-anything-3.git")
            return None

        # 支持的模型:
        #   - depth-anything/DA3NESTED-GIANT-LARGE-1.1 : 度量深度(米) + 天空检测 (推荐)
        #   - depth-anything/DA3NESTED-GIANT-LARGE     : 度量深度(米) + 天空检测
        #   - depth-anything/DA3METRIC-LARGE           : 规范化深度 (需要焦距转换)
        #   - depth-anything/DA3MONO-LARGE             : 相对深度 (非度量)
        model_id = str(config.get('depth_model_id_v3', 'depth-anything/DA3NESTED-GIANT-LARGE-1.1'))
        device = _get_torch_device(config)

        print(f"  加载 Depth Anything V3: {model_id} (device={device})")

        t_load = time.perf_counter()
        try:
            model = DepthAnything3.from_pretrained(model_id)
            model = model.to(device=device)
            model.eval()
        except Exception as e:
            print(f"  ❌ 加载 Depth Anything V3 失败: {e}")
            import traceback
            traceback.print_exc()
            return None

        if profile:
            print(f"  ⏱️  load DA3 model: {time.perf_counter() - t_load:.3f}s")

        _model_cache['depth_model_v3'] = model

    return _model_cache['depth_model_v3']


def get_depth_model_depth_pro(config: Dict[str, Any]):
    """
    获取 Apple Depth Pro 模型 (带缓存)
    边缘锐利，度量深度精确，适合街景
    """
    if _model_cache['depth_model_depth_pro'] is None:
        if not TORCH_AVAILABLE:
            return None, None

        try:
            profile = bool(config.get('profile', False))
            t_import = time.perf_counter()
            print("  初始化 Depth Pro (Apple)...")
            import depth_pro
            if profile:
                print(f"  ⏱️  import depth_pro: {time.perf_counter() - t_import:.3f}s")
        except ImportError as e:
            print(f"  ❌ 无法导入 depth_pro: {e}")
            print(f"  请安装: pip install git+https://github.com/apple/ml-depth-pro.git")
            print(f"  然后运行: source get_pretrained_models.sh 下载权重")
            return None, None

        device = _get_torch_device(config)

        print(f"  加载 Depth Pro (device={device})")

        t_load = time.perf_counter()
        try:
            model, transform = depth_pro.create_model_and_transforms()
            model = model.to(device)
            model.eval()
        except Exception as e:
            print(f"  ❌ 加载 Depth Pro 失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None

        if profile:
            print(f"  ⏱️  load Depth Pro model: {time.perf_counter() - t_load:.3f}s")

        _model_cache['depth_model_depth_pro'] = model
        _model_cache['depth_transform_depth_pro'] = transform

    return _model_cache['depth_model_depth_pro'], _model_cache['depth_transform_depth_pro']


def _get_torch_device(config: Dict[str, Any]):
    if not TORCH_AVAILABLE:
        return None
    device_str = str(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    return torch.device(device_str)


def preload_models(config: Dict[str, Any]):
    """
    在主线程中预加载所有 AI 模型到 GPU，避免 worker 线程竞争加载。
    应在任何 worker 启动前调用一次。
    """
    import sys
    print("预加载 AI 模型...", file=sys.stderr)
    t0 = time.perf_counter()

    if config.get('enable_semantic', True):
        get_semantic_model(config)

    depth_version = str(config.get('depth_version', 'v3')).lower()
    if depth_version == 'v3':
        get_depth_model_v3(config)
    elif depth_version == 'depth_pro':
        get_depth_model_depth_pro(config)
    else:
        get_depth_model(config)

    print(f"模型预加载完成 ({time.perf_counter() - t0:.1f}s)", file=sys.stderr)


def clear_model_cache():
    """
    清除模型缓存 (释放GPU内存)
    """
    global _model_cache
    _model_cache['semantic_model'] = None
    _model_cache['semantic_processor'] = None
    _model_cache['semantic_backend'] = None
    _model_cache['semantic_device'] = None
    _model_cache['depth_model'] = None
    _model_cache['depth_processor'] = None
    _model_cache['depth_model_v3'] = None
    _model_cache['depth_model_depth_pro'] = None
    _model_cache['depth_transform_depth_pro'] = None
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
