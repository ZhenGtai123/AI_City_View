"""pipeline.stage6_generate_images

阶段6: 生成20张图片
功能: 生成所有输出图片（20张）
使用纯numpy操作，无需GPU（LUT查表和boolean masking在CPU上更快）
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import cv2
import numpy as np

from .ade20k_palette import ade20k_palette_bgr_dict


def stage6_generate_images(
    original_copy: np.ndarray,
    semantic_map: np.ndarray,
    depth_map: np.ndarray,
    openness_map: np.ndarray,
    foreground_mask: np.ndarray,
    middleground_mask: np.ndarray,
    background_mask: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """阶段6: 生成20张图片（按 plan.md 的固定 key 集合）。"""
    _validate_inputs(
        original_copy,
        semantic_map,
        depth_map,
        openness_map,
        foreground_mask,
        middleground_mask,
        background_mask,
    )

    backend = str(config.get('semantic_backend', '')).strip().lower()
    use_ade20k_palette = bool(config.get('use_ade20k_palette', False))

    colors = config.get('colors', None)
    if backend.startswith('oneformer') and use_ade20k_palette:
        colors = ade20k_palette_bgr_dict(int(semantic_map.max()))

    semantic_colored = _colorize_semantic(semantic_map, colors, oneformer=backend.startswith('oneformer'))
    depth_colored = _colorize_depth(depth_map, colormap=str(config.get('depth_colormap', 'INFERNO')))
    openness_colored = _colorize_openness(openness_map)
    fmb_colored = _create_fmb_visualization(foreground_mask, middleground_mask, background_mask)

    mask_images = _create_mask_images(foreground_mask, middleground_mask, background_mask)

    original = original_copy.copy()
    layered = _generate_layered_images(
        semantic_colored,
        depth_colored,
        openness_colored,
        original,
        foreground_mask,
        middleground_mask,
        background_mask,
    )

    images: Dict[str, np.ndarray] = {}
    images['semantic_map'] = semantic_colored
    images['depth_map'] = depth_colored
    images['openness_map'] = openness_colored
    images['fmb_map'] = fmb_colored
    images.update(mask_images)
    images['original'] = original
    images.update(layered)

    _validate_output_images(images)
    return {'images': images}


def _validate_inputs(
    original_copy: np.ndarray,
    semantic_map: np.ndarray,
    depth_map: np.ndarray,
    openness_map: np.ndarray,
    fg_mask: np.ndarray,
    mg_mask: np.ndarray,
    bg_mask: np.ndarray,
) -> None:
    if original_copy.ndim != 3 or original_copy.shape[2] != 3:
        raise ValueError(f"original_copy 必须是 (H,W,3)，当前 shape={original_copy.shape}")
    if original_copy.dtype != np.uint8:
        raise ValueError(f"original_copy 必须是 uint8，当前 dtype={original_copy.dtype}")

    for name, arr in (
        ('semantic_map', semantic_map),
        ('depth_map', depth_map),
        ('openness_map', openness_map),
    ):
        if arr.ndim != 2:
            raise ValueError(f"{name} 必须是二维 (H,W)，当前 shape={arr.shape}")
        if arr.dtype != np.uint8:
            raise ValueError(f"{name} 必须是 uint8，当前 dtype={arr.dtype}")

    H, W = semantic_map.shape
    if depth_map.shape != (H, W) or openness_map.shape != (H, W):
        raise ValueError("semantic/depth/openness 尺寸必须一致")
    if original_copy.shape[:2] != (H, W):
        raise ValueError("original_copy 与语义/深度尺寸必须一致")

    for name, mask in (('foreground_mask', fg_mask), ('middleground_mask', mg_mask), ('background_mask', bg_mask)):
        if mask.shape != (H, W):
            raise ValueError(f"{name} 尺寸必须为 (H,W)，当前 shape={mask.shape}")


def _colorize_semantic(semantic_map: np.ndarray, colors: Mapping[int, Any] | None, *, oneformer: bool = False) -> np.ndarray:
    H, W = semantic_map.shape

    palette = [
        (255, 200, 150),
        (100, 255, 100),
        (50, 150, 50),
        (120, 120, 180),
        (180, 120, 120),
        (100, 200, 255),
    ]

    max_id = int(semantic_map.max())
    if colors is None:
        colors_full: Dict[int, Any] = ({0: (0, 0, 0)} if not oneformer else {})
        start = 1 if not oneformer else 0
        for class_id in range(start, max_id + 1):
            colors_full[class_id] = palette[(class_id - start) % len(palette)]
        colors = colors_full
    else:
        colors_full = {int(k): v for k, v in dict(colors).items()}
        if not oneformer:
            colors_full.setdefault(0, (0, 0, 0))
        for class_id in range(1, max_id + 1):
            if class_id not in colors_full:
                colors_full[class_id] = palette[(class_id - 1) % len(palette)]
        colors = colors_full

    # numpy LUT查表 (比GPU传输+查表更快)
    color_lut = np.zeros((256, 3), dtype=np.uint8)
    for class_id, bgr in colors.items():
        if 0 <= class_id < 256:
            color_lut[class_id] = np.array(bgr, dtype=np.uint8)

    colored = color_lut[semantic_map]
    return colored


def _colorize_depth(depth_map: np.ndarray, colormap: str = 'INFERNO') -> np.ndarray:
    cmap = colormap.strip().upper()
    if cmap == 'JET':
        return cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    if cmap == 'VIRIDIS':
        return cv2.applyColorMap(depth_map, cv2.COLORMAP_VIRIDIS)
    return cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)


def _colorize_openness(openness_map: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(openness_map, cv2.COLOR_GRAY2BGR)


def _create_fmb_visualization(fg_mask: np.ndarray, mg_mask: np.ndarray, bg_mask: np.ndarray) -> np.ndarray:
    H, W = fg_mask.shape
    fmb = np.zeros((H, W, 3), dtype=np.uint8)
    fg = fg_mask.astype(bool)
    mg = mg_mask.astype(bool)
    bg = bg_mask.astype(bool)
    fmb[fg] = [0, 255, 0]      # 绿色 = 前景
    fmb[mg] = [0, 255, 255]    # 黄色 = 中景
    fmb[bg] = [255, 0, 0]      # 蓝色 = 背景
    return fmb


def _create_mask_images(fg_mask: np.ndarray, mg_mask: np.ndarray, bg_mask: np.ndarray) -> Dict[str, np.ndarray]:
    fg = (fg_mask.astype(bool) * 255).astype(np.uint8)
    mg = (mg_mask.astype(bool) * 255).astype(np.uint8)
    bg = (bg_mask.astype(bool) * 255).astype(np.uint8)

    return {
        'foreground_map': cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR),
        'middleground_map': cv2.cvtColor(mg, cv2.COLOR_GRAY2BGR),
        'background_map': cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR),
    }


def _apply_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.zeros_like(image)
    m = mask.astype(bool)
    masked[m] = image[m]
    return masked


def _generate_layered_images(
    semantic_colored: np.ndarray,
    depth_colored: np.ndarray,
    openness_colored: np.ndarray,
    original: np.ndarray,
    fg_mask: np.ndarray,
    mg_mask: np.ndarray,
    bg_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    base_images = {
        'semantic': semantic_colored,
        'depth': depth_colored,
        'openness': openness_colored,
        'original': original,
    }
    masks = {
        'foreground': fg_mask.astype(bool),
        'middleground': mg_mask.astype(bool),
        'background': bg_mask.astype(bool),
    }

    # 纯numpy路径: boolean indexing 在V-cache CPU上非常快
    results: Dict[str, np.ndarray] = {}
    for base_name, base_img in base_images.items():
        for mask_name, m in masks.items():
            results[f"{base_name}_{mask_name}"] = _apply_mask_to_image(base_img, m)

    return results


def _validate_output_images(images: Dict[str, np.ndarray]) -> None:
    expected_order = [
        'semantic_map', 'depth_map', 'openness_map', 'fmb_map',
        'foreground_map', 'middleground_map', 'background_map',
        'original',
        'semantic_foreground', 'semantic_middleground', 'semantic_background',
        'depth_foreground', 'depth_middleground', 'depth_background',
        'openness_foreground', 'openness_middleground', 'openness_background',
        'original_foreground', 'original_middleground', 'original_background',
    ]
    missing = [k for k in expected_order if k not in images]
    if missing:
        raise ValueError(f"阶段6缺失图片: {missing}")
    if len(images) != 20:
        extra = [k for k in images.keys() if k not in expected_order]
        raise ValueError(f"阶段6输出图片数量必须为20，当前={len(images)}，extra_keys={extra}")

    for name, img in images.items():
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"输出图片 {name} 必须是 (H,W,3)，当前 shape={img.shape}")
        if img.dtype != np.uint8:
            raise ValueError(f"输出图片 {name} 必须是 uint8，当前 dtype={img.dtype}")
