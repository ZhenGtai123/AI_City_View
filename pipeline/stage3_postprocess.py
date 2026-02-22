"""pipeline.stage3_postprocess

阶段3: 后处理优化
功能: 智能空洞填充 + 中值滤波平滑
"""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np


def stage3_postprocess(semantic_map: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """阶段3: 后处理优化。

    按 plan.md：
    - 空洞填充：对每个类别二值掩码做闭运算，仅写回原本为0的像素
    - 平滑：对整张语义图做中值滤波
    """
    if semantic_map.ndim != 2:
        raise ValueError(f"semantic_map 必须是二维 (H, W)，当前 shape={semantic_map.shape}")
    if semantic_map.dtype != np.uint8:
        raise ValueError(f"semantic_map 必须是 uint8，当前 dtype={semantic_map.dtype}")

    enable_hole_filling = bool(config.get('enable_hole_filling', True))
    enable_median_blur = bool(config.get('enable_median_blur', True))
    hole_fill_kernel_size = int(config.get('hole_fill_kernel_size', 5))
    blur_kernel_size = int(config.get('blur_kernel_size', 5))

    processed_map = semantic_map.copy()
    holes_filled = 0
    pixels_modified = 0

    # ========== 3.1 智能空洞填充（形态学闭运算）==========
    if enable_hole_filling:
        holes_before = int(np.sum(processed_map == 0))
        num_classes = int(processed_map.max())

        if holes_before == 0:
            pass
        elif num_classes == 0:
            # 全是0，无法填充
            pass
        else:
            # kernel size 不强制奇数，但确保 >= 3
            hole_fill_kernel_size = max(3, hole_fill_kernel_size)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (hole_fill_kernel_size, hole_fill_kernel_size),
            )

            filled_map = processed_map.copy()
            for class_id in range(1, num_classes + 1):
                class_mask = (processed_map == class_id).astype(np.uint8)
                if class_mask.sum() == 0:
                    continue

                closed_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel)
                new_pixels = (closed_mask == 1) & (filled_map == 0)
                if new_pixels.any():
                    filled_map[new_pixels] = class_id

            processed_map = filled_map
            holes_after = int(np.sum(processed_map == 0))
            holes_filled = max(0, holes_before - holes_after)

    # ========== 3.2 中值滤波平滑 ==========
    if enable_median_blur:
        blur_kernel_size = max(3, blur_kernel_size)
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1

        # 保存天空像素 (class 2): 中值滤波会吞掉树缝间的小块天空
        sky_before = (processed_map == 2)

        before_blur = processed_map
        smoothed_map = cv2.medianBlur(before_blur, ksize=blur_kernel_size)
        pixels_modified = int(np.sum(smoothed_map != before_blur))

        # 恢复被中值滤波消除的天空像素
        if sky_before.any():
            smoothed_map[sky_before] = 2

        processed_map = smoothed_map

    return {
        'semantic_map_processed': processed_map,
        'processing_stats': {
            'holes_filled': int(holes_filled),
            'pixels_modified': int(pixels_modified),
        },
    }


