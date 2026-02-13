"""pipeline.stage4_depth_layering

阶段4: 景深分层
功能: 将深度图分为前景/中景/背景三层
支持语义感知分层：天空等特定类别强制归入指定层
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set

import numpy as np


# ADE20K 中需要强制归入背景的类别ID
# 注意: 不再强制将天空归入背景，因为 Depth Anything V2/V3
# 能够正确识别天空为最远处（深度值接近255）
# 如果需要恢复旧行为，可以添加 sky=2
ADE20K_FORCE_BACKGROUND_IDS: Set[int] = set()  # 空集合，依赖深度模型

# ADE20K 中可能需要强制归入前景的类别ID（暂时留空，可扩展）
ADE20K_FORCE_FOREGROUND_IDS: Set[int] = set()


def stage4_depth_layering(
    depth_map: np.ndarray,
    config: Dict[str, Any],
    semantic_map: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """阶段4: 景深分层（支持语义感知）。

    depth_map: (H, W) uint8, 值范围 [0, 255]
      - 0: 最近（前景）
      - 255: 最远（背景）
    
    semantic_map: (H, W) uint8, 可选。如果提供，将使用语义信息改进分层：
      - 天空(id=2)强制归入背景
      - 基于非天空区域计算深度阈值
    """
    if depth_map.ndim != 2:
        raise ValueError(f"depth_map 必须是二维 (H, W)，当前 shape={depth_map.shape}")
    if depth_map.dtype != np.uint8:
        raise ValueError(f"depth_map 必须是 uint8，当前 dtype={depth_map.dtype}")

    H, W = depth_map.shape
    total_pixels = int(H * W)
    
    # 检查是否启用语义感知分层
    use_semantic_aware = config.get('use_semantic_aware_layering', True)
    
    # 获取语义掩码
    sky_mask = None
    force_bg_mask = np.zeros((H, W), dtype=bool)
    force_fg_mask = np.zeros((H, W), dtype=bool)
    
    if semantic_map is not None and use_semantic_aware:
        if semantic_map.shape != (H, W):
            print(f"  ⚠️  语义图尺寸不匹配，禁用语义感知分层")
        else:
            # 构建强制背景掩码（天空等）
            for class_id in ADE20K_FORCE_BACKGROUND_IDS:
                force_bg_mask |= (semantic_map == class_id)
            
            # 构建强制前景掩码
            for class_id in ADE20K_FORCE_FOREGROUND_IDS:
                force_fg_mask |= (semantic_map == class_id)
            
            sky_mask = (semantic_map == 2)  # 记录天空掩码用于统计
    
    # 计算深度阈值
    split_method = str(config.get('split_method', 'percentile')).lower()
    fg_ratio = float(config.get('fg_ratio', 0.33))
    bg_ratio = float(config.get('bg_ratio', 0.33))

    if split_method == 'fixed_threshold':
        threshold_1 = float(config.get('threshold_1', 85))
        threshold_2 = float(config.get('threshold_2', 170))
        t1, t2 = threshold_1, threshold_2
    else:
        # 百分位数方法
        fg_ratio = min(max(fg_ratio, 0.0), 1.0)
        bg_ratio = min(max(bg_ratio, 0.0), 1.0)
        p1 = fg_ratio * 100.0
        p2 = (1.0 - bg_ratio) * 100.0
        if p1 >= p2:
            p1, p2 = 33.0, 66.0
        
        # 关键改进：基于非强制区域计算阈值
        if use_semantic_aware and force_bg_mask.any():
            # 排除强制背景区域（如天空）来计算阈值
            non_forced_mask = ~(force_bg_mask | force_fg_mask)
            if non_forced_mask.sum() > 100:  # 确保有足够像素
                depth_for_threshold = depth_map[non_forced_mask]
                t1 = float(np.percentile(depth_for_threshold, p1))
                t2 = float(np.percentile(depth_for_threshold, p2))
            else:
                t1 = float(np.percentile(depth_map, p1))
                t2 = float(np.percentile(depth_map, p2))
        else:
            t1 = float(np.percentile(depth_map, p1))
            t2 = float(np.percentile(depth_map, p2))

    # 基于深度计算初始掩码
    foreground_mask = depth_map <= t1
    background_mask = depth_map > t2
    middleground_mask = ~(foreground_mask | background_mask)
    
    # 应用语义强制规则
    semantic_corrections = {
        'sky_to_background': 0,
        'forced_to_foreground': 0,
    }
    
    if use_semantic_aware:
        # 天空等强制归入背景
        if force_bg_mask.any():
            # 从前景和中景移除，加入背景
            was_fg = foreground_mask & force_bg_mask
            was_mg = middleground_mask & force_bg_mask
            semantic_corrections['sky_to_background'] = int(was_fg.sum() + was_mg.sum())
            
            foreground_mask = foreground_mask & ~force_bg_mask
            middleground_mask = middleground_mask & ~force_bg_mask
            background_mask = background_mask | force_bg_mask
        
        # 强制前景（如果配置了的话）
        if force_fg_mask.any():
            was_mg = middleground_mask & force_fg_mask
            was_bg = background_mask & force_fg_mask
            semantic_corrections['forced_to_foreground'] = int(was_mg.sum() + was_bg.sum())
            
            middleground_mask = middleground_mask & ~force_fg_mask
            background_mask = background_mask & ~force_fg_mask
            foreground_mask = foreground_mask | force_fg_mask

    fg_pixels = int(foreground_mask.sum())
    mg_pixels = int(middleground_mask.sum())
    bg_pixels = int(background_mask.sum())

    def _percent(x: int) -> float:
        return (x / total_pixels * 100.0) if total_pixels > 0 else 0.0

    result = {
        'foreground_mask': foreground_mask.astype(bool),
        'middleground_mask': middleground_mask.astype(bool),
        'background_mask': background_mask.astype(bool),
        'depth_thresholds': {
            'P33': float(t1),
            'P66': float(t2),
        },
        'layer_stats': {
            'foreground_pixels': fg_pixels,
            'middleground_pixels': mg_pixels,
            'background_pixels': bg_pixels,
            'foreground_percent': float(_percent(fg_pixels)),
            'middleground_percent': float(_percent(mg_pixels)),
            'background_percent': float(_percent(bg_pixels)),
        },
    }
    
    # 添加语义感知统计
    if use_semantic_aware and semantic_map is not None:
        result['semantic_aware_stats'] = {
            'enabled': True,
            'sky_pixels_forced_to_background': semantic_corrections['sky_to_background'],
            'pixels_forced_to_foreground': semantic_corrections['forced_to_foreground'],
        }
        if sky_mask is not None:
            result['semantic_aware_stats']['total_sky_pixels'] = int(sky_mask.sum())
    
    return result


