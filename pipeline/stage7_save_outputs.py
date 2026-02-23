"""pipeline.stage7_save_outputs

阶段7: 保存输出
功能: 保存23张PNG图片和元数据JSON
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np


def stage7_save_outputs(
    images: Dict[str, Any],
    output_dir: str,
    image_basename: str,
    metadata: Dict[str, Any],
    depth_metric: Any = None,
) -> Dict[str, Any]:
    """
    阶段7: 保存输出

    参数:
        images: Dict[str, np.ndarray] - 来自阶段6的23张图片
        output_dir: str - 输出目录路径
        image_basename: str - 原图文件名（不含扩展名）
        metadata: dict - 处理元数据
        depth_metric: np.ndarray (float32) - 度量深度(米), 可选

    返回:
        dict - 包含保存结果
    """
    output_path = _prepare_output_directory(output_dir)

    image_order = [
        # 基础分析图
        'semantic_map', 'depth_map', 'openness_map', 'fmb_map',
        # 掩码图
        'foreground_map', 'middleground_map', 'background_map',
        # 原图
        'original',
        # 语义分层
        'semantic_foreground', 'semantic_middleground', 'semantic_background',
        # 深度分层
        'depth_foreground', 'depth_middleground', 'depth_background',
        # 开放度分层
        'openness_foreground', 'openness_middleground', 'openness_background',
        # 原图分层
        'original_foreground', 'original_middleground', 'original_background',
        # FMB分层
        'fmb_foreground', 'fmb_middleground', 'fmb_background',
    ]

    saved_files: List[str] = []
    errors: List[str] = []

    cfg = (metadata.get('config', {}) or {}) if isinstance(metadata, dict) else {}
    # OpenCV PNG compression: 0 (fastest, largest) .. 9 (slowest, smallest)
    png_compression = int(cfg.get('png_compression', 3))
    png_compression = max(0, min(9, png_compression))

    # 保存图片: 先按 image_order 顺序保存核心图，再保存额外图
    all_names = list(image_order)
    for name in images:
        if name not in all_names:
            all_names.append(name)

    for name in all_names:
        if name not in images:
            if name in image_order:
                errors.append(f"缺失图片: {name}")
            continue

        img = images[name]
        if not isinstance(img, np.ndarray):
            errors.append(f"图片类型错误: {name} 不是 np.ndarray")
            continue

        filename = f"{name}.png"
        filepath = output_path / filename
        try:
            ok = cv2.imwrite(
                str(filepath),
                img,
                [cv2.IMWRITE_PNG_COMPRESSION, png_compression],
            )
            if ok:
                saved_files.append(str(filepath))
            else:
                errors.append(f"保存失败: {filepath}")
        except Exception as e:
            errors.append(f"保存 {name} 时出错: {str(e)}")

    # 保存度量深度 (.npy)
    if depth_metric is not None and isinstance(depth_metric, np.ndarray):
        try:
            npy_path = output_path / 'depth_metric.npy'
            np.save(str(npy_path), depth_metric)
            saved_files.append(str(npy_path))
        except Exception as e:
            errors.append(f"保存 depth_metric.npy 时出错: {str(e)}")

    # 保存元数据
    try:
        metadata_path = output_path / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(_to_jsonable(metadata), f, ensure_ascii=False, indent=2)
        saved_files.append(str(metadata_path))
    except Exception as e:
        errors.append(f"保存 metadata.json 时出错: {str(e)}")

    return {
        'output_dir': str(output_path),
        'saved_files': saved_files,
        'success': len(errors) == 0,
        'errors': errors,
    }


def _prepare_output_directory(output_dir: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not os.access(output_path, os.W_OK):
        raise PermissionError(f"无法写入目录: {output_path}")

    return output_path


def _to_jsonable(obj: Any) -> Any:
    """将常见 numpy/Path 类型转换为可 json 序列化结构。"""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return str(obj)


