"""
阶段1: 图片预处理 (单图模式)
功能: 读取单张图片，创建副本，提取属性，生成元数据
用于原有pipeline的兼容
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def stage1_preprocess_single(image_path: str) -> Dict[str, Any]:
    """
    阶段1: 图片预处理 (单图模式)
    
    参数:
        image_path: str - 输入图片路径
    
    返回:
        dict - 包含以下键:
            - original: np.ndarray - 原始图片 (H, W, 3) BGR格式
            - original_copy: np.ndarray - 原始图片副本 (H, W, 3) BGR格式
            - height: int - 图片高度
            - width: int - 图片宽度
            - metadata: dict - 元数据
    
    异常:
        FileNotFoundError: 文件不存在
        ValueError: 无法读取图片或图片格式不正确
    """
    # ========== 步骤1: 读取图片文件 ==========
    image_path_obj = Path(image_path)
    
    if not image_path_obj.exists():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    
    # 使用OpenCV读取图片 (BGR格式)
    original = cv2.imread(str(image_path_obj))
    
    if original is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # ========== 步骤2: 创建副本 ==========
    # 深拷贝原始图片
    original_copy = original.copy()
    
    # ========== 步骤3: 提取图片属性 ==========
    height, width, channels = original.shape
    
    # 验证图片格式
    if channels != 3:
        raise ValueError(f"图片必须是3通道 (RGB/BGR), 当前通道数: {channels}")
    
    if original.dtype != np.uint8:
        raise ValueError(f"图片必须是8位格式, 当前类型: {original.dtype}")
    
    if height <= 0 or width <= 0:
        raise ValueError(f"图片尺寸无效: {width}x{height}")
    
    # ========== 步骤4: 生成元数据 ==========
    # 提取文件名信息
    filename = image_path_obj.name
    basename = image_path_obj.stem
    extension = image_path_obj.suffix
    
    # 计算像素总数
    total_pixels = height * width
    
    # 生成尺寸字符串
    size_str = f"{width}x{height}"
    
    # 时间戳
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    metadata = {
        'filename': filename,
        'basename': basename,
        'extension': extension,
        'size_str': size_str,
        'width': width,
        'height': height,
        'total_pixels': total_pixels,
        'channels': channels,
        'dtype': str(original.dtype),
        'timestamp': timestamp,
        'file_path': str(image_path_obj.absolute())
    }
    
    # ========== 返回结果 ==========
    result = {
        'original': original,
        'original_copy': original_copy,
        'height': height,
        'width': width,
        'metadata': metadata
    }
    
    return result
