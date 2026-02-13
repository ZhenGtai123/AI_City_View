"""pipeline.gpu_utils

GPU加速工具模块
提供统一的GPU操作接口，支持PyTorch CUDA加速
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional

# GPU加速支持
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        TORCH_DEVICE = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        TORCH_DEVICE = torch.device('cpu')
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_DEVICE = None


def to_gpu(arr: np.ndarray, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """将NumPy数组传输到GPU"""
    if not TORCH_AVAILABLE:
        raise RuntimeError("CUDA not available")
    tensor = torch.from_numpy(arr)
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor.to(TORCH_DEVICE)


def to_cpu(tensor: torch.Tensor) -> np.ndarray:
    """将GPU张量传回CPU"""
    return tensor.cpu().numpy()


# ==================== GPU加速的图像操作 ====================

def gpu_colorize_semantic(semantic_map: np.ndarray, color_lut: np.ndarray) -> np.ndarray:
    """GPU加速的语义图着色
    
    Args:
        semantic_map: (H, W) uint8 语义图
        color_lut: (256, 3) uint8 颜色查找表 (BGR格式)
        
    Returns:
        colored: (H, W, 3) uint8 着色后的图像
    """
    if not TORCH_AVAILABLE:
        # CPU fallback using numpy advanced indexing
        return color_lut[semantic_map]
    
    # GPU path
    sem_gpu = torch.from_numpy(semantic_map.astype(np.int64)).to(TORCH_DEVICE)
    lut_gpu = torch.from_numpy(color_lut).to(TORCH_DEVICE)
    
    # 索引操作在GPU上
    colored_gpu = lut_gpu[sem_gpu]
    
    return colored_gpu.cpu().numpy().astype(np.uint8)


def gpu_apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """GPU加速的掩码应用
    
    Args:
        image: (H, W, 3) uint8 图像
        mask: (H, W) bool 掩码
        
    Returns:
        masked: (H, W, 3) uint8 应用掩码后的图像
    """
    if not TORCH_AVAILABLE:
        # CPU path
        result = np.zeros_like(image)
        result[mask] = image[mask]
        return result
    
    # GPU path
    img_gpu = torch.from_numpy(image).to(TORCH_DEVICE)
    mask_gpu = torch.from_numpy(mask).to(TORCH_DEVICE)
    
    # 使用where操作
    result_gpu = torch.where(mask_gpu.unsqueeze(-1), img_gpu, torch.zeros_like(img_gpu))
    
    return result_gpu.cpu().numpy()


def gpu_batch_apply_masks(image: np.ndarray, masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """GPU加速的批量掩码应用 - 一次传输多次操作
    
    Args:
        image: (H, W, 3) uint8 图像
        masks: {'name': mask_array} 字典，每个mask是(H, W) bool
        
    Returns:
        results: {'name': masked_image} 字典
    """
    if not TORCH_AVAILABLE:
        # CPU path
        return {name: gpu_apply_mask(image, mask) for name, mask in masks.items()}
    
    # GPU path - 一次传输图像
    img_gpu = torch.from_numpy(image).to(TORCH_DEVICE)
    zero_gpu = torch.zeros_like(img_gpu)
    
    results = {}
    for name, mask in masks.items():
        mask_gpu = torch.from_numpy(mask).to(TORCH_DEVICE)
        result_gpu = torch.where(mask_gpu.unsqueeze(-1), img_gpu, zero_gpu)
        results[name] = result_gpu.cpu().numpy()
    
    return results


def gpu_morphology_close(binary_mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """GPU加速的形态学闭操作
    
    使用卷积模拟膨胀和腐蚀
    """
    if not TORCH_AVAILABLE:
        import cv2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # GPU path using convolution
    mask_gpu = torch.from_numpy(binary_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(TORCH_DEVICE)
    
    # 创建圆形核
    kernel = _create_ellipse_kernel(kernel_size).to(TORCH_DEVICE)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    
    pad = kernel_size // 2
    
    # 膨胀: 如果核范围内有任何1，结果为1
    dilated = torch.nn.functional.conv2d(mask_gpu, kernel, padding=pad)
    dilated = (dilated > 0).float()
    
    # 腐蚀: 如果核范围内全是1，结果为1
    kernel_sum = kernel.sum()
    eroded = torch.nn.functional.conv2d(dilated, kernel, padding=pad)
    result = (eroded >= kernel_sum).float()
    
    return result.squeeze().cpu().numpy().astype(np.uint8)


def _create_ellipse_kernel(size: int) -> torch.Tensor:
    """创建椭圆形核"""
    kernel = torch.zeros((size, size), dtype=torch.float32)
    center = size // 2
    for i in range(size):
        for j in range(size):
            if ((i - center) ** 2 + (j - center) ** 2) <= center ** 2:
                kernel[i, j] = 1.0
    return kernel


def gpu_median_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """GPU加速的中值滤波
    
    注意: PyTorch没有内置中值滤波，使用OpenCV更快
    """
    import cv2
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image, kernel_size)


# ==================== 批量图像生成 ====================

def gpu_generate_layered_images(
    base_images: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """GPU加速的分层图像生成
    
    一次性生成所有 base_image x mask 组合
    
    Args:
        base_images: {'semantic': img, 'depth': img, ...}
        masks: {'foreground': mask, 'middleground': mask, 'background': mask}
        
    Returns:
        results: {'semantic_foreground': img, 'semantic_middleground': img, ...}
    """
    if not TORCH_AVAILABLE:
        # CPU path
        results = {}
        for base_name, base_img in base_images.items():
            for mask_name, mask in masks.items():
                result = np.zeros_like(base_img)
                m = mask.astype(bool)
                result[m] = base_img[m]
                results[f"{base_name}_{mask_name}"] = result
        return results
    
    # GPU path - 批量处理
    results = {}
    
    # 预先将所有掩码传到GPU
    masks_gpu = {name: torch.from_numpy(mask.astype(bool)).to(TORCH_DEVICE) 
                 for name, mask in masks.items()}
    
    for base_name, base_img in base_images.items():
        # 传输基础图像到GPU
        img_gpu = torch.from_numpy(base_img).to(TORCH_DEVICE)
        zero_gpu = torch.zeros_like(img_gpu)
        
        # 对每个掩码应用
        for mask_name, mask_gpu in masks_gpu.items():
            result_gpu = torch.where(mask_gpu.unsqueeze(-1), img_gpu, zero_gpu)
            results[f"{base_name}_{mask_name}"] = result_gpu.cpu().numpy()
    
    return results


# ==================== 统计信息 ====================

def get_gpu_info() -> Dict:
    """获取GPU信息"""
    if not TORCH_AVAILABLE:
        return {'available': False}
    
    return {
        'available': True,
        'device_name': torch.cuda.get_device_name(0),
        'memory_allocated': torch.cuda.memory_allocated(0) / 1024**2,  # MB
        'memory_cached': torch.cuda.memory_reserved(0) / 1024**2,  # MB
    }
