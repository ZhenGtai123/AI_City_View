"""
阶段1: 图片预处理
功能: 读取图片文件，裁剪并分割全景图，生成左右两张图片
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple


def crop_and_split_panorama(equirect_img: np.ndarray, bottom_crop_ratio: float = 0.3803) -> Tuple[np.ndarray, np.ndarray]:
    """
    裁剪并分割全景图
    
    参数:
        equirect_img: 全景图像 (BGR格式)
        bottom_crop_ratio: 要舍弃的下半部分比例（默认0.3803即38.03%）
    
    返回:
        front_half: 前半部分图像 (0-180度)
        back_half: 后半部分图像 (180-360度)
    """
    height, width = equirect_img.shape[:2]
    
    # 步骤1: 裁剪掉下方部分
    keep_height = int(height * (1 - bottom_crop_ratio))
    cropped_img = equirect_img[:keep_height, :]
    
    # 步骤2: 从中间对半分开
    new_width = cropped_img.shape[1]
    mid_point = new_width // 2
    
    # 前半部分 (0-180度)
    front_half = cropped_img[:, :mid_point].copy()
    # 后半部分 (180-360度)
    back_half = cropped_img[:, mid_point:].copy()
    
    return front_half, back_half


def stage1_preprocess(image_path: str) -> Dict[str, Any]:
    """
    阶段1: 图片预处理
    
    参数:
        image_path: str - 输入图片路径
    
    返回:
        dict - 包含以下键:
            - front_half: np.ndarray - 前半部分图片 (0-180度) BGR格式
            - back_half: np.ndarray - 后半部分图片 (180-360度) BGR格式
            - front_metadata: dict - 前半部分元数据
            - back_metadata: dict - 后半部分元数据
            - original_metadata: dict - 原始图片元数据
    
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
    
    # ========== 步骤2: 验证图片格式 ==========
    height, width, channels = original.shape
    
    if channels != 3:
        raise ValueError(f"图片必须是3通道 (RGB/BGR), 当前通道数: {channels}")
    
    if original.dtype != np.uint8:
        raise ValueError(f"图片必须是8位格式, 当前类型: {original.dtype}")
    
    if height <= 0 or width <= 0:
        raise ValueError(f"图片尺寸无效: {width}x{height}")
    
    # ========== 步骤3: 裁剪并分割全景图 ==========
    front_half, back_half = crop_and_split_panorama(original, bottom_crop_ratio=0.3803)
    
    # ========== 步骤4: 生成原始图片元数据 ==========
    filename = image_path_obj.name
    basename = image_path_obj.stem
    extension = image_path_obj.suffix
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    original_metadata = {
        'filename': filename,
        'basename': basename,
        'extension': extension,
        'original_size': f"{width}x{height}",
        'original_width': width,
        'original_height': height,
        'total_pixels': height * width,
        'channels': channels,
        'dtype': str(original.dtype),
        'timestamp': timestamp,
        'file_path': str(image_path_obj.absolute()),
        'crop_ratio': 0.3803
    }
    
    # ========== 步骤5: 生成前半部分元数据 ==========
    front_height, front_width = front_half.shape[:2]
    front_metadata = {
        'view': 'front',
        'view_angle': '0-180°',
        'width': front_width,
        'height': front_height,
        'size_str': f"{front_width}x{front_height}",
        'total_pixels': front_height * front_width,
        'channels': 3,
        'dtype': str(front_half.dtype),
        'basename': f"{basename}_front_half"
    }
    
    # ========== 步骤6: 生成后半部分元数据 ==========
    back_height, back_width = back_half.shape[:2]
    back_metadata = {
        'view': 'back',
        'view_angle': '180-360°',
        'width': back_width,
        'height': back_height,
        'size_str': f"{back_width}x{back_height}",
        'total_pixels': back_height * back_width,
        'channels': 3,
        'dtype': str(back_half.dtype),
        'basename': f"{basename}_back_half"
    }
    
    # ========== 返回结果 ==========
    result = {
        'front_half': front_half,
        'back_half': back_half,
        'front_metadata': front_metadata,
        'back_metadata': back_metadata,
        'original_metadata': original_metadata
    }
    
    # ========== 质量检查 ==========
    _validate_stage1_result(result)
    
    return result


def _validate_stage1_result(result: Dict[str, Any]) -> None:
    """
    验证阶段1的输出结果
    
    参数:
        result: dict - 阶段1的输出结果
    
    异常:
        AssertionError: 验证失败
    """
    front_half = result['front_half']
    back_half = result['back_half']
    front_metadata = result['front_metadata']
    back_metadata = result['back_metadata']
    original_metadata = result['original_metadata']
    
    # 检查图片成功加载
    assert front_half is not None, "front_half 不能为 None"
    assert back_half is not None, "back_half 不能为 None"
    
    # 检查形状
    assert len(front_half.shape) == 3, "front_half 必须是3维数组"
    assert len(back_half.shape) == 3, "back_half 必须是3维数组"
    
    # 检查通道数
    assert front_half.shape[2] == 3, f"front_half 通道数必须是3, 当前值: {front_half.shape[2]}"
    assert back_half.shape[2] == 3, f"back_half 通道数必须是3, 当前值: {back_half.shape[2]}"
    
    # 检查数据类型
    assert front_half.dtype == np.uint8, f"front_half dtype 必须是 uint8, 当前值: {front_half.dtype}"
    assert back_half.dtype == np.uint8, f"back_half dtype 必须是 uint8, 当前值: {back_half.dtype}"
    
    # 检查尺寸一致性（前后半部分应该尺寸相同）
    assert front_half.shape == back_half.shape, "front_half 和 back_half 形状必须一致"
    
    # 检查元数据完整性
    front_keys = ['view', 'view_angle', 'width', 'height', 'size_str', 'total_pixels', 'basename']
    for key in front_keys:
        assert key in front_metadata, f"front_metadata 缺少必要字段: {key}"
        assert key in back_metadata, f"back_metadata 缺少必要字段: {key}"
    
    # 检查元数据值
    assert front_metadata['width'] == front_half.shape[1], "front_metadata 中的 width 与图片实际宽度不一致"
    assert front_metadata['height'] == front_half.shape[0], "front_metadata 中的 height 与图片实际高度不一致"
    assert back_metadata['width'] == back_half.shape[1], "back_metadata 中的 width 与图片实际宽度不一致"
    assert back_metadata['height'] == back_half.shape[0], "back_metadata 中的 height 与图片实际高度不一致"
    
    # 检查原始元数据
    original_keys = ['filename', 'basename', 'original_width', 'original_height']
    for key in original_keys:
        assert key in original_metadata, f"original_metadata 缺少必要字段: {key}"


if __name__ == '__main__':
    # 测试代码
    import sys
    
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
        try:
            result = stage1_preprocess(test_image_path)
            print("✅ 阶段1测试成功!")
            print(f"\n原始图片: {result['original_metadata']['original_size']}")
            print(f"  - 文件名: {result['original_metadata']['filename']}")
            print(f"  - 总像素: {result['original_metadata']['total_pixels']:,}")
            print(f"\n前半部分 ({result['front_metadata']['view_angle']}):")
            print(f"  - 尺寸: {result['front_metadata']['size_str']}")
            print(f"  - 总像素: {result['front_metadata']['total_pixels']:,}")
            print(f"\n后半部分 ({result['back_metadata']['view_angle']}):")
            print(f"  - 尺寸: {result['back_metadata']['size_str']}")
            print(f"  - 总像素: {result['back_metadata']['total_pixels']:,}")
            
            # 可选: 保存测试结果
            save_test = input("\n是否保存测试结果到output文件夹? (y/n): ")
            if save_test.lower() == 'y':
                from pathlib import Path
                output_dir = Path('output')
                output_dir.mkdir(exist_ok=True)
                
                front_path = output_dir / f"{result['front_metadata']['basename']}.jpg"
                back_path = output_dir / f"{result['back_metadata']['basename']}.jpg"
                
                cv2.imwrite(str(front_path), result['front_half'])
                cv2.imwrite(str(back_path), result['back_half'])
                
                print(f"✅ 已保存:")
                print(f"  - {front_path}")
                print(f"  - {back_path}")
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("用法: python stage1_preprocess.py <图片路径>")
        print("示例: python stage1_preprocess.py input/test_image.jpg")


