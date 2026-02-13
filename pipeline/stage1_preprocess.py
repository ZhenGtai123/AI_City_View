"""
阶段1: 图片预处理
功能: 读取全景图文件，基于等距柱状投影几何，裁剪为三个视角(left/front/right)
算法: 基于 Yin & Wang (2016) 的等距柱状投影裁剪方法
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

# ============================================================================
# 裁剪参数配置 - 基于角度/比例 (v2.0)
# ============================================================================
HORIZONTAL_FOV = 90     # 水平视角: 90° (每个裁剪框的宽度)
VERTICAL_UP = 45        # 向上角度: 45° (从地平线向上)
VERTICAL_DOWN = 10      # 向下角度: 10° (从地平线向下)

# 三个视角的朝向角度 (相对于前方 0°)
VIEW_DIRECTIONS = {
    'left':  -90,   # 左转 90°
    'front':   0,   # 正前方
    'right':  90,   # 右转 90°
}

# 自动计算比例参数（基于等距柱状投影几何特性）
# 水平方向：360° = 100% 宽度
CROP_WIDTH_RATIO = HORIZONTAL_FOV / 360.0  # 0.25 (25%)

# 垂直方向：180° = 100% 高度，地平线在中央 (90°)
CROP_TOP_RATIO = (90 - VERTICAL_UP) / 180.0       # 0.25
CROP_BOTTOM_RATIO = (90 + VERTICAL_DOWN) / 180.0  # 0.5556
CROP_HEIGHT_RATIO = CROP_BOTTOM_RATIO - CROP_TOP_RATIO  # 0.3056

# 每个视角的水平中心位置（比例）
VIEW_CENTER_RATIOS = {
    view: 0.5 + (angle / 360.0)
    for view, angle in VIEW_DIRECTIONS.items()
}

# 视角元数据
VIEW_METADATA = {
    'left':  {'view_angle': '-135° ~ -45°', 'direction': '-90°'},
    'front': {'view_angle': '-45° ~ +45°',  'direction': '0°'},
    'right': {'view_angle': '+45° ~ +135°', 'direction': '+90°'},
}


def calculate_crop_regions(width: int, height: int) -> Dict[str, Dict[str, int]]:
    """
    根据图像尺寸和比例参数，计算三个视角的裁剪区域

    参数:
        width: 全景图宽度（像素）
        height: 全景图高度（像素）

    返回:
        dict: 三个视角的裁剪坐标 {view: {x_start, x_end, y_start, y_end}}
    """
    crop_width = round(width * CROP_WIDTH_RATIO)
    crop_top = round(height * CROP_TOP_RATIO)
    crop_bottom = round(height * CROP_BOTTOM_RATIO)

    crop_regions = {}
    for view, center_ratio in VIEW_CENTER_RATIOS.items():
        center_x = round(width * center_ratio)
        x_start = center_x - crop_width // 2
        x_end = x_start + crop_width

        crop_regions[view] = {
            'x_start': x_start,
            'x_end': x_end,
            'y_start': crop_top,
            'y_end': crop_bottom,
        }

    return crop_regions


def crop_panorama_three_views(equirect_img: np.ndarray) -> Dict[str, np.ndarray]:
    """
    将等距柱状投影全景图裁剪为三个视角 (left, front, right)

    基于 Yin & Wang (2016) 的方法:
    - 水平FOV: 90° per view
    - 垂直范围: 45° up + 10° down = 55°
    - 三个视角: left(-90°), front(0°), right(+90°)

    参数:
        equirect_img: 全景图像 (BGR格式, numpy数组)

    返回:
        dict: {'left': np.ndarray, 'front': np.ndarray, 'right': np.ndarray}
    """
    height, width = equirect_img.shape[:2]
    crop_regions = calculate_crop_regions(width, height)

    views = {}
    for view, region in crop_regions.items():
        cropped = equirect_img[
            region['y_start']:region['y_end'],
            region['x_start']:region['x_end']
        ].copy()
        views[view] = cropped

    return views


def stage1_preprocess(image_path: str) -> Dict[str, Any]:
    """
    阶段1: 图片预处理 - 裁剪全景图为三个视角

    参数:
        image_path: str - 输入图片路径

    返回:
        dict - 包含以下键:
            - views: dict - {'left': np.ndarray, 'front': np.ndarray, 'right': np.ndarray}
            - view_metadata: dict - 每个视角的元数据
            - original_metadata: dict - 原始图片元数据
            - view_names: list - ['left', 'front', 'right']

    异常:
        FileNotFoundError: 文件不存在
        ValueError: 无法读取图片或图片格式不正确
    """
    # ========== 步骤1: 读取图片文件 ==========
    image_path_obj = Path(image_path)

    if not image_path_obj.exists():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

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

    # ========== 步骤3: 裁剪为三个视角 ==========
    views = crop_panorama_three_views(original)

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
        'crop_method': 'three_view_v2',
        'horizontal_fov': HORIZONTAL_FOV,
        'vertical_up': VERTICAL_UP,
        'vertical_down': VERTICAL_DOWN,
    }

    # ========== 步骤5: 生成每个视角的元数据 ==========
    view_names = list(VIEW_DIRECTIONS.keys())
    view_metadata = {}

    for view_name in view_names:
        img = views[view_name]
        vh, vw = img.shape[:2]
        meta = VIEW_METADATA[view_name]
        view_metadata[view_name] = {
            'view': view_name,
            'view_angle': meta['view_angle'],
            'direction': meta['direction'],
            'width': vw,
            'height': vh,
            'size_str': f"{vw}x{vh}",
            'total_pixels': vh * vw,
            'channels': 3,
            'dtype': str(img.dtype),
            'basename': f"{basename}_{view_name}",
        }

    # ========== 返回结果 ==========
    result = {
        'views': views,
        'view_metadata': view_metadata,
        'original_metadata': original_metadata,
        'view_names': view_names,
    }

    _validate_stage1_result(result)

    return result


def _validate_stage1_result(result: Dict[str, Any]) -> None:
    """验证阶段1的输出结果"""
    views = result['views']
    view_metadata = result['view_metadata']
    original_metadata = result['original_metadata']
    view_names = result['view_names']

    assert len(views) == 3, f"必须有3个视角, 当前: {len(views)}"
    assert set(view_names) == {'left', 'front', 'right'}, f"视角名称不正确: {view_names}"

    # 检查每个视角
    ref_shape = None
    for view_name in view_names:
        img = views[view_name]
        meta = view_metadata[view_name]

        assert img is not None, f"{view_name} 不能为 None"
        assert len(img.shape) == 3, f"{view_name} 必须是3维数组"
        assert img.shape[2] == 3, f"{view_name} 通道数必须是3, 当前值: {img.shape[2]}"
        assert img.dtype == np.uint8, f"{view_name} dtype 必须是 uint8, 当前值: {img.dtype}"

        # 检查尺寸一致性
        if ref_shape is None:
            ref_shape = img.shape
        else:
            assert img.shape == ref_shape, f"{view_name} 形状 {img.shape} 与其他视角 {ref_shape} 不一致"

        # 检查元数据
        required_keys = ['view', 'view_angle', 'width', 'height', 'size_str', 'total_pixels', 'basename']
        for key in required_keys:
            assert key in meta, f"{view_name} metadata 缺少字段: {key}"

        assert meta['width'] == img.shape[1], f"{view_name} width 不一致"
        assert meta['height'] == img.shape[0], f"{view_name} height 不一致"

    # 检查原始元数据
    for key in ['filename', 'basename', 'original_width', 'original_height']:
        assert key in original_metadata, f"original_metadata 缺少字段: {key}"


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
        try:
            result = stage1_preprocess(test_image_path)
            print("阶段1测试成功!")
            print(f"\n原始图片: {result['original_metadata']['original_size']}")
            print(f"  - 文件名: {result['original_metadata']['filename']}")
            print(f"  - 总像素: {result['original_metadata']['total_pixels']:,}")

            for view_name in result['view_names']:
                meta = result['view_metadata'][view_name]
                print(f"\n{view_name.upper()} 视角 ({meta['view_angle']}):")
                print(f"  - 尺寸: {meta['size_str']}")
                print(f"  - 总像素: {meta['total_pixels']:,}")

            save_test = input("\n是否保存测试结果到output文件夹? (y/n): ")
            if save_test.lower() == 'y':
                output_dir = Path('output')
                output_dir.mkdir(exist_ok=True)

                for view_name in result['view_names']:
                    meta = result['view_metadata'][view_name]
                    out_path = output_dir / f"{meta['basename']}.jpg"
                    cv2.imwrite(str(out_path), result['views'][view_name])
                    print(f"  - {out_path}")

        except Exception as e:
            print(f"测试失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("用法: python stage1_preprocess.py <图片路径>")
        print("示例: python stage1_preprocess.py input/test_image.jpg")
