"""
批量处理分割后的全景图
对input文件夹中的每张全景图：
1. 裁剪并分割为前后两张图
2. 对每张图分别运行完整的pipeline
"""

import cv2
import sys
from pathlib import Path
from typing import List
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.stage1_preprocess import crop_and_split_panorama
from main import process_image_pipeline, get_default_config


def process_split_image(image_path: str, output_root: str) -> None:
    """
    处理单张全景图：分割并对左右两部分分别运行pipeline
    
    参数:
        image_path: 输入图片路径
        output_root: 输出根目录
    """
    image_path_obj = Path(image_path)
    basename = image_path_obj.stem
    
    print(f"\n{'='*60}")
    print(f"处理: {image_path_obj.name}")
    print(f"{'='*60}")
    
    # 1. 读取原始图片
    print("步骤1: 读取全景图...")
    original = cv2.imread(str(image_path_obj))
    if original is None:
        print(f"❌ 无法读取图片: {image_path}")
        return
    
    height, width = original.shape[:2]
    print(f"  原始尺寸: {width}x{height}")
    
    # 2. 裁剪并分割
    print("步骤2: 裁剪并分割全景图...")
    front_half, back_half = crop_and_split_panorama(original, bottom_crop_ratio=0.3803)
    print(f"  前半部分尺寸: {front_half.shape[1]}x{front_half.shape[0]}")
    print(f"  后半部分尺寸: {back_half.shape[1]}x{back_half.shape[0]}")
    
    # 3. 保存临时分割图片
    output_root_path = Path(output_root)
    temp_dir = output_root_path / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    front_temp_path = temp_dir / f"{basename}_front_half.jpg"
    back_temp_path = temp_dir / f"{basename}_back_half.jpg"
    
    cv2.imwrite(str(front_temp_path), front_half)
    cv2.imwrite(str(back_temp_path), back_half)
    print(f"  临时文件已保存到 {temp_dir}")
    
    # 4. 获取配置
    config = get_default_config()
    
    # 5. 处理前半部分
    print("\n步骤3: 处理前半部分 (0-180°)...")
    front_output_dir = output_root_path / f"{basename}_front"
    front_output_dir.mkdir(parents=True, exist_ok=True)
    try:
        process_image_pipeline(str(front_temp_path), str(front_output_dir), config)
        print(f"  ✅ 前半部分处理完成")
    except Exception as e:
        print(f"  ❌ 前半部分处理失败: {e}")
    
    # 6. 处理后半部分
    print("\n步骤4: 处理后半部分 (180-360°)...")
    back_output_dir = output_root_path / f"{basename}_back"
    back_output_dir.mkdir(parents=True, exist_ok=True)
    try:
        process_image_pipeline(str(back_temp_path), str(back_output_dir), config)
        print(f"  ✅ 后半部分处理完成")
    except Exception as e:
        print(f"  ❌ 后半部分处理失败: {e}")
    
    # 7. 清理临时文件（可选）
    # front_temp_path.unlink()
    # back_temp_path.unlink()
    
    print(f"\n✅ {basename} 处理完成！")


def batch_process_split(input_dir: str, output_root: str, limit: int = 0) -> None:
    """
    批量处理文件夹中的所有全景图
    
    参数:
        input_dir: 输入文件夹路径
        output_root: 输出根目录
        limit: 限制处理图片数量（0表示处理全部）
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"❌ 输入路径不存在: {input_dir}")
        return
    
    # 获取所有图片文件
    image_files: List[Path] = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(input_path.glob(ext))
    
    image_files = sorted(image_files)
    
    if limit > 0:
        image_files = image_files[:limit]
    
    total = len(image_files)
    print(f"\n找到 {total} 张图片待处理")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_root}")
    print(f"{'='*60}\n")
    
    # 处理每张图片
    for idx, img_path in enumerate(image_files, 1):
        print(f"\n进度: [{idx}/{total}]")
        try:
            process_split_image(str(img_path), output_root)
        except Exception as e:
            print(f"❌ 处理 {img_path.name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"✅ 批量处理完成！共处理 {total} 张图片")
    print(f"{'='*60}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='批量处理分割后的全景图')
    parser.add_argument('input', type=str, help='输入文件夹路径')
    parser.add_argument('output', type=str, help='输出根目录')
    parser.add_argument('--limit', type=int, default=0, help='限制处理图片数量（0=全部）')
    
    args = parser.parse_args()
    
    batch_process_split(args.input, args.output, args.limit)
