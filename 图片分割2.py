import cv2
import numpy as np
import os


def crop_panorama_4096x2048(equirect_img, bottom_crop_ratio=0.3803):
    """
    针对4096x2048像素的全景图进行裁剪：
    1. 舍弃下半部分38.03%
    2. 从中间对半分开

    参数:
    - equirect_img: 4096x2048像素的全景图
    - bottom_crop_ratio: 要舍弃的下半部分比例（默认0.3803即38.03%）

    返回:
    - front_half: 前半部分图像
    - back_half: 后半部分图像
    - cropped_img: 裁剪后的完整图像
    """
    # 验证输入尺寸
    height, width = equirect_img.shape[:2]
    if width != 4096 or height != 2048:
        print(f"警告: 图像尺寸为 {width}x{height}，预期为 4096x2048")

    # 步骤1: 计算要保留的高度（舍弃下半部分38.03%）
    keep_height = int(height * (1 - bottom_crop_ratio))
    print(f"原始尺寸: {width}x{height}")
    print(f"舍弃下半部分 {bottom_crop_ratio * 100:.2f}% 后，保留高度: {keep_height} 像素")

    # 裁剪掉下半部分
    cropped_img = equirect_img[:keep_height, :]
    print(f"裁剪后尺寸: {cropped_img.shape[1]}x{cropped_img.shape[0]}")

    # 步骤2: 将裁剪后的图像从中间对半分开
    new_height, new_width = cropped_img.shape[:2]
    mid_point = new_width // 2

    # 前半部分 (0-180度)
    front_half = cropped_img[:, :mid_point]
    # 后半部分 (180-360度)
    back_half = cropped_img[:, mid_point:]

    print(f"前半部分尺寸: {front_half.shape[1]}x{front_half.shape[0]}")
    print(f"后半部分尺寸: {back_half.shape[1]}x{back_half.shape[0]}")

    return front_half, back_half, cropped_img


def save_cropped_results(front_half, back_half, cropped_img, input_filename, output_folder="output"):
    """
    保存裁剪结果，保持原始像素尺寸，文件名包含原始图片名称

    参数:
    - input_filename: 原始图片的文件名（包含扩展名）
    """
    os.makedirs(output_folder, exist_ok=True)

    # 获取不带扩展名的文件名
    base_name = os.path.splitext(os.path.basename(input_filename))[0]

    # 保存裁剪后的完整图像
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_cropped_panorama.jpg"), cropped_img)

    # 保存前半部分
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_front_half.jpg"), front_half)

    # 保存后半部分
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_back_half.jpg"), back_half)

    print(f"所有结果已保存到 {output_folder} 文件夹")
    print(f"- {base_name}_cropped_panorama.jpg")
    print(f"- {base_name}_front_half.jpg")
    print(f"- {base_name}_back_half.jpg")


# 批量处理input文件夹中的所有图片
input_folder = 'input'
output_folder = 'output'

# 获取input文件夹中的所有jpg文件
if os.path.exists(input_folder):
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"找到 {len(image_files)} 张图片待处理\n")
    
    for idx, filename in enumerate(image_files, 1):
        input_path = os.path.join(input_folder, filename)
        print(f"\n{'='*60}")
        print(f"处理进度: [{idx}/{len(image_files)}] - {filename}")
        print(f"{'='*60}")
        
        # 读取图片
        equirect_img = cv2.imread(input_path)
        
        if equirect_img is None:
            print(f"错误: 无法读取图片 {filename}")
            continue
        
        # 执行裁剪（裁掉下方38.03%）
        front_half, back_half, cropped_img = crop_panorama_4096x2048(
            equirect_img,
            bottom_crop_ratio=0.3803
        )
        
        # 保存结果
        save_cropped_results(front_half, back_half, cropped_img, filename, output_folder)
    
    print(f"\n{'='*60}")
    print(f"✓ 全部完成！共处理 {len(image_files)} 张图片")
    print(f"{'='*60}")
else:
    print(f"错误: 找不到 {input_folder} 文件夹")