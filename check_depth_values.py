"""
诊断脚本：检查 Depth Anything 3 的深度输出
用于调试天空深度值问题
"""
import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

def check_depth_output(image_path: str):
    """检查单张图片的深度输出"""
    import torch
    from depth_anything_3.api import DepthAnything3

    print(f"\n{'='*60}")
    print(f"检查图片: {image_path}")
    print(f"{'='*60}")

    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图片")
        return

    H, W = img.shape[:2]
    print(f"图片尺寸: {W}x{H}")

    # 加载模型
    print("\n加载 DA3NESTED-GIANT-LARGE-1.1 模型...")
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print(f"设备: {device}")

    # 推理
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    print("\n运行深度估计...")
    with torch.inference_mode():
        prediction = model.inference(
            [pil_img],
            export_dir=None,
            export_format='mini_npz',
            process_res=672,
        )
        depth = prediction.depth[0]
        depth = depth.cpu().numpy() if hasattr(depth, 'cpu') else np.array(depth)

    print(f"\n深度图尺寸: {depth.shape}")
    print(f"深度值范围: [{depth.min():.4f}, {depth.max():.4f}]")
    print(f"深度均值: {depth.mean():.4f}")

    # 分析不同区域
    # 假设图片上半部分更可能是天空
    h_third = H // 3
    top_region = cv2.resize(depth, (W, H))[:h_third, :]
    bottom_region = cv2.resize(depth, (W, H))[-h_third:, :]

    print(f"\n区域分析 (假设上部=天空, 下部=地面):")
    print(f"  上部1/3 深度: [{top_region.min():.4f}, {top_region.max():.4f}], 均值={top_region.mean():.4f}")
    print(f"  下部1/3 深度: [{bottom_region.min():.4f}, {bottom_region.max():.4f}], 均值={bottom_region.mean():.4f}")

    if top_region.mean() > bottom_region.mean():
        print("\n✅ 上部深度 > 下部深度 (符合预期: 天空=远, 地面=近)")
        print("   建议: depth_invert_v3 = False")
    else:
        print("\n⚠️  上部深度 < 下部深度 (可能需要反转)")
        print("   建议: depth_invert_v3 = True")

    # 保存深度可视化
    output_dir = Path(image_path).parent / "depth_debug"
    output_dir.mkdir(exist_ok=True)

    # 归一化到 0-255
    depth_resized = cv2.resize(depth, (W, H))
    depth_norm = (depth_resized - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)

    # 保存不同版本
    cv2.imwrite(str(output_dir / "depth_raw_normalized.png"), depth_uint8)
    cv2.imwrite(str(output_dir / "depth_raw_inverted.png"), 255 - depth_uint8)
    cv2.imwrite(str(output_dir / "depth_inferno.png"), cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO))
    cv2.imwrite(str(output_dir / "depth_inferno_inverted.png"), cv2.applyColorMap(255 - depth_uint8, cv2.COLORMAP_INFERNO))

    print(f"\n已保存调试图片到: {output_dir}/")
    print("  - depth_raw_normalized.png  (原始归一化)")
    print("  - depth_raw_inverted.png    (反转)")
    print("  - depth_inferno.png         (INFERNO色彩)")
    print("  - depth_inferno_inverted.png (INFERNO反转)")
    print("\n请检查哪个版本的天空显示为最亮(黄色)，那个就是正确的设置")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python check_depth_values.py <图片路径>")
        sys.exit(1)

    check_depth_output(sys.argv[1])
