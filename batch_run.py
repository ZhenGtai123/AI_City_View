"""
批量处理全景图（支持多线程）
对input文件夹中的每张全景图：
1. 自动裁剪为三个视角 (left/front/right)，每个90° FOV
2. 对每个视角分别运行完整的pipeline
3. 每张全景图内部3个视角已自动流水线并行(GPU/CPU重叠)
4. --workers控制同时处理几张全景图

推荐参数 (7800X3D + 4070):
  --workers=2  批量处理最佳（6个视角线程，GPU持续满载，CPU ~6核）
  --workers=1  单张处理默认（3个视角线程，GPU/CPU流水线）
"""

import sys
import time
import threading
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from main import process_panorama, get_default_config


def batch_process_panoramas(
    input_dir: str, 
    output_root: str, 
    limit: int = 0,
    workers: int = 1
) -> None:
    """
    批量处理文件夹中的所有全景图（支持多线程）
    
    参数:
        input_dir: 输入文件夹路径
        output_root: 输出根目录
        limit: 限制处理图片数量（0表示处理全部）
        workers: 并行worker数量（1=单线程）
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"[FAIL] 输入路径不存在: {input_dir}")
        return
    
    # 获取所有图片文件（去重，Windows上大小写不敏感会重复匹配）
    image_files: List[Path] = []
    seen = set()
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        for f in input_path.glob(ext):
            key = str(f).lower()
            if key not in seen:
                seen.add(key)
                image_files.append(f)

    image_files = sorted(image_files)
    
    if limit > 0:
        image_files = image_files[:limit]
    
    total = len(image_files)
    print(f"\n找到 {total} 张全景图待处理")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_root}")
    if workers > 1:
        print(f">> 多线程模式：{workers} 个并行worker")
    print(f"{'='*60}\n")
    
    # 获取配置
    config = get_default_config()
    
    success_count = 0
    fail_count = 0
    start_time = time.time()
    
    def process_single(img_path: Path, idx: int) -> dict:
        """处理单张图片的wrapper函数"""
        try:
            print(f"\n[{idx}/{total}] 开始: {img_path.name}")
            result = process_panorama(str(img_path), output_root, config)
            if result['success']:
                print(f"[{idx}/{total}] [OK] 完成: {img_path.name} ({result['total_time']:.2f}秒)")
                return {'success': True, 'name': img_path.name, 'time': result['total_time']}
            else:
                print(f"[{idx}/{total}] [FAIL] 失败: {img_path.name} - {result.get('error', 'Unknown')}")
                return {'success': False, 'name': img_path.name, 'error': result.get('error')}
        except Exception as e:
            print(f"[{idx}/{total}] [FAIL] 异常: {img_path.name} - {e}")
            return {'success': False, 'name': img_path.name, 'error': str(e)}
    
    if workers > 1:
        # 多线程处理
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_single, img_path, idx): (img_path, idx)
                for idx, img_path in enumerate(image_files, 1)
            }
            
            for future in as_completed(futures):
                result = future.result()
                if result['success']:
                    success_count += 1
                else:
                    fail_count += 1
    else:
        # 单线程处理
        for idx, img_path in enumerate(image_files, 1):
            result = process_single(img_path, idx)
            if result['success']:
                success_count += 1
            else:
                fail_count += 1
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"[OK] 批量处理完成！")
    print(f"  成功: {success_count}/{total}")
    print(f"  失败: {fail_count}/{total}")
    print(f"  总耗时: {total_time:.2f}秒")
    if success_count > 0:
        print(f"  平均耗时: {total_time/total:.2f}秒/张")
    print(f"{'='*60}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='批量处理全景图（自动分割，支持多线程）')
    parser.add_argument('input', type=str, help='输入文件夹路径')
    parser.add_argument('output', type=str, help='输出根目录')
    parser.add_argument('--limit', type=int, default=0, help='限制处理图片数量（0=全部）')
    parser.add_argument('--workers', type=int, default=1,
                        help='同时处理几张全景图（默认1，推荐2用于批量处理）')
    
    args = parser.parse_args()
    
    batch_process_panoramas(args.input, args.output, args.limit, args.workers)
