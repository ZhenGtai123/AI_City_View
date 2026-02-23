"""
视觉分析Pipeline - 单图处理主入口
处理1张全景图，裁剪为三个视角(left/front/right)，每个视角生成20张输出图片
GPU/CPU流水线: 一个视角做GPU推理时，其他视角的CPU后处理并行执行
"""

import sys
import cv2
import time
from pathlib import Path
from typing import Any, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.stage1_preprocess import crop_panorama_three_views
from pipeline.stage2_ai_inference import stage2_ai_inference
from pipeline.stage3_postprocess import stage3_postprocess
from pipeline.stage4_depth_layering import stage4_depth_layering
from pipeline.stage4_intelligent_fmb import stage4_intelligent_fmb, stage4_metric_fmb
from pipeline.stage5_openness import stage5_openness
from pipeline.stage6_generate_images import stage6_generate_images
from pipeline.stage7_save_outputs import stage7_save_outputs


def _hex_to_bgr(hex_color: str) -> tuple:
    """将十六进制颜色转换为BGR元组"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)  # BGR格式


def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    import json
    config_path = Path(__file__).parent / "Semantic_configuration.json"

    with open(config_path, 'r', encoding='utf-8') as f:
        semantic_config = json.load(f)

    # 转换配置格式：添加bgr字段供stage2使用
    for item in semantic_config:
        if 'color' in item and 'bgr' not in item:
            item['bgr'] = _hex_to_bgr(item['color'])

    return {
        'split_method': 'percentile',
        'semantic_items': semantic_config,

        # === 语义分割 ===
        'enable_semantic': True,

        # === 深度估计配置 ===
        # depth_backend: 'v3' (推荐，DA3NESTED内置天空分割), 'depth_pro', 'v2'
        'depth_backend': 'v3',

        # ---- Depth Anything V3 配置 ----
        # DA3模型选择:
        #   DA3METRIC-LARGE (0.35B) - 规范化深度,需焦距转换,适合8GB VRAM (推荐)
        #   DA3NESTED-GIANT-LARGE-1.1 (1.4B) - 真实米数+天空检测,需16GB+ VRAM
        #   DA3MONO-LARGE (0.35B) - 相对深度,无米数
        'depth_model_id_v3': 'depth-anything/DA3METRIC-LARGE',
        # 焦距 (用于DA3METRIC规范化深度→米数转换)
        # DA3METRIC输出canonical depth at focal=300, 转换: meters = canonical * (focal/300)
        # 对于90° FOV等距柱状裁剪(512px宽): 有效焦距 ≈ w/(2*tan(45°)) = 256
        # 默认300 (=使用原始canonical值，误差约15%)
        'depth_focal_length': 300,
        # 处理分辨率 (必须是14的倍数):
        #   504(快) / 672(平衡) / 1008(高精度) / 1512(最高)
        'depth_process_res': 672,
        'depth_invert_v3': False,
    }


def process_half_image(
    image_data: np.ndarray,
    basename: str,
    output_dir: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    处理单个视角图片（left/front/right中的一个）
    """
    height, width = image_data.shape[:2]

    # Stage1数据（直接使用传入的图片数据）
    stage1_data = {
        'original': image_data,
        'original_copy': image_data.copy(),
        'height': height,
        'width': width,
        'metadata': {
            'basename': basename,
            'width': width,
            'height': height,
        }
    }

    # Stage 2: AI推理（GPU锁在stage2内部，仅包裹推理代码块）
    print("  Stage 2: AI推理 (深度估计)...")
    stage2_result = stage2_ai_inference(stage1_data['original'], config)

    depth_map = stage2_result['depth_map']
    semantic_map = stage2_result['semantic_map']
    depth_metric = stage2_result.get('depth_metric')   # float32 米, 可能为 None
    sky_mask = stage2_result.get('sky_mask')            # bool, 可能为 None

    # Stage 3: 语义后处理
    print("  Stage 3: 语义后处理...")
    stage3_result = stage3_postprocess(semantic_map, config)
    semantic_map_processed = stage3_result['semantic_map_processed']

    # Stage 4: 景深分层
    print("  Stage 4: 景深分层...")
    if depth_metric is not None:
        # 有度量深度 → 使用米数阈值 (0-10m/10-50m/>50m)
        print("    使用度量深度分层 (metric FMB)")
        stage4_result = stage4_metric_fmb(
            depth_metric, config,
            semantic_map=semantic_map,
            sky_mask=sky_mask,
        )
    else:
        # 回退到旧方法
        fmb_method = str(config.get('fmb_method', 'intelligent')).lower()
        if fmb_method == 'intelligent':
            stage4_result = stage4_intelligent_fmb(
                depth_map, config, semantic_map=semantic_map
            )
        else:
            stage4_result = stage4_depth_layering(
                depth_map, config, semantic_map=semantic_map
            )

    # Stage 5: 开放度计算
    print("  Stage 5: 开放度计算...")
    stage5_result = stage5_openness(semantic_map_processed, config)

    foreground_mask = stage4_result['foreground_mask']
    middleground_mask = stage4_result['middleground_mask']
    background_mask = stage4_result['background_mask']

    # Stage 6: 生成图片
    print("  Stage 6: 生成图片...")
    stage6_result = stage6_generate_images(
        stage1_data['original_copy'],
        semantic_map_processed,
        depth_map,
        stage5_result['openness_map'],
        foreground_mask,
        middleground_mask,
        background_mask,
        config,
        depth_metric=depth_metric,
    )

    # 构建增强 metadata (包含深度统计和 FMB 信息)
    enhanced_metadata = dict(stage1_data['metadata'])
    if 'depth_stats' in stage4_result:
        enhanced_metadata['depth_stats'] = stage4_result['depth_stats']
    if 'depth_thresholds' in stage4_result:
        enhanced_metadata['fmb_thresholds'] = stage4_result['depth_thresholds']
    if 'layer_stats' in stage4_result:
        enhanced_metadata['fmb_layer_stats'] = stage4_result['layer_stats']
    # 颜色图例 (固定映射) + 全色阶对应表
    if depth_metric is not None:
        fg_max = float(config.get('fmb_foreground_max', 10.0))
        mg_max = float(config.get('fmb_middleground_max', 50.0))

        # 生成完整色阶对应表: 每1米一个条目 (0-100m) + sky
        # 与 stage6._colorize_depth_metric 使用相同的分段线性插值
        color_scale = []
        max_far = 100.0
        for m in range(int(max_far) + 1):
            d = float(m)
            if d < fg_max:
                t = d / fg_max
                b, g, r = int(0), int(t * 255), int(180 + t * 75)
            elif d < mg_max:
                t = (d - fg_max) / (mg_max - fg_max)
                b, g, r = int(t * 255), 255, int(255 - t * 255)
            else:
                t = min((d - mg_max) / (max_far - mg_max), 1.0)
                b, g, r = int(255 - t * 75), int(255 - t * 255), 0
            b, g, r = max(0, min(255, b)), max(0, min(255, g)), max(0, min(255, r))
            hex_color = f'#{r:02X}{g:02X}{b:02X}'
            color_scale.append({
                'meters': m,
                'rgb': [r, g, b],
                'hex': hex_color,
            })
        color_scale.append({
            'meters': 'sky',
            'rgb': [255, 255, 255],
            'hex': '#FFFFFF',
        })

        enhanced_metadata['depth_color_legend'] = {
            'scheme': 'fixed_metric',
            'segments': [
                {'range': f'0-{fg_max}m', 'label': 'foreground', 'color_start': '#B40000', 'color_end': '#FFFF00'},
                {'range': f'{fg_max}-{mg_max}m', 'label': 'middleground', 'color_start': '#FFFF00', 'color_end': '#00FFFF'},
                {'range': f'>{mg_max}m', 'label': 'background', 'color_start': '#00FFFF', 'color_end': '#B40000'},
                {'range': 'sky', 'label': 'sky', 'color': '#FFFFFF'},
            ],
            'color_scale': color_scale,
        }

    # 额外输出: sky_mask (白=天空, 黑=非天空) + 原始语义 class ID
    output_images = stage6_result['images']
    if sky_mask is not None:
        output_images['sky_mask'] = (sky_mask.astype(np.uint8) * 255)
    output_images['semantic_raw'] = semantic_map  # 原始 class ID (单通道 uint8)

    # Stage 7: 保存输出
    print("  Stage 7: 保存输出...")
    stage7_result = stage7_save_outputs(
        output_images,
        output_dir,
        basename,
        enhanced_metadata,
        depth_metric=depth_metric,
    )

    return {
        'success': True,
        'output_dir': output_dir,
        'saved_files': stage7_result.get('saved_files', [])
    }


def process_panorama(
    image_path: str,
    output_dir: str,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    处理单张全景图：裁剪为三个视角(left/front/right)并分别处理
    """
    if config is None:
        config = get_default_config()

    start_time = time.time()
    image_path_obj = Path(image_path)
    basename = image_path_obj.stem

    print(f"\n{'='*60}")
    print(f"处理全景图: {image_path_obj.name}")
    print(f"{'='*60}")

    # 读取原始图片
    print("步骤1: 读取全景图...")
    original = cv2.imread(str(image_path_obj))
    if original is None:
        return {'success': False, 'error': f'无法读取图片: {image_path}'}

    height, width = original.shape[:2]
    print(f"  原始尺寸: {width}x{height}")

    # 裁剪为三个视角
    print("\n步骤2: 裁剪全景图为三个视角 (left/front/right)...")
    views = crop_panorama_three_views(original)

    for view_name, view_img in views.items():
        print(f"  {view_name}: {view_img.shape[1]}x{view_img.shape[0]}")

    # 流水线并行处理三个视角
    # GPU推理由stage2内部的_GPU_INFERENCE_LOCK串行保护
    # CPU后处理(stage3-7 + 深度归一化)并行执行，不阻塞GPU
    output_path = Path(output_dir)
    view_results = {}
    view_times = {}

    def _process_view(view_name: str, view_img: np.ndarray):
        view_output = output_path / f"{basename}_{view_name}"
        view_output.mkdir(parents=True, exist_ok=True)
        t = time.time()
        result = process_half_image(
            view_img,
            f"{basename}_{view_name}",
            str(view_output),
            config
        )
        elapsed = time.time() - t
        return view_name, str(view_output), elapsed, result

    with ThreadPoolExecutor(max_workers=3) as view_pool:
        futures = {
            view_pool.submit(_process_view, name, img): name
            for name, img in views.items()
        }
        for future in as_completed(futures):
            view_name, view_output, elapsed, result = future.result()
            view_results[view_name] = view_output
            view_times[view_name] = elapsed
            print(f"  {view_name} 视角完成 ({elapsed:.2f}秒)")

    total_time = time.time() - start_time
    print(f"\n全部完成！总耗时: {total_time:.2f}秒")

    return {
        'success': True,
        'output_dir': output_dir,
        **{f'{v}_output': p for v, p in view_results.items()},
        **{f'{v}_time': t for v, t in view_times.items()},
        'total_time': total_time,
    }


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("用法: python main.py <图片路径> <输出目录>")
        print("示例: python main.py input/panorama.jpg output")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = sys.argv[2]

    result = process_panorama(image_path, output_dir)

    if not result['success']:
        print(f"\n处理失败: {result.get('error', 'Unknown error')}")
        sys.exit(1)
