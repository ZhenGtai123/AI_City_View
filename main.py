"""
视觉分析Pipeline - 主入口
处理1张输入图片，生成20张输出图片
"""

import sys
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.stage1_preprocess_single import stage1_preprocess_single as stage1_preprocess
from pipeline.stage2_ai_inference import stage2_ai_inference
from pipeline.stage3_postprocess import stage3_postprocess
from pipeline.stage4_depth_layering import stage4_depth_layering
from pipeline.stage5_openness import stage5_openness
from pipeline.stage6_generate_images import stage6_generate_images
from pipeline.stage7_save_outputs import stage7_save_outputs


def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    s = str(hex_color).strip()
    if s.startswith('#'):
        s = s[1:]
    if len(s) != 6:
        raise ValueError(f"颜色必须是 #RRGGBB 格式，当前={hex_color!r}")
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
    except ValueError as e:
        raise ValueError(f"颜色必须是 #RRGGBB 格式，当前={hex_color!r}") from e
    return (b, g, r)


def load_semantic_configuration(config_path: str | Path) -> Dict[str, Any]:
    """Load Semantic_configuration.json.

    Expected format: a JSON list of objects:
      {"name": str, "color": "#RRGGBB", "openness": 0|1, "countable": 0|1(optional)}

        Returns config keys compatible with the pipeline:
            - semantic_items: List[dict] raw items (name/openness/bgr) for mapping
            - classes: List[str] prompt classes for LangSAM (class_id 1..N)
            - openness_config: List[int] where index == class_id (0 is background)
            - colors: Dict[int, (b,g,r)] OpenCV BGR colors where key == class_id (0 is background)
    """
    p = Path(config_path).expanduser()
    data = json.loads(p.read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise ValueError(f"语义配置必须是 JSON 数组(list)，当前类型={type(data).__name__}")

    # IMPORTANT:
    # - LangSAM prompt loop uses enumerate(classes, start=1) -> class_id 1..N
    # - Background is always semantic id 0 and must NOT be part of the prompt list.
    classes: List[str] = []
    openness_config: List[int] = [0]
    colors: Dict[int, Tuple[int, int, int]] = {0: (0, 0, 0)}
    semantic_items: List[Dict[str, Any]] = []

    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"语义配置第{idx}项必须是对象(dict)，当前={type(item).__name__}")

        name = str(item.get('name', '')).strip()
        if not name:
            raise ValueError(f"语义配置第{idx}项缺少 name")

        openness = item.get('openness', 0)
        try:
            openness_i = int(openness)
        except Exception as e:
            raise ValueError(f"语义配置第{idx}项 openness 必须是0或1，当前={openness!r}") from e
        if openness_i not in (0, 1):
            raise ValueError(f"语义配置第{idx}项 openness 必须是0或1，当前={openness!r}")

        color = item.get('color', None)
        if color is None:
            raise ValueError(f"语义配置第{idx}项缺少 color")
        bgr = _hex_to_bgr(str(color))

        classes.append(name)
        openness_config.append(openness_i)
        colors[len(classes)] = bgr
        semantic_items.append({'name': name, 'openness': openness_i, 'bgr': bgr})

    return {
        'semantic_items': semantic_items,
        'classes': classes,
        'openness_config': openness_config,
        'colors': colors,
    }


def get_default_config() -> dict:
    config: Dict[str, Any] = {
        # 语义分割：OneFormer + ADE20K 150类标准
        # - 输出 semantic_map 为 uint8，类别 ID 为 0..149 (与模型输出一致)
        'semantic_backend': 'oneformer_ade20k',
        'oneformer_model_id': 'shi-labs/oneformer_ade20k_swin_large',
        # Visualization: use ADE20K official palette for OneFormer outputs
        'use_ade20k_palette': True,
        # openness/classes/colors 在 ADE20K 150 类场景下建议按需配置；默认留空表示“未指定”。
        'classes': [],
        'openness_config': [],
        'colors': None,
        'encoder': 'vitb',
        'enable_hole_filling': True,
        'hole_fill_kernel_size': 5,
        'enable_median_blur': True,
        'blur_kernel_size': 5,
        'split_method': 'percentile',
        'fg_ratio': 0.33,
        'bg_ratio': 0.33,
        # PNG压缩等级：0最快(文件大)，9最慢(文件小)
        'png_compression': 3,
        # 性能分析：设置环境变量 PIPELINE_PROFILE=1 启用
        'profile': os.environ.get('PIPELINE_PROFILE', '0') == '1',
    }

    # Optional override: force backend (preferred way to switch between OneFormer/LangSAM)
    backend_override = str(os.environ.get('PIPELINE_SEMANTIC_BACKEND', '')).strip()
    if backend_override:
        config['semantic_backend'] = backend_override

    # If a custom semantic config exists, prefer using it (LangSAM backend).
    # Override path via env: PIPELINE_SEMANTIC_CONFIG=/path/to/Semantic_configuration.json
    semantic_cfg_env = str(os.environ.get('PIPELINE_SEMANTIC_CONFIG', '')).strip()
    semantic_cfg_path = Path(semantic_cfg_env).expanduser() if semantic_cfg_env else (Path(__file__).parent / 'Semantic_configuration.json')
    if semantic_cfg_path.exists():
        try:
            loaded = load_semantic_configuration(semantic_cfg_path)
            # Always keep raw items for OneFormer(ADE20K) name->id mapping.
            config['semantic_items'] = loaded.get('semantic_items', [])
            # For LangSAM backend, also apply prompt-based ids (1..N) colors/openness.
            if str(config.get('semantic_backend', '')).strip().lower() == 'langsam':
                config.update({
                    'classes': loaded.get('classes', []),
                    'openness_config': loaded.get('openness_config', []),
                    'colors': loaded.get('colors', None),
                })
            config['semantic_config_path'] = str(semantic_cfg_path)
        except Exception as e:
            print(f"警告: 读取语义配置失败，将继续使用默认配置: {semantic_cfg_path} ({e})")

    return config

def process_image_pipeline(image_path, output_dir, config):
    """
    完整的7阶段Pipeline
    
    参数:
        image_path: str - 输入图片路径
        output_dir: str - 输出目录
        config: dict - 配置参数
    
    返回:
        result: dict - 处理结果
    """
    import time
    start_time = time.time()
    profile = bool(config.get('profile', False))
    timings = {}

    def _tic(label: str):
        if profile:
            timings[label] = {'start': time.perf_counter()}

    def _toc(label: str):
        if profile and label in timings and 'start' in timings[label]:
            timings[label]['seconds'] = time.perf_counter() - timings[label]['start']
    
    try:
        payload = process_image_pipeline_stage1_to_5(image_path, output_dir, config, _tic=_tic, _toc=_toc)
        print("阶段6: 生成20张输出图片...")
        _tic('stage6')
        stage6_result = stage6_generate_images(
            payload['original_copy'],
            payload['semantic_map_processed'],
            payload['depth_map'],
            payload['openness_map'],
            payload['fg_mask'], payload['mg_mask'], payload['bg_mask'],
            config,
        )
        _toc('stage6')
        all_images = stage6_result['images']
        print(f"  已生成 {len(all_images)} 张图片")

        print("阶段7: 保存输出...")
        _tic('stage7')
        stage7_result = stage7_save_outputs(
            all_images,
            payload['output_dir'],
            payload['image_basename'],
            payload['metadata'],
        )
        _toc('stage7')

        duration = time.time() - start_time
        print(f"\n✅ 处理完成! 耗时: {duration:.2f}秒")
        print(f"输出目录: {payload['output_dir']}")
        print(f"生成文件: {len(stage7_result['saved_files'])} 个")

        if profile:
            print("\n⏱️  分阶段耗时:")
            for k in ['stage1', 'stage2', 'stage3', 'stage4', 'stage5', 'stage6', 'stage7']:
                if k in timings and 'seconds' in timings[k]:
                    print(f"  {k}: {timings[k]['seconds']:.3f}s")

        return {
            'success': True,
            'output_dir': payload['output_dir'],
            'saved_files': stage7_result['saved_files'],
            'metadata': payload['metadata'],
            'duration_seconds': duration,
        }

    except Exception as e:
        import traceback
        error_msg = f"处理失败: {str(e)}\n{traceback.format_exc()}"
        print(f"\n❌ {error_msg}")
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
        }


def process_image_pipeline_stage1_to_5(
    image_path: str,
    output_dir: str,
    config: Dict[str, Any],
    *,
    _tic=None,
    _toc=None,
) -> Dict[str, Any]:
    """Run stages 1..5 and return a payload for deferred stage6+7.

    This is used by batch mode to overlap GPU inference (stage2) with CPU-bound
    generation/saving (stage6/7).
    """
    def _noop(_label: str):
        return None
    tic = _tic or _noop
    toc = _toc or _noop

    import time
    start_time = time.time()

    print(f"开始处理: {image_path}")

    # ========== 阶段1: 预处理 ==========
    print("阶段1: 图片预处理...")
    tic('stage1')
    stage1_result = stage1_preprocess(image_path)
    toc('stage1')
    original = stage1_result['original']
    original_copy = stage1_result['original_copy']
    H, W = stage1_result['height'], stage1_result['width']
    print(f"  图片尺寸: {W}x{H}")

    # ========== 阶段2: AI推理 ==========
    print("阶段2: AI模型推理...")
    tic('stage2')
    stage2_result = stage2_ai_inference(original, config)
    toc('stage2')
    semantic_map = stage2_result['semantic_map']
    depth_map = stage2_result['depth_map']
    print(f"  语义分割完成, 类别数: {semantic_map.max()}")
    print(f"  深度估计完成, 范围: [{depth_map.min()}, {depth_map.max()}]")

    # ========== 阶段3: 后处理 ==========
    print("阶段3: 后处理优化...")
    tic('stage3')
    stage3_result = stage3_postprocess(semantic_map, config)
    toc('stage3')
    semantic_map_processed = stage3_result['semantic_map_processed']
    print(f"  空洞填充: {stage3_result.get('processing_stats', {}).get('holes_filled', 0)} 像素")

    # ========== 阶段4: 景深分层 ==========
    print("阶段4: 景深分层...")
    tic('stage4')
    stage4_result = stage4_depth_layering(depth_map, config)
    toc('stage4')
    fg_mask = stage4_result['foreground_mask']
    mg_mask = stage4_result['middleground_mask']
    bg_mask = stage4_result['background_mask']
    print(f"  前景: {stage4_result['layer_stats']['foreground_percent']:.1f}%")
    print(f"  中景: {stage4_result['layer_stats']['middleground_percent']:.1f}%")
    print(f"  背景: {stage4_result['layer_stats']['background_percent']:.1f}%")

    # ========== 阶段5: 开放度计算 ==========
    print("阶段5: 开放度计算...")
    tic('stage5')
    stage5_result = stage5_openness(semantic_map_processed, config)
    toc('stage5')
    openness_map = stage5_result['openness_map']
    print(f"  开放度: {stage5_result['openness_stats']['openness_ratio']:.1%}")

    metadata = {
        'input': stage1_result['metadata'],
        'config': config,
        'statistics': {
            'semantic': stage3_result.get('processing_stats', {}),
            'layers': stage4_result['layer_stats'],
            'openness': stage5_result['openness_stats'],
        },
        'processing_time': {
            'start': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
            'end': time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_seconds': time.time() - start_time,
        },
    }

    return {
        'output_dir': output_dir,
        'image_basename': Path(image_path).stem,
        'metadata': metadata,
        'original_copy': original_copy,
        'semantic_map_processed': semantic_map_processed,
        'depth_map': depth_map,
        'openness_map': openness_map,
        'fg_mask': fg_mask,
        'mg_mask': mg_mask,
        'bg_mask': bg_mask,
    }


def process_image_pipeline_stage6_to_7(payload: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Run stages 6..7 from a payload produced by stage1..5."""
    stage6_result = stage6_generate_images(
        payload['original_copy'],
        payload['semantic_map_processed'],
        payload['depth_map'],
        payload['openness_map'],
        payload['fg_mask'], payload['mg_mask'], payload['bg_mask'],
        config,
    )
    all_images = stage6_result['images']
    stage7_result = stage7_save_outputs(
        all_images,
        payload['output_dir'],
        payload['image_basename'],
        payload['metadata'],
    )
    return {
        'output_dir': payload['output_dir'],
        'saved_files': stage7_result.get('saved_files', []),
        'metadata': payload['metadata'],
        'success': bool(stage7_result.get('success', False)),
        'errors': stage7_result.get('errors', []),
    }


if __name__ == '__main__':
    config = get_default_config()
    
    # 示例使用
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else 'output'
        
        result = process_image_pipeline(image_path, output_dir, config)
        
        if result['success']:
            print("\n处理成功!")
        else:
            print(f"\n处理失败: {result['error']}")
            sys.exit(1)
    else:
        print("用法: python main.py <图片路径> [输出目录]")
        print("示例: python main.py input/photo.jpg output/")


