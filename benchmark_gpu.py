#!/usr/bin/env python3
"""GPU Pipeline Benchmark"""

import time
import numpy as np
import cv2

# Benchmark GPU-accelerated pipeline
depth_path = 'output_1/1001193_118.839948347_32.053621965_front/depth_map.png'
depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
H, W = depth_map.shape
print(f'Image size: {W}x{H} = {W*H:,} pixels')

# Check GPU
from pipeline.gpu_utils import TORCH_AVAILABLE, get_gpu_info
print('GPU available:', TORCH_AVAILABLE)
if TORCH_AVAILABLE:
    info = get_gpu_info()
    print('GPU:', info.get('device_name', 'unknown'))

# Stage 3: Postprocess (morphology) - still CPU, OpenCV is fast
from pipeline.stage3_postprocess import stage3_postprocess
semantic_map = np.random.randint(0, 42, (H, W), dtype=np.uint8)

t0 = time.perf_counter()
for _ in range(10):
    stage3_postprocess(semantic_map, {'enable_hole_filling': True, 'enable_median_blur': True})
t_stage3 = (time.perf_counter() - t0) / 10 * 1000
print(f'Stage 3 (postprocess): {t_stage3:.1f}ms')

# Stage 4: FMB - GPU accelerated
from pipeline.stage4_intelligent_fmb import stage4_intelligent_fmb
config = {
    'fmb_speed': 'fast',
    'enable_kmeans_optimization': True,
    'enable_intelligent_objects': False,
    'enable_hole_filling': False,
    'openness_config': [1] * 150,
}

# Warmup
stage4_intelligent_fmb(depth_map, config, None)

t0 = time.perf_counter()
for _ in range(10):
    stage4_intelligent_fmb(depth_map, config, None)
t_stage4 = (time.perf_counter() - t0) / 10 * 1000
print(f'Stage 4 (FMB): {t_stage4:.1f}ms')

# Stage 5: Openness - GPU accelerated
from pipeline.stage5_openness import stage5_openness
config5 = {'classes': ['sky', 'ground'], 'openness_config': [1, 0] * 21}

# Warmup
stage5_openness(semantic_map, config5)

t0 = time.perf_counter()
for _ in range(100):
    stage5_openness(semantic_map, config5)
t_stage5 = (time.perf_counter() - t0) / 100 * 1000
print(f'Stage 5 (openness): {t_stage5:.2f}ms')

# Stage 6: Generate images - GPU accelerated
from pipeline.stage6_generate_images import stage6_generate_images
fg = np.random.rand(H, W) > 0.5
mg = np.random.rand(H, W) > 0.7
bg = ~(fg | mg)
original = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
openness = np.random.randint(0, 2, (H, W), dtype=np.uint8) * 255

# Warmup
stage6_generate_images(original, semantic_map, depth_map, openness, fg, mg, bg, {})

t0 = time.perf_counter()
for _ in range(10):
    stage6_generate_images(original, semantic_map, depth_map, openness, fg, mg, bg, {})
t_stage6 = (time.perf_counter() - t0) / 10 * 1000
print(f'Stage 6 (generate images): {t_stage6:.1f}ms')

print()
print('=' * 50)
print('Total (Stage 3-6): %.1fms' % (t_stage3 + t_stage4 + t_stage5 + t_stage6))
