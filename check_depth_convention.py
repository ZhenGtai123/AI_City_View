#!/usr/bin/env python3
"""Check the depth convention used in the depth maps."""
import numpy as np
import cv2
import os
import glob

print("Checking depth convention...")
print()

# Find images with sky
count = 0
for depth_path in glob.glob('output_1/*/depth_map.png'):
    dir_name = os.path.dirname(depth_path)
    semantic_path = os.path.join(dir_name, 'semantic_map.png')
    if not os.path.exists(semantic_path):
        continue
        
    semantic_map = cv2.imread(semantic_path, cv2.IMREAD_GRAYSCALE)
    sky_mask = (semantic_map == 2)  # ADE20K sky class
    sky_pct = np.sum(sky_mask) / sky_mask.size * 100
    
    if sky_pct > 5:  # At least 5% sky
        depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        sky_depths = depth_map[sky_mask]
        non_sky_mask = (semantic_map != 2) & (semantic_map > 0)
        non_sky_depths = depth_map[non_sky_mask]
        
        print(f'Image: {os.path.basename(dir_name)}')
        print(f'  Sky: {sky_pct:.1f}%')
        print(f'  Sky depth mean: {sky_depths.mean():.1f}')
        print(f'  Non-sky depth mean: {non_sky_depths.mean():.1f}')
        
        if sky_depths.mean() > non_sky_depths.mean():
            print(f'  => HIGH depth = FAR (sky is farthest)')
        else:
            print(f'  => LOW depth = FAR (sky is farthest, depth INVERTED!)')
        print()
        
        count += 1
        if count >= 5:
            break

if count == 0:
    # No sky found, check by image position (top vs bottom)
    print("No sky found in images. Checking by position...")
    for depth_path in glob.glob('output_1/*/depth_map.png')[:5]:
        depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        H, W = depth_map.shape
        
        top = depth_map[:H//4, :].mean()
        bottom = depth_map[3*H//4:, :].mean()
        
        print(f'Image: {os.path.basename(os.path.dirname(depth_path))}')
        print(f'  Top (far) depth mean: {top:.1f}')
        print(f'  Bottom (near) depth mean: {bottom:.1f}')
        
        if top > bottom:
            print(f'  => HIGH depth = FAR')
        else:
            print(f'  => LOW depth = FAR (INVERTED!)')
        print()
