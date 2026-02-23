"""pipeline.stage4_intelligent_fmb

阶段4: 智能FMB景深分层
功能: 基于理论的51%-26%-23%分布，K-means边界优化，智能连通域处理，孔洞填充

基于 fmb_v21_forced_sky_background.ipynb 算法
支持多线程并行处理提升性能

性能说明:
  后处理阶段(百分位/K-means/掩码)使用纯numpy/sklearn
  GPU仅用于stage2的神经网络推理，不在此阶段使用
  7800X3D的96MB V-cache使numpy在<1M元素时比GPU+PCIe传输更快
"""

from __future__ import annotations

import numpy as np
import cv2
from scipy import ndimage
from sklearn.cluster import MiniBatchKMeans
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any, Set
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# 全局线程池（复用以减少开销）
_thread_pool: Optional[ThreadPoolExecutor] = None

def _get_thread_pool(max_workers: int = 4) -> ThreadPoolExecutor:
    """获取或创建线程池"""
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(max_workers=max_workers)
    return _thread_pool
# ==================== 数据结构定义 ====================

@dataclass
class ObjectFeatures:
    """对象的综合特征描述"""
    num_pixels: int
    semantic_class: int
    is_closed: bool
    is_countable: bool
    effective_aspect_ratio: float
    bbox_aspect_ratio: float
    compactness: float
    elongation: float
    depth_mean: float
    depth_std: float
    depth_range: float
    vertical_depth_gradient: float
    horizontal_depth_gradient: float
    centroid_y: float
    centroid_x: float
    bottom_y: int
    relative_y_position: float
    touching_ground: bool
    touching_sky: bool
    neighbor_classes: List[int]
    edge_density: float
    internal_structure_complexity: float


# ==================== 强制语义规则 ====================

class ForcedSemanticRules:
    """强制语义规则系统 - 某些类别总是分配到特定层"""

    def __init__(self, openness_config: List[int], class_names: Optional[List[str]] = None):
        self.openness_config = openness_config
        self.class_names = class_names or []
        self.forced_background_classes: Set[int] = set()
        self.forced_foreground_classes: Set[int] = set()
        self.forced_middleground_classes: Set[int] = set()

        self._initialize_forced_rules()

    def _initialize_forced_rules(self):
        """基于类别名称初始化强制规则"""
        # ADE20K: sky=2
        self.forced_background_classes.add(2)  # sky

        # 如果有类名列表，扫描更多
        for idx, name in enumerate(self.class_names):
            name_lower = name.lower()
            # 天空、海洋等总是背景
            if 'sky' in name_lower:
                self.forced_background_classes.add(idx)
            elif 'sea' in name_lower and 'seat' not in name_lower:
                self.forced_background_classes.add(idx)

    def get_forced_layer(self, semantic_class: int) -> Optional[int]:
        """获取强制层（0=前景, 1=中景, 2=背景），如果没有强制规则返回None"""
        if semantic_class in self.forced_background_classes:
            return 2
        elif semantic_class in self.forced_foreground_classes:
            return 0
        elif semantic_class in self.forced_middleground_classes:
            return 1
        return None

    def is_forced(self, semantic_class: int) -> bool:
        return self.get_forced_layer(semantic_class) is not None


# ==================== 智能孔洞填充 ====================

class IntelligentHoleFilling:
    """智能孔洞填充系统（支持多线程）"""

    min_hole_size = 10
    max_hole_size = 5000
    depth_threshold_ratio = 0.15

    def __init__(self, depth_map: np.ndarray, fmb_map: np.ndarray):
        self.depth_map = depth_map
        self.fmb_map = fmb_map
        self.H, self.W = depth_map.shape
        self.neighbor_radius = 5

    def process(self) -> Tuple[np.ndarray, Dict]:
        """处理所有层的孔洞（多线程并行版）"""
        filled_map = self.fmb_map.copy()
        fill_info = {
            'total_holes_detected': 0,
            'holes_filled': 0,
            'holes_preserved': 0,
        }

        max_holes_per_layer = 50

        def process_layer(layer: int) -> List[Tuple[np.ndarray, int, bool]]:
            """处理单层的孔洞（可并行）"""
            results = []
            holes = self._detect_holes_in_layer(filled_map, layer, max_holes=max_holes_per_layer)

            for hole_mask in holes:
                hole_size = np.sum(hole_mask)
                if hole_size < self.min_hole_size or hole_size > self.max_hole_size:
                    continue

                should_fill, _ = self._analyze_hole(hole_mask, layer)
                results.append((hole_mask, layer, should_fill))

            return results

        # 并行处理3个层
        pool = _get_thread_pool(max_workers=3)
        futures = [pool.submit(process_layer, layer) for layer in [0, 1, 2]]

        all_results = []
        for future in futures:
            all_results.extend(future.result())

        # 应用结果
        fill_info['total_holes_detected'] = len(all_results)
        for hole_mask, layer, should_fill in all_results:
            if should_fill:
                filled_map[hole_mask] = layer
                fill_info['holes_filled'] += 1
            else:
                fill_info['holes_preserved'] += 1

        return filled_map, fill_info

    def _detect_holes_in_layer(self, fmb_map: np.ndarray, layer: int, max_holes: int = 50) -> List[np.ndarray]:
        """检测层中的孔洞（高度优化版 - 使用批量处理）"""
        layer_mask = (fmb_map == layer).astype(np.uint8)

        # 使用形态学操作填充小孔洞
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        layer_closed = cv2.morphologyEx(layer_mask, cv2.MORPH_CLOSE, kernel)

        # 获取所有潜在孔洞区域
        inverted = 1 - layer_closed
        num_labels, labeled, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)

        if num_labels <= 1:
            return []

        h, w = fmb_map.shape

        # 过滤候选孔洞并按面积排序
        candidates = []
        for label_id in range(1, num_labels):
            x, y, width, height, area = stats[label_id]

            # 过滤条件
            if area < self.min_hole_size or area > self.max_hole_size:
                continue
            if x == 0 or y == 0 or x + width >= w or y + height >= h:
                continue

            candidates.append((label_id, area))

        if not candidates:
            return []

        # 按面积排序，只处理最大的max_holes个
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:max_holes]

        # 批量生成孔洞掩码
        holes = []
        for label_id, _ in candidates:
            hole_mask = (labeled == label_id)
            holes.append(hole_mask)

        return holes

    def _analyze_hole(self, hole_mask: np.ndarray, surrounding_layer: int) -> Tuple[bool, Dict]:
        """分析孔洞是否应该填充"""
        hole_depths = self.depth_map[hole_mask]
        hole_mean_depth = np.mean(hole_depths)
        hole_std_depth = np.std(hole_depths)

        kernel = np.ones((self.neighbor_radius * 2 + 1, self.neighbor_radius * 2 + 1), np.uint8)
        dilated = cv2.dilate(hole_mask.astype(np.uint8), kernel, iterations=1)
        neighbor_mask = (dilated > 0) & (~hole_mask) & (self.fmb_map == surrounding_layer)

        if not neighbor_mask.any():
            return False, {'reason': 'no_valid_neighbors'}

        neighbor_depths = self.depth_map[neighbor_mask]
        neighbor_mean_depth = np.mean(neighbor_depths)
        neighbor_std_depth = np.std(neighbor_depths)
        depth_difference = abs(hole_mean_depth - neighbor_mean_depth)

        all_depths_min = min(hole_depths.min(), neighbor_depths.min())
        all_depths_max = max(hole_depths.max(), neighbor_depths.max())
        local_depth_range = max(all_depths_max - all_depths_min, 1)

        normalized_difference = depth_difference / local_depth_range
        should_fill = normalized_difference < self.depth_threshold_ratio

        if hole_std_depth > neighbor_std_depth * 2:
            should_fill = False

        return should_fill, {'depth_difference': depth_difference}


# ==================== 智能层决策系统 ====================

class IntelligentLayerDecisionSystem:
    """智能集成层决策系统"""

    def __init__(self, depth_map: np.ndarray, semantic_map: np.ndarray,
                 fmb_map: np.ndarray, openness_config: List[int],
                 countability_config: List[int], forced_rules: ForcedSemanticRules):
        self.depth_map = depth_map
        self.semantic_map = semantic_map
        self.fmb_map = fmb_map
        self.openness = openness_config
        self.countability = countability_config
        self.H, self.W = depth_map.shape
        self.forced_rules = forced_rules

        self.strategy_weights = {
            'bottom_sampling': 0.3,
            'horizontal_analysis': 0.3,
            'vertical_structure': 0.2,
            'spatial_context': 0.1,
            'depth_consistency': 0.1
        }

    def extract_object_features(self, region_mask: np.ndarray) -> Optional[ObjectFeatures]:
        """提取对象的综合特征"""
        y_coords, x_coords = np.where(region_mask)
        if len(y_coords) == 0:
            return None

        num_pixels = len(y_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        x_min, x_max = np.min(x_coords), np.max(x_coords)

        semantic_values = self.semantic_map[region_mask]
        semantic_class = int(np.bincount(semantic_values).argmax())
        is_closed = self.openness[semantic_class] == 1 if semantic_class < len(self.openness) else False
        is_countable = self.countability[semantic_class] == 1 if semantic_class < len(self.countability) else False

        unique_rows = len(np.unique(y_coords))
        unique_cols = len(np.unique(x_coords))
        effective_aspect_ratio = unique_rows / unique_cols if unique_cols > 0 else 1.0
        bbox_aspect_ratio = (y_max - y_min + 1) / (x_max - x_min + 1) if (x_max - x_min + 1) > 0 else 1.0
        compactness = num_pixels / ((y_max - y_min + 1) * (x_max - x_min + 1)) if (y_max - y_min + 1) * (x_max - x_min + 1) > 0 else 0

        if num_pixels > 5:
            coords = np.column_stack((x_coords - np.mean(x_coords), y_coords - np.mean(y_coords)))
            cov_matrix = np.cov(coords.T)
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            elongation = np.sqrt(max(eigenvalues) / (min(eigenvalues) + 1e-6))
        else:
            elongation = 1.0

        depths = self.depth_map[region_mask].astype(float)
        depth_mean = np.mean(depths)
        depth_std = np.std(depths)
        depth_range = np.max(depths) - np.min(depths)

        return ObjectFeatures(
            num_pixels=num_pixels, semantic_class=semantic_class,
            is_closed=is_closed, is_countable=is_countable,
            effective_aspect_ratio=effective_aspect_ratio, bbox_aspect_ratio=bbox_aspect_ratio,
            compactness=compactness, elongation=elongation,
            depth_mean=depth_mean, depth_std=depth_std, depth_range=depth_range,
            vertical_depth_gradient=0.0, horizontal_depth_gradient=0.0,
            centroid_y=np.mean(y_coords), centroid_x=np.mean(x_coords),
            bottom_y=y_max, relative_y_position=np.mean(y_coords) / self.H,
            touching_ground=(y_max >= self.H - 5), touching_sky=(y_min <= 5),
            neighbor_classes=[], edge_density=0.0, internal_structure_complexity=0.0
        )

    def intelligent_layer_decision(self, region_mask: np.ndarray) -> Tuple[Optional[int], float, Dict]:
        """智能层决策的主函数"""
        features = self.extract_object_features(region_mask)
        if features is None:
            return None, 0, {}

        # 首先检查强制规则
        if self.forced_rules.is_forced(features.semantic_class):
            forced_layer = self.forced_rules.get_forced_layer(features.semantic_class)
            return forced_layer, 1.0, {'reason': 'forced_semantic_rule'}

        # 对于开放类（非closed），使用K-means原始标签
        if not features.is_closed:
            dominant_label = self._get_dominant_kmeans_label(region_mask)
            return dominant_label, 0.95, {'reason': 'open_object'}

        # 对于闭合对象，使用多策略决策
        strategies_results = {}
        strategies_results['bottom'] = self._bottom_sampling_strategy(region_mask, features)
        strategies_results['depth'] = self._depth_consistency_strategy(region_mask, features)
        strategies_results['spatial'] = self._spatial_context_strategy(features)

        return self._intelligent_fusion(strategies_results, features)

    def _get_dominant_kmeans_label(self, region_mask: np.ndarray) -> int:
        """获取区域中最主要的K-means标签"""
        labels = self.fmb_map[region_mask]
        unique_labels, counts = np.unique(labels, return_counts=True)
        return int(unique_labels[np.argmax(counts)])

    def _bottom_sampling_strategy(self, region_mask: np.ndarray, features: ObjectFeatures) -> Dict:
        """底部采样策略（向量化优化）"""
        y_coords, x_coords = np.where(region_mask)

        if features.num_pixels < 100:
            bottom_ratio = 0.25
        elif features.num_pixels < 500:
            bottom_ratio = 0.15
        else:
            bottom_ratio = 0.10

        num_bottom_pixels = max(int(features.num_pixels * bottom_ratio), min(10, features.num_pixels))
        y_sorted_indices = np.argsort(y_coords)[::-1][:num_bottom_pixels]

        bottom_labels = self.fmb_map[y_coords[y_sorted_indices], x_coords[y_sorted_indices]]

        if len(bottom_labels) > 0:
            layer_counts = np.bincount(bottom_labels, minlength=3)
            best_layer = int(np.argmax(layer_counts))
            confidence = layer_counts[best_layer] / len(bottom_labels)
        else:
            best_layer = 1
            confidence = 0.0

        return {'layer': best_layer, 'confidence': confidence}

    def _depth_consistency_strategy(self, region_mask: np.ndarray, features: ObjectFeatures) -> Dict:
        """深度一致性策略"""
        layer_depth_profiles = self._get_layer_depth_profiles()
        match_scores = {}

        for layer, (depth_mean, depth_std) in layer_depth_profiles.items():
            z_score = abs(features.depth_mean - depth_mean) / (depth_std + 1e-6)
            match_scores[layer] = np.exp(-0.5 * z_score ** 2)

        if features.depth_std > 20:
            for layer in match_scores:
                match_scores[layer] *= 0.7

        best_layer = max(match_scores, key=match_scores.get)
        confidence = match_scores[best_layer]

        return {'layer': best_layer, 'confidence': confidence}

    def _spatial_context_strategy(self, features: ObjectFeatures) -> Dict:
        """空间上下文策略"""
        if features.relative_y_position < 0.3:
            suggested_layer = 2
            confidence = 0.5
        elif features.relative_y_position > 0.7:
            suggested_layer = 0
            confidence = 0.5
        else:
            suggested_layer = 1
            confidence = 0.4

        if features.touching_ground and features.relative_y_position > 0.7:
            suggested_layer = 0
            confidence = 0.7

        return {'layer': suggested_layer, 'confidence': confidence}

    def _get_layer_depth_profiles(self) -> Dict[int, Tuple[float, float]]:
        """获取每层的深度统计"""
        profiles = {}
        for layer in [0, 1, 2]:
            layer_mask = self.fmb_map == layer
            if np.sum(layer_mask) > 0:
                layer_depths = self.depth_map[layer_mask].astype(float)
                profiles[layer] = (np.mean(layer_depths), np.std(layer_depths))
            else:
                # 默认值
                profiles[layer] = [(50, 30), (128, 40), (200, 30)][layer]
        return profiles

    def _intelligent_fusion(self, strategies_results: Dict, features: ObjectFeatures) -> Tuple[int, float, Dict]:
        """智能融合多个策略的结果"""
        layer_votes = {0: 0.0, 1: 0.0, 2: 0.0}

        strategy_weights = {
            'bottom': 0.4,
            'depth': 0.35,
            'spatial': 0.25
        }

        for strategy_name, result in strategies_results.items():
            layer = result['layer']
            confidence = result['confidence']
            weight = strategy_weights.get(strategy_name, 0.2)
            layer_votes[layer] += confidence * weight

        total_votes = sum(layer_votes.values())
        if total_votes > 0:
            for layer in layer_votes:
                layer_votes[layer] /= total_votes

        final_layer = max(layer_votes, key=layer_votes.get)
        final_confidence = layer_votes[final_layer]

        return final_layer, final_confidence, {'layer_votes': layer_votes}


# ==================== 主函数 ====================

def stage4_intelligent_fmb(
    depth_map: np.ndarray,
    config: Dict[str, Any],
    semantic_map: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """阶段4: 智能FMB景深分层

    基于理论的51%-26%-23%分布，使用K-means边界优化

    参数:
        depth_map: (H, W) uint8, 值范围 [0, 255]，0=近，255=远
        config: 配置参数
        semantic_map: (H, W) uint8, 语义分割图

    返回:
        包含 foreground_mask, middleground_mask, background_mask 等的字典
    """
    if depth_map.ndim != 2:
        raise ValueError(f"depth_map 必须是二维 (H, W)，当前 shape={depth_map.shape}")

    H, W = depth_map.shape
    total_pixels = H * W
    depth_float = depth_map.astype(np.float32)

    # 获取配置
    enable_kmeans = config.get('enable_kmeans_optimization', True)
    enable_intelligent_objects = config.get('enable_intelligent_objects', True)
    enable_hole_filling = config.get('enable_hole_filling', True)

    openness_config = config.get('openness_config', [])
    countability_config = config.get('countability_config', [])
    class_names = config.get('classes', [])

    # 有效像素掩码 (depth=0是合法的最近前景值，不能排除)
    valid_mask = depth_float >= 0
    depth_valid = depth_float[valid_mask]

    if len(depth_valid) == 0:
        empty_mask = np.zeros((H, W), dtype=bool)
        full_mask = np.ones((H, W), dtype=bool)
        return {
            'foreground_mask': empty_mask,
            'middleground_mask': empty_mask,
            'background_mask': full_mask,
            'depth_thresholds': {'P33': 85.0, 'P66': 170.0},
            'layer_stats': {
                'foreground_pixels': 0, 'middleground_pixels': 0, 'background_pixels': total_pixels,
                'foreground_percent': 0.0, 'middleground_percent': 0.0, 'background_percent': 100.0
            }
        }

    # 理论分布: 前景51%, 中景26%, 背景23%
    # 深度约定: LOW depth = NEAR (foreground), HIGH depth = FAR (background)
    depth_min = float(depth_valid.min())
    depth_max = float(depth_valid.max())

    # numpy排序 + 百分位 (7800X3D V-cache下比GPU传输更快)
    depth_sorted = np.sort(depth_valid)
    idx1 = int(len(depth_sorted) * 0.51)
    idx2 = int(len(depth_sorted) * 0.77)
    thresh1 = float(depth_sorted[min(idx1, len(depth_sorted) - 1)])
    thresh2 = float(depth_sorted[min(idx2, len(depth_sorted) - 1)])

    # 处理阈值合并的边界情况
    depth_range = depth_max - depth_min
    min_gap = depth_range * 0.05

    if thresh2 - thresh1 < min_gap:
        thresh1 = depth_min + depth_range * 0.33
        thresh2 = depth_min + depth_range * 0.66

    # K-means边界优化 (纯CPU路径)
    if enable_kmeans and len(depth_valid) > 100:
        fmb_map = _kmeans_fmb_segmentation(depth_float, valid_mask, depth_valid, thresh1, thresh2)
    else:
        fmb_map = np.full((H, W), 2, dtype=np.uint8)
        fmb_map[depth_float < thresh1] = 0
        fmb_map[(depth_float >= thresh1) & (depth_float < thresh2)] = 1

    kmeans_original = fmb_map.copy()

    # 初始化强制语义规则
    forced_rules = ForcedSemanticRules(openness_config, class_names)

    # 应用强制语义规则（如天空强制背景）- 向量化
    forced_corrections = 0
    if semantic_map is not None:
        unique_classes = np.unique(semantic_map)
        for sem_class in unique_classes:
            if sem_class == 0:
                continue
            forced_layer = forced_rules.get_forced_layer(sem_class)
            if forced_layer is not None:
                class_mask = semantic_map == sem_class
                changed = np.sum(fmb_map[class_mask] != forced_layer)
                fmb_map[class_mask] = forced_layer
                forced_corrections += changed

    # 智能连通域处理（对闭合对象）- 多线程并行优化版
    intelligent_adjustments = 0
    min_region_size = config.get('min_intelligent_region_size', 50)
    max_regions_per_class = config.get('max_regions_per_class', 100)

    if enable_intelligent_objects and semantic_map is not None and len(openness_config) > 0:
        decision_system = IntelligentLayerDecisionSystem(
            depth_float, semantic_map, fmb_map, openness_config,
            countability_config or [0] * len(openness_config), forced_rules
        )

        unique_classes = np.unique(semantic_map)
        closed_classes = [c for c in unique_classes
                         if c != 0 and c < len(openness_config)
                         and openness_config[c] == 1
                         and not forced_rules.is_forced(c)]

        def process_class_regions(sem_class: int) -> List[Tuple[np.ndarray, int]]:
            """处理单个语义类的所有区域（可并行）"""
            results = []
            class_mask = (semantic_map == sem_class).astype(np.uint8)
            labeled_array, num_features = ndimage.label(class_mask)

            if num_features == 0:
                return results

            if num_features > max_regions_per_class:
                region_sizes = ndimage.sum(class_mask, labeled_array, range(1, num_features + 1))
                top_regions = np.argsort(region_sizes)[::-1][:max_regions_per_class] + 1
            else:
                top_regions = range(1, num_features + 1)

            for region_id in top_regions:
                region_mask = labeled_array == region_id
                pixel_count = np.sum(region_mask)

                if pixel_count < min_region_size:
                    continue

                layer, confidence, _ = decision_system.intelligent_layer_decision(region_mask)
                if layer is not None and confidence > 0.5:
                    results.append((region_mask, layer))

            return results

        # 使用线程池并行处理各个语义类
        if len(closed_classes) > 1:
            pool = _get_thread_pool(max_workers=4)
            futures = [pool.submit(process_class_regions, c) for c in closed_classes]
            all_results = []
            for future in futures:
                all_results.extend(future.result())
        else:
            all_results = []
            for c in closed_classes:
                all_results.extend(process_class_regions(c))

        # 应用所有调整
        for region_mask, layer in all_results:
            fmb_map[region_mask] = layer
            intelligent_adjustments += 1

    # 智能孔洞填充（多线程并行处理各层）
    hole_fill_info = None
    if enable_hole_filling:
        hole_filler = IntelligentHoleFilling(depth_float, fmb_map)
        fmb_map, hole_fill_info = hole_filler.process()

    # 生成最终掩码（向量化）
    foreground_mask = (fmb_map == 0)
    middleground_mask = (fmb_map == 1)
    background_mask = (fmb_map == 2)

    fg_pixels = int(np.sum(foreground_mask))
    mg_pixels = int(np.sum(middleground_mask))
    bg_pixels = int(np.sum(background_mask))

    def _percent(x: int) -> float:
        return (x / total_pixels * 100.0) if total_pixels > 0 else 0.0

    result = {
        'foreground_mask': foreground_mask.astype(bool),
        'middleground_mask': middleground_mask.astype(bool),
        'background_mask': background_mask.astype(bool),
        'fmb_map': fmb_map,  # 0=前景, 1=中景, 2=背景
        'kmeans_original': kmeans_original,
        'depth_thresholds': {
            'P33': float(thresh1),
            'P66': float(thresh2),
        },
        'layer_stats': {
            'foreground_pixels': fg_pixels,
            'middleground_pixels': mg_pixels,
            'background_pixels': bg_pixels,
            'foreground_percent': float(_percent(fg_pixels)),
            'middleground_percent': float(_percent(mg_pixels)),
            'background_percent': float(_percent(bg_pixels)),
        },
        'intelligent_stats': {
            'forced_corrections': forced_corrections,
            'intelligent_adjustments': intelligent_adjustments,
            'hole_filling': hole_fill_info,
        }
    }

    return result


def _kmeans_fmb_segmentation(
    depth_map: np.ndarray,
    valid_mask: np.ndarray,
    depth_valid: np.ndarray,
    thresh1: float,
    thresh2: float
) -> np.ndarray:
    """使用K-means优化FMB边界 (纯CPU，sklearn MiniBatchKMeans)"""
    H, W = depth_map.shape

    # 特征工程
    n_features = 2
    features = np.zeros((len(depth_valid), n_features), dtype=np.float32)

    initial_labels = np.where(depth_valid < thresh1, 0.0,
                              np.where(depth_valid < thresh2, 1.0, 2.0))
    features[:, 0] = initial_labels

    boundary_range = (np.max(depth_valid) - np.min(depth_valid)) * 0.08
    sigma = boundary_range / 3

    dist_to_thresh1 = np.abs(depth_valid - thresh1)
    dist_to_thresh2 = np.abs(depth_valid - thresh2)
    min_boundary_dist = np.minimum(dist_to_thresh1, dist_to_thresh2)

    boundary_sensitivity = np.zeros_like(depth_valid, dtype=np.float32)
    in_range = min_boundary_dist <= boundary_range
    if sigma > 0:
        boundary_sensitivity[in_range] = np.exp(-min_boundary_dist[in_range]**2 / (2 * sigma**2))
    features[:, 1] = boundary_sensitivity

    weights = np.array([0.95, 0.05], dtype=np.float32)
    weighted_features = features * weights

    centers = np.array([[0, 0], [0.95, 0], [1.9, 0]], dtype=np.float32)

    try:
        kmeans = MiniBatchKMeans(
            n_clusters=3,
            init=centers,
            n_init=1,
            max_iter=30,
            random_state=42,
            batch_size=min(1024, len(weighted_features)),
            reassignment_ratio=0.01
        )
        cluster_labels = kmeans.fit_predict(weighted_features)
    except Exception:
        cluster_labels = initial_labels.astype(np.int32)

    cluster_depths = np.array([
        np.mean(depth_valid[cluster_labels == i]) if np.any(cluster_labels == i) else 0.0
        for i in range(3)
    ])
    depth_order = np.argsort(cluster_depths)

    label_remap = np.zeros(3, dtype=np.uint8)
    for new_label, old_label in enumerate(depth_order):
        label_remap[old_label] = new_label

    fmb_map = np.full((H, W), 2, dtype=np.uint8)
    mapped_labels = label_remap[cluster_labels]
    fmb_map[valid_mask] = mapped_labels

    return fmb_map


# ==================== 度量深度 FMB (新) ====================

def stage4_metric_fmb(
    depth_metric: np.ndarray,
    config: Dict[str, Any],
    semantic_map: Optional[np.ndarray] = None,
    sky_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """阶段4: 基于度量深度(米)的FMB分层

    逐像素按距离阈值分层，不使用连通域分析和强规则。
    树跨越多个深度时，自然按像素级别分到不同层。

    参数:
        depth_metric: (H, W) float32, 单位米, 天空=inf
        config: 配置参数
            - fmb_foreground_max: 前景最大距离 (默认 10m)
            - fmb_middleground_max: 中景最大距离 (默认 50m)
        semantic_map: (H, W) uint8, 语义图 (可选, 用于统计)
        sky_mask: (H, W) bool, 天空掩码 (可选)

    返回:
        dict: foreground_mask, middleground_mask, background_mask, depth_stats 等
    """
    if depth_metric.ndim != 2:
        raise ValueError(f"depth_metric 必须是二维 (H, W)，当前 shape={depth_metric.shape}")

    H, W = depth_metric.shape
    total_pixels = H * W

    # 可配置阈值
    fg_max = float(config.get('fmb_foreground_max', 10.0))    # 0 ~ 10m
    mg_max = float(config.get('fmb_middleground_max', 50.0))  # 10 ~ 50m

    # 天空掩码: sky_mask 参数 或 inf 值
    is_sky = np.isinf(depth_metric)
    if sky_mask is not None:
        is_sky = is_sky | sky_mask

    # 逐像素分层
    fmb_map = np.full((H, W), 2, dtype=np.uint8)  # 默认背景
    fmb_map[depth_metric < fg_max] = 0              # 前景
    fmb_map[(depth_metric >= fg_max) & (depth_metric < mg_max)] = 1  # 中景
    # depth >= mg_max 或 inf(天空) 保持为 2 (背景)

    # 生成掩码
    foreground_mask = (fmb_map == 0)
    middleground_mask = (fmb_map == 1)
    background_mask = (fmb_map == 2)

    fg_pixels = int(np.sum(foreground_mask))
    mg_pixels = int(np.sum(middleground_mask))
    bg_pixels = int(np.sum(background_mask))
    sky_pixels = int(np.sum(is_sky))

    def _pct(x: int) -> float:
        return (x / total_pixels * 100.0) if total_pixels > 0 else 0.0

    # 深度统计 (排除天空)
    finite_depth = depth_metric[np.isfinite(depth_metric)]
    if len(finite_depth) > 0:
        depth_stats = {
            'min_meters': float(np.min(finite_depth)),
            'max_meters': float(np.max(finite_depth)),
            'mean_meters': float(np.mean(finite_depth)),
            'median_meters': float(np.median(finite_depth)),
            'p10_meters': float(np.percentile(finite_depth, 10)),
            'p25_meters': float(np.percentile(finite_depth, 25)),
            'p75_meters': float(np.percentile(finite_depth, 75)),
            'p90_meters': float(np.percentile(finite_depth, 90)),
        }
    else:
        depth_stats = {
            'min_meters': 0.0, 'max_meters': 0.0,
            'mean_meters': 0.0, 'median_meters': 0.0,
            'p10_meters': 0.0, 'p25_meters': 0.0,
            'p75_meters': 0.0, 'p90_meters': 0.0,
        }

    return {
        'foreground_mask': foreground_mask,
        'middleground_mask': middleground_mask,
        'background_mask': background_mask,
        'fmb_map': fmb_map,
        'depth_thresholds': {
            'foreground_max_meters': fg_max,
            'middleground_max_meters': mg_max,
        },
        'layer_stats': {
            'foreground_pixels': fg_pixels,
            'middleground_pixels': mg_pixels,
            'background_pixels': bg_pixels,
            'sky_pixels': sky_pixels,
            'foreground_percent': _pct(fg_pixels),
            'middleground_percent': _pct(mg_pixels),
            'background_percent': _pct(bg_pixels),
            'sky_percent': _pct(sky_pixels),
        },
        'depth_stats': depth_stats,
    }
