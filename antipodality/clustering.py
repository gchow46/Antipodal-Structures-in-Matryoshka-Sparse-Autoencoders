"""
Clustering utils

"""

import numpy as np
from typing import Tuple, List, Union, Optional
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from .utils import MatryoshkaUtils
from .constants import MATRYOSHKA_LEVELS
from .similarity import cosine_matrix


def group_then_cluster_order(
    feature_indices: Union[np.ndarray, List[int]],
    W_enc_dense: np.ndarray,
    clustering_method: str = 'average'
) -> Tuple[List[int], List[int]]:
    """
    Group by Matryoshka level, then cluster within each level
    """
    # Input validation
    feature_indices = np.asarray(feature_indices)
    if len(feature_indices) == 0:
        return [], [0]

    # Get level info for all features
    level_info_list = [MatryoshkaUtils.get_level_info(idx) for idx in feature_indices]

    # Group by level
    level_groups = {}
    for pos, (level_idx, level_size, level, level_color) in enumerate(level_info_list):
        if level not in level_groups:
            level_groups[level] = []
        level_groups[level].append(pos)

    # Order within each level by clustering
    final_order = []
    level_boundaries = []

    for level in MATRYOSHKA_LEVELS:
        if level in level_groups:
            positions = level_groups[level]
            if len(positions) > 1:
                # Cluster within this level
                level_weights = W_enc_dense[positions]
                C_level = cosine_matrix(level_weights)

                # Hierarchical clustering with specified method
                cosine_clean = np.nan_to_num(C_level, nan=1.0)
                distance_matrix = np.clip(1.0 - cosine_clean, 0.0, 2.0)
                condensed_dist = squareform(distance_matrix, checks=False)
                linkage_matrix = linkage(condensed_dist, method=clustering_method)
                leaf_order = leaves_list(linkage_matrix)

                # Map back to original positions
                ordered_positions = [positions[i] for i in leaf_order]
            else:
                ordered_positions = positions

            level_boundaries.append(len(final_order))  # Start of this level
            final_order.extend(ordered_positions)

    level_boundaries.append(len(final_order))  # End of last level
    return final_order, level_boundaries

