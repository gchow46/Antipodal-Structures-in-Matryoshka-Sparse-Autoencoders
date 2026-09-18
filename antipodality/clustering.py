"""Cluster encoder directions within each Matryoshka level."""

import numpy as np
from typing import Tuple, List, Union
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
    """Return row positions ordered by level and clustering, with level boundaries."""
    feature_indices = np.asarray(feature_indices)
    if len(feature_indices) == 0:
        return [], [0]

    level_info = [MatryoshkaUtils.get_level_info(idx) for idx in feature_indices]
    level_groups = {}
    for position, (_, _, level, _) in enumerate(level_info):
        level_groups.setdefault(level, []).append(position)

    final_order = []
    level_boundaries = []
    for level in MATRYOSHKA_LEVELS:
        if level not in level_groups:
            continue
        positions = level_groups[level]
        if len(positions) > 1:
            level_weights = W_enc_dense[positions]
            similarities = cosine_matrix(level_weights)
            cosine_clean = np.nan_to_num(similarities, nan=1.0)
            distance_matrix = np.clip(1.0 - cosine_clean, 0.0, 2.0)
            condensed_dist = squareform(distance_matrix, checks=False)
            linkage_matrix = linkage(condensed_dist, method=clustering_method)
            leaf_order = leaves_list(linkage_matrix)
            ordered_positions = [positions[i] for i in leaf_order]
        else:
            ordered_positions = positions

        level_boundaries.append(len(final_order))
        final_order.extend(ordered_positions)

    level_boundaries.append(len(final_order))
    return final_order, level_boundaries
