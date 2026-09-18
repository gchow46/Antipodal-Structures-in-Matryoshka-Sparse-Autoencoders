"""Weight-space antipodality and density statistics."""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr

from .similarity import normalize_weights, blocked_pair_scores, cosine_matrix
from .constants import MATRYOSHKA_LEVELS, LEVEL_RANGES
from .utils import MatryoshkaUtils
from .clustering import group_then_cluster_order


def compute_antipodality_scores(
    W_enc: np.ndarray,
    W_dec: np.ndarray,
    feature_indices: Optional[np.ndarray] = None,
    top_k: int = 1,
    block_size: int = 2048,
    antipodal_only: bool = True,
) -> Dict[str, np.ndarray | Dict[str, float]]:
    """Score features against the selected pool; partner indices are positions in that pool."""
    with torch.no_grad():
        if feature_indices is None:
            indices = np.arange(W_enc.shape[0], dtype=np.int64)
        else:
            indices = np.asarray(feature_indices, dtype=np.int64)

        E, D = normalize_weights(W_enc, W_dec, indices)
        scores, partners = blocked_pair_scores(E, D, top_k, block_size, antipodal_only)
        summary_stats = validate_scores(scores)
        return {
            "feature_indices": indices,
            "antipodality_scores": scores,
            "antipodal_partners": partners,
            "summary_stats": summary_stats,
        }


def validate_scores(scores: np.ndarray) -> Dict[str, float]:
    """Summarize finite scores, returning zeros when none are available."""
    scores = np.asarray(scores)
    finite_scores = scores[np.isfinite(scores)]
    if len(finite_scores) == 0:
        return {
            'count': 0,
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0
        }

    return {
        'count': int(len(finite_scores)),
        'mean': float(np.mean(finite_scores)),
        'std': float(np.std(finite_scores)),
        'min': float(np.min(finite_scores)),
        'max': float(np.max(finite_scores)),
        'median': float(np.median(finite_scores))
    }


def dense_feature_indices(densities: np.ndarray, threshold: float) -> Tuple[np.ndarray, Dict[int, Dict[str, float]]]:
    """Return indices strictly above the threshold and counts by exclusive level."""
    densities = np.asarray(densities)
    dense_mask = densities > threshold
    dense_indices = np.where(dense_mask)[0]

    level_stats = {}
    for i, level in enumerate(MATRYOSHKA_LEVELS):
        level_range = LEVEL_RANGES[i]
        level_mask = (dense_indices >= level_range[0]) & (dense_indices < level_range[1])
        level_dense = dense_indices[level_mask]
        total_in_level = level_range[1] - level_range[0]
        level_stats[level] = {
            'dense_count': int(len(level_dense)),
            'total_count': int(total_in_level),
            'dense_fraction': float(len(level_dense) / total_in_level if total_in_level > 0 else 0.0),
            'mean_density': float(np.mean(densities[level_dense])) if len(level_dense) > 0 else 0.0
        }

    return dense_indices, level_stats


def dense_sparse_means(
    densities: np.ndarray,
    scores: np.ndarray,
    threshold: float
) -> Dict[str, float | int | None]:
    """Count dense/sparse features and average their finite scores."""
    valid_mask = np.isfinite(scores)
    dense_mask = densities > threshold
    dense_valid_mask = dense_mask & valid_mask
    sparse_valid_mask = (~dense_mask) & valid_mask
    dense_scores = scores[dense_valid_mask]
    sparse_scores = scores[sparse_valid_mask]

    return {
        'dense_count': int(np.sum(dense_mask)),
        'sparse_count': int(np.sum(~dense_mask)),
        'dense_mean_antipodality': float(np.mean(dense_scores)) if len(dense_scores) > 0 else None,
        'sparse_mean_antipodality': float(np.mean(sparse_scores)) if len(sparse_scores) > 0 else None
    }


def spearman_corr(
    densities: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, float | int | None]:
    """Return Spearman correlation and the dense/sparse score summary."""
    means_dict = dense_sparse_means(densities, scores, threshold)
    rho, p = spearmanr(densities, scores)
    result = {'spearman_r': rho, 'spearman_p': p}
    result.update(means_dict)
    return result


def analyze_matryoshka_hierarchy(
    indices: np.ndarray,
    scores: np.ndarray,
    partners: np.ndarray,
) -> Dict[str, object]:
    """Summarize scores within each exclusive Matryoshka level."""
    indices, scores = np.asarray(indices), np.asarray(scores)
    feature_levels = [MatryoshkaUtils.get_level(idx) for idx in indices]
    level_analysis = {}
    for level in MATRYOSHKA_LEVELS:
        level_range = LEVEL_RANGES[MATRYOSHKA_LEVELS.index(level)]
        level_mask = (indices >= level_range[0]) & (indices < level_range[1])
        if np.any(level_mask):
            level_scores = scores[level_mask]
            finite_level_scores = level_scores[np.isfinite(level_scores)]
            level_analysis[level] = {
                'count': int(np.sum(level_mask)),
                'mean_antipodality': float(np.mean(finite_level_scores)) if len(finite_level_scores) > 0 else 0.0,
                'std_antipodality': float(np.std(finite_level_scores)) if len(finite_level_scores) > 0 else 0.0,
                'median_antipodality': float(np.median(finite_level_scores)) if len(finite_level_scores) > 0 else 0.0
            }

    return {'level_analysis': level_analysis, 'feature_levels': feature_levels}


def find_top_pairs(
    indices: np.ndarray,
    scores: np.ndarray,
    partners: np.ndarray,
    top_k: int = 10,
    W_enc: Optional[np.ndarray] = None,
    W_dec: Optional[np.ndarray] = None,
) -> List[Dict[str, object]]:
    """Deduplicate partner pairs by original SAE indices, keeping the highest score."""
    indices, scores, partners = np.asarray(indices), np.asarray(scores), np.asarray(partners)
    valid_mask = np.isfinite(scores) & (partners >= 0) & (partners < len(indices))
    valid_indices = indices[valid_mask]
    valid_scores = scores[valid_mask]
    valid_partners = partners[valid_mask]
    if len(valid_indices) == 0:
        return []

    pairs_dict = {}
    for feat_idx, score, partner_pos in zip(valid_indices, valid_scores, valid_partners):
        partner_idx = indices[partner_pos]
        pair_key = (min(feat_idx, partner_idx), max(feat_idx, partner_idx))
        if pair_key not in pairs_dict or score > pairs_dict[pair_key]['antipodality_score']:
            pair_info = {
                'feature1_idx': int(feat_idx),
                'feature2_idx': int(partner_idx),
                'feature1_level': MatryoshkaUtils.get_level(feat_idx),
                'feature2_level': MatryoshkaUtils.get_level(partner_idx),
                'antipodality_score': float(score)
            }
            if W_enc is not None and W_dec is not None:
                with torch.no_grad():
                    enc_sim = float(torch.cosine_similarity(
                        torch.from_numpy(W_enc[feat_idx:feat_idx+1]).float(),
                        torch.from_numpy(W_enc[partner_idx:partner_idx+1]).float(),
                        dim=1
                    ))
                    dec_sim = float(torch.cosine_similarity(
                        torch.from_numpy(W_dec[feat_idx:feat_idx+1]).float(),
                        torch.from_numpy(W_dec[partner_idx:partner_idx+1]).float(),
                        dim=1
                    ))
                    pair_info['encoder_similarity'] = enc_sim
                    pair_info['decoder_similarity'] = dec_sim
            pairs_dict[pair_key] = pair_info

    sorted_pairs = sorted(pairs_dict.values(), key=lambda x: x['antipodality_score'], reverse=True)
    return sorted_pairs[:top_k]


def select_topk_dense(
    dense_indices: np.ndarray,
    densities: np.ndarray,
    W_enc: np.ndarray,
    W_dec: np.ndarray,
    k: int,
    clustering_method: str = "average",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Select the densest features, then order them by level and encoder clustering."""
    dense_indices = np.asarray(dense_indices)
    densities = np.asarray(densities)
    if len(dense_indices) < 10:
        raise ValueError(f"Need at least 10 dense features for clustering, got {len(dense_indices)}")

    dense_densities = densities[dense_indices]
    top_k_positions = np.argsort(dense_densities)[-k:][::-1]
    selected_indices = dense_indices[top_k_positions]
    W_enc_selected = W_enc[selected_indices]
    W_dec_selected = W_dec[selected_indices]
    final_order, level_boundaries = group_then_cluster_order(selected_indices, W_enc_selected, clustering_method)
    ordered_indices = selected_indices[final_order]
    W_enc_ordered = W_enc_selected[final_order]
    W_dec_ordered = W_dec_selected[final_order]
    return ordered_indices, W_enc_ordered, W_dec_ordered, level_boundaries


def antipodal_pairs_from_mats(
    C_enc: np.ndarray,
    C_dec: np.ndarray,
    threshold: float,
    ordered_indices: np.ndarray,
) -> List[Dict[str, float | int]]:
    """Return negative-negative pairs whose cosine product meets the threshold."""
    C_enc, C_dec = np.asarray(C_enc), np.asarray(C_dec)
    ordered_indices = np.asarray(ordered_indices)
    pairs = []
    n = C_enc.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            enc_sim = C_enc[i, j]
            dec_sim = C_dec[i, j]
            if enc_sim < 0 and dec_sim < 0:
                antipodal_score = (-enc_sim) * (-dec_sim)
                if antipodal_score >= threshold:
                    pairs.append({
                        'matrix_i': int(i),
                        'matrix_j': int(j),
                        'feature1_idx': int(ordered_indices[i]),
                        'feature2_idx': int(ordered_indices[j]),
                        'enc_sim': float(enc_sim),
                        'dec_sim': float(dec_sim),
                        'antipodal_score': float(antipodal_score)
                    })

    pairs.sort(key=lambda x: x['antipodal_score'], reverse=True)
    return pairs
