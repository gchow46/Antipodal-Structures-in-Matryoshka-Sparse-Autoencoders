"""
Pure analysis functions for antipodality computation.

This module contains stateless, testable functions for the core analysis logic.
No matplotlib, no filesystem I/O, no sae_lens - just pure computational functions
that transform inputs to outputs.
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Iterable
from scipy.stats import spearmanr

# Internal helpers
from .similarity import (
    normalize_weights,
    blocked_pair_scores,
    cosine_matrix,
)
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
    """
    Wraps normalize_weights + blocked_pair_scores.

    """
    with torch.no_grad():
        # Determine which features to analyze
        if feature_indices is None:
            indices = np.arange(W_enc.shape[0], dtype=np.int64)
        else:
            indices = np.asarray(feature_indices, dtype=np.int64)

        # Step 1: Normalize weights
        E, D = normalize_weights(W_enc, W_dec, indices)

        # Step 2: Compute blocked pair scores
        scores, partners, computation_metadata = blocked_pair_scores(E, D, top_k, block_size, antipodal_only)

        # Step 3: Validate scores and compute statistics
        summary_stats = validate_scores(scores)

        return {
            "feature_indices": indices,
            "antipodality_scores": scores,
            "antipodal_partners": partners,
            "summary_stats": summary_stats,
        }


def validate_scores(scores: np.ndarray) -> Dict[str, float]:
    """
    Compute basic statistical summaries of antipodality scores.
    """
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
    """
    Returns (indices_above_threshold, level_stats_by_level_value).

    """
    densities = np.asarray(densities)
    dense_mask = densities > threshold
    dense_indices = np.where(dense_mask)[0]

    # Analyze by Matryoshka level
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
    """
    Compute mean antipodality scores for dense vs sparse features.

    Args:
        densities: Array of density values for all features
        scores: Array of antipodality scores for all features
        threshold: Density threshold to separate dense from sparse features

    Returns:
        Dict with dense/sparse counts and mean antipodality scores
    """
    # Filter out invalid scores for comparison
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
    """
    Returns dict with 'spearman_r', 'spearman_p', and dense/sparse means & counts.
    """
    # Get dense/sparse breakdown
    means_dict = dense_sparse_means(densities, scores, threshold)

    # Compute Spearman correlation
    rho, p = spearmanr(densities, scores)

    # Merge results
    result = {
        'spearman_r': rho,
        'spearman_p': p,
    }
    result.update(means_dict)

    return result






def analyze_matryoshka_hierarchy(
    indices: np.ndarray,
    scores: np.ndarray,
    partners: np.ndarray,
) -> Dict[str, object]:
    """
    Analyze features by Matryoshka level
    """
    indices, scores = np.asarray(indices), np.asarray(scores)

    # Get feature levels for reference
    feature_levels = [MatryoshkaUtils.get_level(idx) for idx in indices]

    # Compute level analysis (inline implementation of scores_by_level)
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

    return {
        'level_analysis': level_analysis,
        'feature_levels': feature_levels
    }


def find_top_pairs(
    indices: np.ndarray,
    scores: np.ndarray,
    partners: np.ndarray,
    top_k: int = 10,
    W_enc: Optional[np.ndarray] = None,
    W_dec: Optional[np.ndarray] = None,
) -> List[Dict[str, object]]:
    """
    Deduplicate by canonical (min(i,j), max(i,j)) over original SAE indices.
    Keep highest score when duplicates occur.
    """
    indices, scores, partners = np.asarray(indices), np.asarray(scores), np.asarray(partners)

    # Filter to valid pairs
    valid_mask = np.isfinite(scores) & (partners >= 0) & (partners < len(indices))
    valid_indices = indices[valid_mask]
    valid_scores = scores[valid_mask]
    valid_partners = partners[valid_mask]

    if len(valid_indices) == 0:
        return []

    # Create pairs and deduplicate
    pairs_dict = {}

    for i, (feat_idx, score, partner_pos) in enumerate(zip(valid_indices, valid_scores, valid_partners)):
        partner_idx = indices[partner_pos]

        # Create canonical pair key
        pair_key = (min(feat_idx, partner_idx), max(feat_idx, partner_idx))

        # Keep highest score for this pair
        if pair_key not in pairs_dict or score > pairs_dict[pair_key]['antipodality_score']:
            pair_info = {
                'feature1_idx': int(feat_idx),
                'feature2_idx': int(partner_idx),
                'feature1_level': MatryoshkaUtils.get_level(feat_idx),
                'feature2_level': MatryoshkaUtils.get_level(partner_idx),
                'antipodality_score': float(score)
            }

            # Compute similarities if weights provided
            if W_enc is not None and W_dec is not None:
                with torch.no_grad():
                    # Row-wise cosine similarity
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

    # Sort by antipodality score (descending) and return top_k
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
    """
    Pick top-k densest from dense_indices, then group_then_cluster_order using W_enc subset.
    Return (ordered_indices, W_enc_ordered, W_dec_ordered, level_boundaries).
    """
    dense_indices = np.asarray(dense_indices)
    densities = np.asarray(densities)

    if len(dense_indices) < 10:
        raise ValueError(f"Need at least 10 dense features for clustering, got {len(dense_indices)}")

    # Select top-k densest
    dense_densities = densities[dense_indices]
    top_k_positions = np.argsort(dense_densities)[-k:][::-1]  # Descending order
    selected_indices = dense_indices[top_k_positions]

    # Extract weights for selected features
    W_enc_selected = W_enc[selected_indices]
    W_dec_selected = W_dec[selected_indices]

    # Group and cluster by Matryoshka level
    final_order, level_boundaries = group_then_cluster_order(selected_indices, W_enc_selected, clustering_method)

    # Reorder weights according to clustering
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
    """
    Scan upper triangle for enc<0 and dec<0; score = (-enc)*(-dec).
    Return sorted list of dicts with matrix positions i,j, original feature indices,
    enc_sim, dec_sim, antipodal_score.
    """
    C_enc, C_dec = np.asarray(C_enc), np.asarray(C_dec)
    ordered_indices = np.asarray(ordered_indices)

    pairs = []
    n = C_enc.shape[0]

    # Scan upper triangle
    for i in range(n):
        for j in range(i + 1, n):
            enc_sim = C_enc[i, j]
            dec_sim = C_dec[i, j]

            # Check if both similarities are negative (antipodal)
            if enc_sim < 0 and dec_sim < 0:
                antipodal_score = (-enc_sim) * (-dec_sim)

                # Only include if score meets threshold
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

    # Sort by antipodal score (descending)
    pairs.sort(key=lambda x: x['antipodal_score'], reverse=True)

    return pairs