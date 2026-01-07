# antipodality/viz/payloads.py

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np

from antipodality.types import EncDecScatterPayload, WithinCrossPayload
from antipodality.constants import MATRYOSHKA_LEVELS, LEVEL_RANGES, LEVEL_COLORS
from antipodality.utils import MatryoshkaUtils, assign_levels
from antipodality import analysis
from antipodality.similarity import cosine_matrix
from antipodality.clustering import group_then_cluster_order
from antipodality.viz import umap as umap_viz
from antipodality.utils import assign_levels
from antipodality.constants import LEVEL_COLORS, MATRYOSHKA_LEVELS
import random



def _get_densities(results: dict) -> np.ndarray:
    return np.asarray(results.get("analysis_metadata", {}).get("densities", []), dtype=float)


def antipodality_extraction(results: dict) -> dict:
    thr = float(results["analysis_metadata"]["density_threshold"])
    densities = _get_densities(results)

    all_sec = results["antipodality_analysis"]["all_features"]
    dense_sec = results["antipodality_analysis"]["dense_features"]

    all_idx = np.asarray(all_sec["feature_indices"], dtype=int)
    all_scores = np.asarray(all_sec["antipodality_scores"], dtype=float)
    all_dens = densities[all_idx] if densities.size else np.zeros_like(all_scores)

    dense_idx = np.asarray(dense_sec["feature_indices"], dtype=int)
    dense_scores = np.asarray(dense_sec["antipodality_scores"], dtype=float)

    corr = results.get("correlation_analysis", {})

    # Subplot 1
    mask = np.isfinite(all_scores)
    valid_idx = all_idx[mask]
    valid_dens = all_dens[mask]
    valid_scores = all_scores[mask]
    level_ids = assign_levels(valid_idx)

    # Subplot 2
    valid_scores_all = all_scores[np.isfinite(all_scores)]

    # Subplot 3
    dense_mask = all_dens > thr
    dense_scores_all = all_scores[dense_mask & np.isfinite(all_scores)]
    sparse_scores_all = all_scores[(~dense_mask) & np.isfinite(all_scores)]

    # Subplot 4: count dense by level
    level_counts: Dict[int, int] = {}
    for i, level in enumerate(MATRYOSHKA_LEVELS):
        lo, hi = LEVEL_RANGES[i]
        m = (dense_idx >= lo) & (dense_idx < hi)
        level_counts[level] = int(np.sum(m))

    layer = int(results["analysis_metadata"]["layer"])
    return dict(
        layer=layer,
        threshold=thr,
        correlation_results=corr,
        valid_indices=valid_idx,
        valid_densities=valid_dens,
        valid_scores=valid_scores,
        level_ids=level_ids,
        valid_scores_all=valid_scores_all,
        dense_scores_all=dense_scores_all,
        sparse_scores_all=sparse_scores_all,
        level_counts=level_counts,
        level_colors=LEVEL_COLORS[:len(MATRYOSHKA_LEVELS)],
    )


def ext_dense_features(results: dict) -> dict:
    dense_sec = results["antipodality_analysis"]["dense_features"]
    dense_scores = np.asarray(dense_sec["antipodality_scores"], dtype=float)
    dense_idx = np.asarray(dense_sec["feature_indices"], dtype=int)

    level_score_groups: List[np.ndarray] = []
    level_labels: List[int] = []
    level_colors: List[str] = []

    for i, level in enumerate(MATRYOSHKA_LEVELS):
        lo, hi = LEVEL_RANGES[i]
        m = (dense_idx >= lo) & (dense_idx < hi)
        vals = dense_scores[m]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            level_score_groups.append(vals)
            level_labels.append(level)
            level_colors.append(LEVEL_COLORS[i])

    corr = results.get("correlation_analysis")
    corr_summary = None
    if corr is not None and "spearman_r" in corr and "spearman_p" in corr:
        corr_summary = {"spearman_r": float(corr["spearman_r"]), "spearman_p": float(corr["spearman_p"])}

    layer = int(results["analysis_metadata"]["layer"])

    # Print detailed boxplot statistics
    if level_score_groups:
        print("Level-wise Antipodality Distribution:")
        print("Format: Level: n=count, median=X.XXX, IQR=[Q1-Q3], whiskers=[low-high], outliers=count, range=[min-max]")
        for i, (level, scores) in enumerate(zip(level_labels, level_score_groups)):
            if len(scores) > 0:
                scores_array = np.array(scores)

                # Compute comprehensive boxplot statistics
                q1, median, q3 = np.percentile(scores_array, [25, 50, 75])
                iqr = q3 - q1
                min_val = np.min(scores_array)
                max_val = np.max(scores_array)

                # Compute outlier thresholds (standard boxplot definition)
                lower_fence = q1 - 1.5 * iqr
                upper_fence = q3 + 1.5 * iqr

                # Identify outliers
                outlier_mask = (scores_array < lower_fence) | (scores_array > upper_fence)
                outlier_count = np.sum(outlier_mask)

                # Compute whisker bounds (furthest non-outlier values)
                non_outlier_scores = scores_array[~outlier_mask]
                if len(non_outlier_scores) > 0:
                    whisker_low = np.min(non_outlier_scores)
                    whisker_high = np.max(non_outlier_scores)
                else:
                    # Fallback if all values are outliers
                    whisker_low = min_val
                    whisker_high = max_val

                print(f"  L{level}: n={len(scores)}, median={median:.3f}, "
                      f"IQR=[{q1:.3f}-{q3:.3f}], whiskers=[{whisker_low:.3f}-{whisker_high:.3f}], "
                      f"outliers={outlier_count}, range=[{min_val:.3f}-{max_val:.3f}]")

    return dict(
        layer=layer,
        level_score_groups=level_score_groups,
        level_labels=level_labels,
        level_colors=level_colors,
        correlation_summary=corr_summary,
    )


def build_enc_dec_scatter_payload(
    results: dict,
    n_top_pairs: int = 5000,
    n_random_pairs: int = 5000,
    positive_threshold: float = 0.01,
    rng: Optional[object] = None,
) -> EncDecScatterPayload:
    meta_zero = dict(n_points=0, n_antipodal=0, n_synonym=0, n_mixed=0, n_random=0,
                     enc_range=(0.0, 0.0), dec_range=(0.0, 0.0), score_range=(0.0, 0.0))
    counts_zero = dict(top_right=0, top_left=0, bottom_left=0, bottom_right=0)

    W_enc = results.get("analysis_metadata", {}).get("W_enc")
    W_dec = results.get("analysis_metadata", {}).get("W_dec")
    if W_enc is None or W_dec is None:
        return EncDecScatterPayload(np.array([]), np.array([]), np.array([]), counts_zero, meta_zero)

    W_enc = np.asarray(W_enc, dtype=float)
    W_dec = np.asarray(W_dec, dtype=float)

    sec = results["antipodality_analysis"]["all_features"]
    scores = np.asarray(sec["antipodality_scores"], dtype=float)
    partners = np.asarray(sec["antipodal_partners"], dtype=int)
    feat_idx = np.asarray(sec["feature_indices"], dtype=int)

    rng = random.Random(42)

    # Gather antipodal pairs above threshold
    positive_mask = scores > positive_threshold
    pos_rows = np.where(positive_mask)[0]
    pairs = []
    seen = set()
    for r in pos_rows:
        p = int(partners[r])
        if p < 0 or p >= len(feat_idx):
            continue
        i = int(feat_idx[r]); j = int(feat_idx[p])
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((i, j))
    pairs = pairs[: max(1, n_top_pairs // 4)]

    # Random sampling for synonyms/mixed/random
    n_features = W_enc.shape[0]
    def cos(a, b):
        na = np.linalg.norm(a) + 1e-8
        nb = np.linalg.norm(b) + 1e-8
        return float(np.dot(a, b) / (na * nb))

    syn, mix, rnd = [], [], []
    target_syn = n_top_pairs // 4
    target_mix = n_top_pairs // 4
    target_rnd = n_random_pairs
    attempts = 0
    max_attempts = max(10000, 10 * (target_syn + target_mix + target_rnd))

    while (len(syn) < target_syn or len(mix) < target_mix or len(rnd) < target_rnd) and attempts < max_attempts:
        i = rng.randrange(n_features); j = rng.randrange(n_features)
        if i == j:
            attempts += 1; continue
        key = (min(i, j), max(i, j))
        if key in seen:
            attempts += 1; continue
        seen.add(key)
        ei, ej = W_enc[i], W_enc[j]
        di, dj = W_dec[i], W_dec[j]
        e = cos(ei, ej); d = cos(di, dj)

        entry = dict(feature1_idx=i, feature2_idx=j, encoder_similarity=e, decoder_similarity=d)
        if e > 0 and d > 0 and len(syn) < target_syn:
            syn.append(entry)
        elif ((e > 0 and d < 0) or (e < 0 and d > 0)) and len(mix) < target_mix:
            mix.append(entry)
        elif len(rnd) < target_rnd:
            rnd.append(entry)
        attempts += 1

    # Combine
    combined = []
    # antipodal block (compute sims now)
    for i, j in pairs:
        e = cos(W_enc[i], W_enc[j]); d = cos(W_dec[i], W_dec[j])
        combined.append(dict(encoder_similarity=e, decoder_similarity=d))
    combined.extend(syn); combined.extend(mix); combined.extend(rnd)

    if not combined:
        return EncDecScatterPayload(np.array([]), np.array([]), np.array([]), counts_zero, meta_zero)

    enc = np.array([c["encoder_similarity"] for c in combined], dtype=float)
    dec = np.array([c["decoder_similarity"] for c in combined], dtype=float)
    antip = np.sqrt(np.clip(-enc, 0, 1) * np.clip(-dec, 0, 1))
    syns  = np.sqrt(np.clip( enc, 0, 1) * np.clip( dec, 0, 1))
    score = np.maximum(antip, syns)

    counts = dict(
        top_right=int(np.sum((enc > 0) & (dec > 0))),
        top_left=int(np.sum((enc < 0) & (dec > 0))),
        bottom_left=int(np.sum((enc < 0) & (dec < 0))),
        bottom_right=int(np.sum((enc > 0) & (dec < 0))),
    )
    layer = int(results["analysis_metadata"]["layer"])
    meta = dict(
        n_points=int(enc.size),
        n_antipodal=len(pairs),
        n_synonym=len(syn),
        n_mixed=len(mix),
        n_random=len(rnd),
        enc_range=(float(enc.min()), float(enc.max())) if enc.size else (0.0, 0.0),
        dec_range=(float(dec.min()), float(dec.max())) if dec.size else (0.0, 0.0),
        score_range=(float(score.min()), float(score.max())) if score.size else (0.0, 0.0),
        layer=layer,
    )
    return EncDecScatterPayload(enc, dec, score, counts, layer, meta)


def build_within_cross_payload(results: dict, top_k: int = 10) -> WithinCrossPayload:
    dense_sec = results["antipodality_analysis"]["dense_features"]
    indices = np.asarray(dense_sec["feature_indices"], dtype=int)
    scores  = np.asarray(dense_sec["antipodality_scores"], dtype=float)
    partners = np.asarray(dense_sec["antipodal_partners"], dtype=int)

    uniq_within = {}
    uniq_cross = {}

    for row, feat in enumerate(indices):
        p = int(partners[row])
        if p < 0 or p >= len(indices):
            continue
        other = int(indices[p])

        a, b = (min(feat, other), max(feat, other))
        L1 = MatryoshkaUtils.get_level(a)
        L2 = MatryoshkaUtils.get_level(b)

        item = dict(
            feature1_idx=a,
            feature2_idx=b,
            antipodality_score=float(scores[row]),
            feature1_level=L1,
            feature2_level=L2,
            pair_label=f"{a}-{b}",
        )
        target = uniq_within if L1 == L2 else uniq_cross
        if (a, b) not in target or item["antipodality_score"] > target[(a, b)]["antipodality_score"]:
            target[(a, b)] = item

    within = sorted(uniq_within.values(), key=lambda x: x["antipodality_score"], reverse=True)
    cross  = sorted(uniq_cross.values(), key=lambda x: x["antipodality_score"], reverse=True)

    within_confirmed = [p for p in within if p["antipodality_score"] >= 0.8]
    cross_confirmed  = [p for p in cross  if p["antipodality_score"] >= 0.8]

    layer = int(results["analysis_metadata"]["layer"])
    stats = dict(
        n_within_all=len(within),
        n_cross_all=len(cross),
        n_within_confirmed=len(within_confirmed),
        n_within_candidates=len(within) - len(within_confirmed),
        n_cross_confirmed=len(cross_confirmed),
        n_cross_candidates=len(cross) - len(cross_confirmed),
        within_mean=float(np.mean([p["antipodality_score"] for p in within])) if within else None,
        cross_mean=float(np.mean([p["antipodality_score"] for p in cross])) if cross else None,
        within_max=float(max([p["antipodality_score"] for p in within])) if within else None,
        cross_max=float(max([p["antipodality_score"] for p in cross])) if cross else None,
        layer=layer,
    )

    return WithinCrossPayload(within[:top_k], cross[:top_k], layer, stats)


def build_unbiased_antipodal_payload(results: dict) -> dict:
    """
    Build payload for unbiased antipodal analysis showing ALL high-scoring pairs.

    Args:
        results: Analysis results dictionary

    Returns:
        dict: Payload with matrices and metadata for unbiased antipodal visualization
    """
    layer = int(results["analysis_metadata"]["layer"])
    threshold = float(results["analysis_metadata"]["density_threshold"])
    densities = _get_densities(results)
    W_enc = results["analysis_metadata"]["W_enc"]
    W_dec = results["analysis_metadata"]["W_dec"]

    # Get top antipodal pairs
    pairs = results.get('top_antipodal_pairs', [])

    # Filter for high-scoring pairs (>=0.8)
    high_scoring_pairs = [p for p in pairs if p.get('antipodality_score', 0) >= 0.8]

    if len(high_scoring_pairs) == 0:
        return {
            'has_data': False,
            'layer': layer
        }

    # Extract unique feature indices from pairs
    indices_set = set()
    for pair in high_scoring_pairs:
        indices_set.add(pair['feature1_idx'])
        indices_set.add(pair['feature2_idx'])
    indices = np.array(sorted(indices_set))

    # Order features by Matryoshka level then cluster
    W_enc_subset = W_enc[indices]
    W_dec_subset = W_dec[indices]
    order, boundaries = group_then_cluster_order(indices, W_enc_subset)

    # Compute cosine similarity matrices
    W_enc_ordered = W_enc_subset[order]
    W_dec_ordered = W_dec_subset[order]
    C_enc = cosine_matrix(W_enc_ordered)
    C_dec = cosine_matrix(W_dec_ordered)

    # Get ordered indices for plotting
    ordered_indices = indices[order]

    return {
        'layer': layer,
        'has_data': True,
        'threshold': threshold,
        'C_enc': C_enc,
        'C_dec': C_dec,
        'ordered_indices': ordered_indices,
        'high_scoring_pairs': high_scoring_pairs,
        'densities': densities,
        'n_pairs': len(high_scoring_pairs),
        'n_features': len(indices)
    }


def build_umap_payload(
    results: dict,
    W_enc: np.ndarray,
    W_dec: np.ndarray,
    dense_indices: np.ndarray,
    top_pairs: list,
    umap_neighbors: int = 15
) -> dict:
    """
    Build payload for UMAP geometric analysis visualization.

    Args:
        results: Analysis results dictionary
        W_enc: Full encoder weight matrix
        W_dec: Full decoder weight matrix
        dense_indices: Indices of dense features
        top_pairs: List of top antipodal pairs
        umap_neighbors: Number of neighbors for UMAP

    Returns:
        Dict with UMAP embeddings, colors, segments and metadata
    """

    # Get layer information
    layer = int(results["analysis_metadata"]["layer"])

    # Extract dense features section
    dense_sec = results["antipodality_analysis"]["dense_features"]

    # Prepare normalized weights for dense features
    dense_W_enc = W_enc[dense_indices]
    dense_W_dec = W_dec[dense_indices]

    # Normalize weights
    dense_W_enc_norm = dense_W_enc / (np.linalg.norm(dense_W_enc, axis=1, keepdims=True) + 1e-8)
    dense_W_dec_norm = dense_W_dec / (np.linalg.norm(dense_W_dec, axis=1, keepdims=True) + 1e-8)

    # Compute UMAP embeddings
    enc_embedding, dec_embedding = umap_viz.umap_embeddings(
        dense_W_enc_norm, dense_W_dec_norm, umap_neighbors
    )

    # Build level colors and legend
    level_assignments = assign_levels(dense_indices)
    colors_by_level = []
    for level_idx in level_assignments:
        if 0 <= level_idx < len(LEVEL_COLORS):
            colors_by_level.append(LEVEL_COLORS[level_idx])
        else:
            colors_by_level.append('gray')

    level_legend = [(f'L{level}', LEVEL_COLORS[i]) for i, level in enumerate(MATRYOSHKA_LEVELS)]

    # Prepare pairs with level info for routing - filter for high antipodality scores
    high_quality_pairs = [p for p in top_pairs if p.get('antipodality_score', 0) >= 0.8]
    pairs_for_routing = []
    for pair in high_quality_pairs[:20]:  # Limit to top 20 high-quality pairs for visualization
        pair_with_levels = pair.copy()
        # Add level info if not present
        feat1_level = None
        feat2_level = None
        for i, feat_idx in enumerate(dense_indices):
            if feat_idx == pair['feature1_idx']:
                feat1_level = level_assignments[i]
            if feat_idx == pair['feature2_idx']:
                feat2_level = level_assignments[i]

        pair_with_levels['feature1_level'] = feat1_level
        pair_with_levels['feature2_level'] = feat2_level
        pair_with_levels['is_within_level'] = feat1_level == feat2_level
        pairs_for_routing.append(pair_with_levels)

    # Generate routing segments
    enc_segments = umap_viz.route_pairs_on_embedding(
        enc_embedding, dense_indices, pairs_for_routing, max_pairs=20
    )
    dec_segments = umap_viz.route_pairs_on_embedding(
        dec_embedding, dense_indices, pairs_for_routing, max_pairs=20
    )

    # Get densities for dense features
    densities = _get_densities(results)

    return {
        'layer': layer,
        'enc_embedding': enc_embedding,
        'dec_embedding': dec_embedding,
        'colors_by_level': colors_by_level,
        'densities': densities[dense_indices],
        'antipodality_scores_mask': np.isfinite(dense_sec['antipodality_scores']),
        'antipodality_scores': dense_sec['antipodality_scores'][np.isfinite(dense_sec['antipodality_scores'])],
        'enc_segments': enc_segments,
        'dec_segments': dec_segments,
        'level_legend': level_legend,
        'title_suffix': f' - Layer {layer}'
    }


def build_dense_focused_matrix_payload(results: dict, top_k: int = 50) -> dict:
    """
    Build payload for dense-focused matrix visualization.

    Args:
        results: Analysis results dictionary
        top_k: Number of top dense features to analyze

    Returns:
        dict: Payload with matrices and metadata for dense-focused matrix visualization
    """
    layer = int(results["analysis_metadata"]["layer"])
    threshold = float(results["analysis_metadata"]["density_threshold"])
    densities = _get_densities(results)
    W_enc = results["analysis_metadata"]["W_enc"]
    W_dec = results["analysis_metadata"]["W_dec"]

    try:
        # Select top-k dense features
        dense_indices, _ = analysis.dense_feature_indices(densities, threshold)
        ordered_indices, W_enc_ordered, W_dec_ordered, level_boundaries = analysis.select_topk_dense(
            dense_indices, densities, W_enc, W_dec, top_k, 'average'
        )
        actual_k = len(ordered_indices)
    except ValueError as e:
        return {
            'has_data': False,
            'layer': layer,
            'message': str(e)
        }

    # Compute cosine similarity matrices
    C_enc = cosine_matrix(W_enc_ordered)
    C_dec = cosine_matrix(W_dec_ordered)

    # Find antipodal relationships within this dense subset
    antipodal_threshold = 0.8
    antipodal_pairs = analysis.antipodal_pairs_from_mats(C_enc, C_dec, antipodal_threshold, ordered_indices)

    return {
        'layer': layer,
        'has_data': True,
        'top_k': top_k,
        'actual_k': actual_k,
        'C_enc': C_enc,
        'C_dec': C_dec,
        'ordered_indices': ordered_indices,
        'antipodal_pairs': antipodal_pairs,
        'antipodal_threshold': antipodal_threshold,
        'n_antipodal_pairs': len(antipodal_pairs)
    }