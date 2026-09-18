"""Prepare analysis results for plotting."""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

from antipodality.types import EncDecScatterPayload, WithinCrossPayload
from antipodality.constants import MATRYOSHKA_LEVELS, LEVEL_RANGES, LEVEL_COLORS
from antipodality.utils import MatryoshkaUtils, assign_levels
from antipodality import analysis
from antipodality.similarity import cosine_matrix
from antipodality.clustering import group_then_cluster_order
from antipodality.viz import umap as umap_viz
import random


def _get_densities(results: dict) -> np.ndarray:
    return np.asarray(results.get("analysis_metadata", {}).get("densities", []), dtype=float)


def antipodality_extraction(results: dict) -> dict:
    """Prepare global score distributions, correlations, and level counts."""
    threshold = float(results["analysis_metadata"]["density_threshold"])
    densities = _get_densities(results)
    all_features = results["antipodality_analysis"]["all_features"]
    dense_features = results["antipodality_analysis"]["dense_features"]

    all_indices = np.asarray(all_features["feature_indices"], dtype=int)
    all_scores = np.asarray(all_features["antipodality_scores"], dtype=float)
    all_densities = densities[all_indices] if densities.size else np.zeros_like(all_scores)
    dense_indices = np.asarray(dense_features["feature_indices"], dtype=int)
    correlation = results.get("correlation_analysis", {})

    finite_mask = np.isfinite(all_scores)
    valid_indices = all_indices[finite_mask]
    valid_densities = all_densities[finite_mask]
    valid_scores = all_scores[finite_mask]
    level_ids = assign_levels(valid_indices)
    valid_scores_all = all_scores[np.isfinite(all_scores)]

    dense_mask = all_densities > threshold
    dense_scores_all = all_scores[dense_mask & np.isfinite(all_scores)]
    sparse_scores_all = all_scores[(~dense_mask) & np.isfinite(all_scores)]

    level_counts: Dict[int, int] = {}
    for i, level in enumerate(MATRYOSHKA_LEVELS):
        lo, hi = LEVEL_RANGES[i]
        level_mask = (dense_indices >= lo) & (dense_indices < hi)
        level_counts[level] = int(np.sum(level_mask))

    layer = int(results["analysis_metadata"]["layer"])
    return dict(
        layer=layer,
        threshold=threshold,
        correlation_results=correlation,
        valid_indices=valid_indices,
        valid_densities=valid_densities,
        valid_scores=valid_scores,
        level_ids=level_ids,
        valid_scores_all=valid_scores_all,
        dense_scores_all=dense_scores_all,
        sparse_scores_all=sparse_scores_all,
        level_counts=level_counts,
        level_colors=LEVEL_COLORS[:len(MATRYOSHKA_LEVELS)],
    )


def _print_level_statistics(level_labels, level_score_groups):
    if not level_score_groups:
        return
    print("Level-wise Antipodality Distribution:")
    print("Format: Level: n=count, median=X.XXX, IQR=[Q1-Q3], whiskers=[low-high], outliers=count, range=[min-max]")
    for level, scores in zip(level_labels, level_score_groups):
        if len(scores) == 0:
            continue
        scores_array = np.array(scores)
        q1, median, q3 = np.percentile(scores_array, [25, 50, 75])
        iqr = q3 - q1
        min_val = np.min(scores_array)
        max_val = np.max(scores_array)
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outlier_mask = (scores_array < lower_fence) | (scores_array > upper_fence)
        outlier_count = np.sum(outlier_mask)
        non_outlier_scores = scores_array[~outlier_mask]
        if len(non_outlier_scores) > 0:
            whisker_low = np.min(non_outlier_scores)
            whisker_high = np.max(non_outlier_scores)
        else:
            whisker_low = min_val
            whisker_high = max_val
        print(f"  L{level}: n={len(scores)}, median={median:.3f}, "
              f"IQR=[{q1:.3f}-{q3:.3f}], whiskers=[{whisker_low:.3f}-{whisker_high:.3f}], "
              f"outliers={outlier_count}, range=[{min_val:.3f}-{max_val:.3f}]")


def ext_dense_features(results: dict) -> dict:
    """Group finite dense-only scores by level and print their boxplot statistics."""
    dense_features = results["antipodality_analysis"]["dense_features"]
    dense_scores = np.asarray(dense_features["antipodality_scores"], dtype=float)
    dense_indices = np.asarray(dense_features["feature_indices"], dtype=int)
    level_score_groups: List[np.ndarray] = []
    level_labels: List[int] = []
    level_colors: List[str] = []

    for i, level in enumerate(MATRYOSHKA_LEVELS):
        lo, hi = LEVEL_RANGES[i]
        level_mask = (dense_indices >= lo) & (dense_indices < hi)
        values = dense_scores[level_mask]
        values = values[np.isfinite(values)]
        if values.size:
            level_score_groups.append(values)
            level_labels.append(level)
            level_colors.append(LEVEL_COLORS[i])

    correlation = results.get("correlation_analysis")
    correlation_summary = None
    if correlation is not None and "spearman_r" in correlation and "spearman_p" in correlation:
        correlation_summary = {
            "spearman_r": float(correlation["spearman_r"]),
            "spearman_p": float(correlation["spearman_p"]),
        }
    layer = int(results["analysis_metadata"]["layer"])
    _print_level_statistics(level_labels, level_score_groups)
    return dict(
        layer=layer,
        level_score_groups=level_score_groups,
        level_labels=level_labels,
        level_colors=level_colors,
        correlation_summary=correlation_summary,
    )


def _cosine(a, b):
    norm_a = np.linalg.norm(a) + 1e-8
    norm_b = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (norm_a * norm_b))


def build_enc_dec_scatter_payload(
    results: dict,
    n_top_pairs: int = 5000,
    n_random_pairs: int = 5000,
    positive_threshold: float = 0.01,
    rng: Optional[object] = None,
) -> EncDecScatterPayload:
    """Combine selected partner pairs with fixed-seed samples of other relationships."""
    meta_zero = dict(n_points=0, n_antipodal=0, n_synonym=0, n_mixed=0, n_random=0,
                     enc_range=(0.0, 0.0), dec_range=(0.0, 0.0), score_range=(0.0, 0.0))
    counts_zero = dict(top_right=0, top_left=0, bottom_left=0, bottom_right=0)
    W_enc = results.get("analysis_metadata", {}).get("W_enc")
    W_dec = results.get("analysis_metadata", {}).get("W_dec")
    if W_enc is None or W_dec is None:
        return EncDecScatterPayload(np.array([]), np.array([]), np.array([]), counts_zero, meta_zero)

    W_enc = np.asarray(W_enc, dtype=float)
    W_dec = np.asarray(W_dec, dtype=float)
    all_features = results["antipodality_analysis"]["all_features"]
    scores = np.asarray(all_features["antipodality_scores"], dtype=float)
    partners = np.asarray(all_features["antipodal_partners"], dtype=int)
    feature_indices = np.asarray(all_features["feature_indices"], dtype=int)
    rng = random.Random(42)

    positive_mask = scores > positive_threshold
    positive_rows = np.where(positive_mask)[0]
    pairs = []
    seen = set()
    for row in positive_rows:
        partner = int(partners[row])
        if partner < 0 or partner >= len(feature_indices):
            continue
        i = int(feature_indices[row])
        j = int(feature_indices[partner])
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((i, j))
    pairs = pairs[:max(1, n_top_pairs // 4)]

    n_features = W_enc.shape[0]
    synonym_pairs, mixed_pairs, random_pairs = [], [], []
    target_synonyms = n_top_pairs // 4
    target_mixed = n_top_pairs // 4
    target_random = n_random_pairs
    attempts = 0
    max_attempts = max(10000, 10 * (target_synonyms + target_mixed + target_random))

    while (len(synonym_pairs) < target_synonyms or len(mixed_pairs) < target_mixed or len(random_pairs) < target_random) and attempts < max_attempts:
        i = rng.randrange(n_features)
        j = rng.randrange(n_features)
        if i == j:
            attempts += 1
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            attempts += 1
            continue
        seen.add(key)
        ei, ej = W_enc[i], W_enc[j]
        di, dj = W_dec[i], W_dec[j]
        enc_sim = _cosine(ei, ej)
        dec_sim = _cosine(di, dj)
        entry = dict(feature1_idx=i, feature2_idx=j, encoder_similarity=enc_sim, decoder_similarity=dec_sim)
        if enc_sim > 0 and dec_sim > 0 and len(synonym_pairs) < target_synonyms:
            synonym_pairs.append(entry)
        elif ((enc_sim > 0 and dec_sim < 0) or (enc_sim < 0 and dec_sim > 0)) and len(mixed_pairs) < target_mixed:
            mixed_pairs.append(entry)
        elif len(random_pairs) < target_random:
            random_pairs.append(entry)
        attempts += 1

    combined = []
    for i, j in pairs:
        enc_sim = _cosine(W_enc[i], W_enc[j])
        dec_sim = _cosine(W_dec[i], W_dec[j])
        combined.append(dict(encoder_similarity=enc_sim, decoder_similarity=dec_sim))
    combined.extend(synonym_pairs)
    combined.extend(mixed_pairs)
    combined.extend(random_pairs)
    if not combined:
        return EncDecScatterPayload(np.array([]), np.array([]), np.array([]), counts_zero, meta_zero)

    enc = np.array([pair["encoder_similarity"] for pair in combined], dtype=float)
    dec = np.array([pair["decoder_similarity"] for pair in combined], dtype=float)
    antipodal_strength = np.sqrt(np.clip(-enc, 0, 1) * np.clip(-dec, 0, 1))
    synonym_strength = np.sqrt(np.clip(enc, 0, 1) * np.clip(dec, 0, 1))
    score = np.maximum(antipodal_strength, synonym_strength)
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
        n_synonym=len(synonym_pairs),
        n_mixed=len(mixed_pairs),
        n_random=len(random_pairs),
        enc_range=(float(enc.min()), float(enc.max())) if enc.size else (0.0, 0.0),
        dec_range=(float(dec.min()), float(dec.max())) if dec.size else (0.0, 0.0),
        score_range=(float(score.min()), float(score.max())) if score.size else (0.0, 0.0),
        layer=layer,
    )
    return EncDecScatterPayload(enc, dec, score, counts, layer, meta)


def build_within_cross_payload(results: dict, top_k: int = 10) -> WithinCrossPayload:
    """Split dense-feature partner pairs into within-level and cross-level groups."""
    dense_features = results["antipodality_analysis"]["dense_features"]
    indices = np.asarray(dense_features["feature_indices"], dtype=int)
    scores = np.asarray(dense_features["antipodality_scores"], dtype=float)
    partners = np.asarray(dense_features["antipodal_partners"], dtype=int)
    within_pairs = {}
    cross_pairs = {}

    for row, feature in enumerate(indices):
        partner = int(partners[row])
        if partner < 0 or partner >= len(indices):
            continue
        other = int(indices[partner])
        a, b = (min(feature, other), max(feature, other))
        first_level = MatryoshkaUtils.get_level(a)
        second_level = MatryoshkaUtils.get_level(b)
        item = dict(
            feature1_idx=a,
            feature2_idx=b,
            antipodality_score=float(scores[row]),
            feature1_level=first_level,
            feature2_level=second_level,
            pair_label=f"{a}-{b}",
        )
        target = within_pairs if first_level == second_level else cross_pairs
        if (a, b) not in target or item["antipodality_score"] > target[(a, b)]["antipodality_score"]:
            target[(a, b)] = item

    within = sorted(within_pairs.values(), key=lambda pair: pair["antipodality_score"], reverse=True)
    cross = sorted(cross_pairs.values(), key=lambda pair: pair["antipodality_score"], reverse=True)
    within_confirmed = [p for p in within if p["antipodality_score"] >= 0.8]
    cross_confirmed = [p for p in cross if p["antipodality_score"] >= 0.8]
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
    """Build similarity matrices for the supplied high-scoring pairs."""
    layer = int(results["analysis_metadata"]["layer"])
    threshold = float(results["analysis_metadata"]["density_threshold"])
    densities = _get_densities(results)
    W_enc = results["analysis_metadata"]["W_enc"]
    W_dec = results["analysis_metadata"]["W_dec"]
    pairs = results.get('top_antipodal_pairs', [])
    high_scoring_pairs = [p for p in pairs if p.get('antipodality_score', 0) >= 0.8]
    if len(high_scoring_pairs) == 0:
        return {'has_data': False, 'layer': layer}

    indices_set = set()
    for pair in high_scoring_pairs:
        indices_set.add(pair['feature1_idx'])
        indices_set.add(pair['feature2_idx'])
    indices = np.array(sorted(indices_set))
    W_enc_subset = W_enc[indices]
    W_dec_subset = W_dec[indices]
    order, _ = group_then_cluster_order(indices, W_enc_subset)
    W_enc_ordered = W_enc_subset[order]
    W_dec_ordered = W_dec_subset[order]
    C_enc = cosine_matrix(W_enc_ordered)
    C_dec = cosine_matrix(W_dec_ordered)
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
    """Embed dense features and route up to 20 high-scoring pairs."""
    layer = int(results["analysis_metadata"]["layer"])
    dense_features = results["antipodality_analysis"]["dense_features"]
    dense_W_enc = W_enc[dense_indices]
    dense_W_dec = W_dec[dense_indices]
    dense_W_enc_norm = dense_W_enc / (np.linalg.norm(dense_W_enc, axis=1, keepdims=True) + 1e-8)
    dense_W_dec_norm = dense_W_dec / (np.linalg.norm(dense_W_dec, axis=1, keepdims=True) + 1e-8)
    enc_embedding, dec_embedding = umap_viz.umap_embeddings(
        dense_W_enc_norm, dense_W_dec_norm, umap_neighbors
    )

    level_assignments = assign_levels(dense_indices)
    colors_by_level = []
    for level_idx in level_assignments:
        if 0 <= level_idx < len(LEVEL_COLORS):
            colors_by_level.append(LEVEL_COLORS[level_idx])
        else:
            colors_by_level.append('gray')
    level_legend = [(f'L{level}', LEVEL_COLORS[i]) for i, level in enumerate(MATRYOSHKA_LEVELS)]

    high_quality_pairs = [p for p in top_pairs if p.get('antipodality_score', 0) >= 0.8]
    pairs_for_routing = []
    for pair in high_quality_pairs[:20]:
        pair_with_levels = pair.copy()
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

    enc_segments = umap_viz.route_pairs_on_embedding(
        enc_embedding, dense_indices, pairs_for_routing, max_pairs=20
    )
    dec_segments = umap_viz.route_pairs_on_embedding(
        dec_embedding, dense_indices, pairs_for_routing, max_pairs=20
    )
    densities = _get_densities(results)
    return {
        'layer': layer,
        'enc_embedding': enc_embedding,
        'dec_embedding': dec_embedding,
        'colors_by_level': colors_by_level,
        'densities': densities[dense_indices],
        'antipodality_scores_mask': np.isfinite(dense_features['antipodality_scores']),
        'antipodality_scores': dense_features['antipodality_scores'][np.isfinite(dense_features['antipodality_scores'])],
        'enc_segments': enc_segments,
        'dec_segments': dec_segments,
        'level_legend': level_legend,
        'title_suffix': f' - Layer {layer}'
    }


def build_dense_focused_matrix_payload(results: dict, top_k: int = 50) -> dict:
    """Build clustered cosine matrices for the densest selected features."""
    layer = int(results["analysis_metadata"]["layer"])
    threshold = float(results["analysis_metadata"]["density_threshold"])
    densities = _get_densities(results)
    W_enc = results["analysis_metadata"]["W_enc"]
    W_dec = results["analysis_metadata"]["W_dec"]
    try:
        dense_indices, _ = analysis.dense_feature_indices(densities, threshold)
        ordered_indices, W_enc_ordered, W_dec_ordered, _ = analysis.select_topk_dense(
            dense_indices, densities, W_enc, W_dec, top_k, 'average'
        )
        actual_k = len(ordered_indices)
    except ValueError as error:
        return {'has_data': False, 'layer': layer, 'message': str(error)}

    C_enc = cosine_matrix(W_enc_ordered)
    C_dec = cosine_matrix(W_dec_ordered)
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
