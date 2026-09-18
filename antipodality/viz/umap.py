"""UMAP embeddings and paired encoder/decoder plots."""

from __future__ import annotations
from typing import Tuple, List, Dict
import numpy as np
import matplotlib.pyplot as plt
from umap.umap_ import UMAP

plt.switch_backend("Agg")
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "figure.dpi": 300, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15,
})

WITHIN_LEVEL_COLOR = 'green'
CROSS_LEVEL_COLOR = 'red'
BACKGROUND_COLOR = 'lightgray'


def get_pair_style(is_within: bool) -> tuple[str, float, float]:
    if is_within:
        return WITHIN_LEVEL_COLOR, 2.0, 0.8
    return CROSS_LEVEL_COLOR, 1.5, 0.6


def umap_embeddings(W_enc_norm: np.ndarray, W_dec_norm: np.ndarray, n_neighbors: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fit separate, fixed-seed cosine embeddings for encoder and decoder directions."""
    n_features = W_enc_norm.shape[0]
    if n_features < 2:
        return np.zeros((n_features, 2)), np.zeros((n_features, 2))

    k = max(2, min(int(n_neighbors), n_features - 1))
    reducer_enc = UMAP(n_neighbors=k, min_dist=0.1, metric="cosine", random_state=42, verbose=False)
    reducer_dec = UMAP(n_neighbors=k, min_dist=0.1, metric="cosine", random_state=42, verbose=False)
    enc_embedding = reducer_enc.fit_transform(W_enc_norm)
    dec_embedding = reducer_dec.fit_transform(W_dec_norm)
    return enc_embedding, dec_embedding


def route_pairs_on_embedding(embedding: np.ndarray, indices: np.ndarray, pairs: List[Dict], max_pairs: int = 20) -> List[Dict]:
    """Build segments for pairs whose endpoints are present in the embedding."""
    if len(pairs) == 0:
        return []

    feature_to_pos = {feat_idx: pos for pos, feat_idx in enumerate(indices)}
    line_segments = []
    for pair in pairs[:max_pairs]:
        feat1_idx = pair['feature1_idx']
        feat2_idx = pair['feature2_idx']
        if feat1_idx not in feature_to_pos or feat2_idx not in feature_to_pos:
            continue
        pos1 = feature_to_pos[feat1_idx]
        pos2 = feature_to_pos[feat2_idx]
        is_within = pair.get('is_within_level', False)
        if 'is_within_level' not in pair:
            feat1_level = pair.get('feature1_level')
            feat2_level = pair.get('feature2_level')
            is_within = feat1_level is not None and feat1_level == feat2_level
        color, linewidth, alpha = get_pair_style(is_within)
        line_segments.append({
            "x": [embedding[pos1, 0], embedding[pos2, 0]],
            "y": [embedding[pos1, 1], embedding[pos2, 1]],
            "color": color,
            "linewidth": linewidth,
            "alpha": alpha,
            "meta": pair.copy()
        })
    return line_segments


def _plot_segment_panel(ax, embedding, segments, title):
    ax.scatter(embedding[:, 0], embedding[:, 1], c=BACKGROUND_COLOR, s=15, alpha=0.4, label='All Features')
    within_plotted = False
    cross_plotted = False
    for segment in segments:
        color = segment['color']
        kwargs = dict(color=color, alpha=segment['alpha'], linewidth=segment['linewidth'])
        if color == WITHIN_LEVEL_COLOR and not within_plotted:
            ax.plot(segment['x'], segment['y'], label='Within-level pairs', **kwargs)
            within_plotted = True
        elif color == CROSS_LEVEL_COLOR and not cross_plotted:
            ax.plot(segment['x'], segment['y'], label='Cross-level pairs', **kwargs)
            cross_plotted = True
        else:
            ax.plot(segment['x'], segment['y'], **kwargs)
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_xlabel('UMAP 1', fontweight='bold')
    ax.set_ylabel('UMAP 2', fontweight='bold')
    ax.grid(True, alpha=0.3)


def _plot_embedding_row(axes, name, embedding, colors, densities, score_mask, scores, segments, level_legend=()):
    axes[0].scatter(embedding[:, 0], embedding[:, 1], c=colors,
                    s=30, alpha=0.7, edgecolors='white', linewidth=0.5)
    axes[0].set_title(f'{name} UMAP - Colored by Matryoshka Level', fontweight='bold')
    axes[0].set_xlabel('UMAP 1', fontweight='bold')
    axes[0].set_ylabel('UMAP 2', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    if level_legend:
        for label, color in level_legend:
            axes[0].scatter([], [], c=color, label=label, s=50)
        axes[0].legend(loc='upper right', bbox_to_anchor=(1.14, 1), fontsize=7,
                       framealpha=0.9, markerscale=0.8)

    density_scatter = axes[1].scatter(embedding[:, 0], embedding[:, 1], c=densities,
                                      s=30, alpha=0.7, cmap='viridis', edgecolors='white', linewidth=0.5)
    plt.colorbar(density_scatter, ax=axes[1], label='Activation Density', shrink=0.8)
    axes[1].set_title(f'{name} UMAP - Colored by Density', fontweight='bold')
    axes[1].set_xlabel('UMAP 1', fontweight='bold')
    axes[1].set_ylabel('UMAP 2', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    valid_embedding = embedding[score_mask]
    score_scatter = axes[2].scatter(valid_embedding[:, 0], valid_embedding[:, 1], c=scores,
                                    s=30, alpha=0.7, cmap='plasma', edgecolors='white', linewidth=0.5)
    plt.colorbar(score_scatter, ax=axes[2], label='Antipodality Score', shrink=0.8)
    axes[2].set_title(f'{name} UMAP - Colored by Antipodality', fontweight='bold')
    axes[2].set_xlabel('UMAP 1', fontweight='bold')
    axes[2].set_ylabel('UMAP 2', fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    _plot_segment_panel(axes[3], embedding, segments,
                        f'{name} UMAP - Confirmed Pairs: {len(segments)} (>=0.8)')


def plot_umap_analysis(payload: Dict, out_path: str) -> None:
    """Plot encoder and decoder embeddings with level, density, score, and pair views."""
    enc_embedding = np.asarray(payload['enc_embedding'])
    dec_embedding = np.asarray(payload['dec_embedding'])
    colors_by_level = payload['colors_by_level']
    densities = np.asarray(payload['densities'])
    score_mask = np.asarray(payload['antipodality_scores_mask'])
    scores = np.asarray(payload['antipodality_scores'])
    enc_segments = payload['enc_segments']
    dec_segments = payload['dec_segments']
    level_legend = payload['level_legend']
    title_suffix = payload['title_suffix']

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'UMAP Geometric Analysis{title_suffix}', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    _plot_embedding_row(
        axes[:4], 'Encoder', enc_embedding, colors_by_level, densities,
        score_mask, scores, enc_segments, level_legend,
    )
    _plot_embedding_row(
        axes[4:], 'Decoder', dec_embedding, colors_by_level, densities,
        score_mask, scores, dec_segments,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close()
