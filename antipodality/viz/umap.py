"""
UMAP module

"""

from __future__ import annotations
from typing import Tuple, List, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from umap.umap_ import UMAP

# Set matplotlib for headless operation
plt.switch_backend("Agg")

# Apply house style
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "figure.dpi": 300, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15,
})

# UMAP visualization colors
WITHIN_LEVEL_COLOR = 'green'
CROSS_LEVEL_COLOR = 'red'
BACKGROUND_COLOR = 'lightgray'

def get_pair_style(is_within: bool) -> tuple[str, float, float]:
    """Get color, linewidth, alpha for pair routing."""
    if is_within:
        return WITHIN_LEVEL_COLOR, 2.0, 0.8
    else:
        return CROSS_LEVEL_COLOR, 1.5, 0.6


def umap_embeddings(W_enc_norm: np.ndarray, W_dec_norm: np.ndarray, n_neighbors: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute UMAP embeddings for encoder and decoder weight matrices.
    """
    n_features = W_enc_norm.shape[0]

    # Handle edge cases
    if n_features < 2:
        return np.zeros((n_features, 2)), np.zeros((n_features, 2))

    # Clamp neighbors parameter
    k = max(2, min(int(n_neighbors), n_features - 1))

    # Configure UMAP for SAE weight geometry
    reducer_enc = UMAP(n_neighbors=k, min_dist=0.1, metric="cosine", random_state=42, verbose=False)
    reducer_dec = UMAP(n_neighbors=k, min_dist=0.1, metric="cosine", random_state=42, verbose=False)

    # Compute embeddings
    enc_embedding = reducer_enc.fit_transform(W_enc_norm)
    dec_embedding = reducer_dec.fit_transform(W_dec_norm)

    return enc_embedding, dec_embedding


def route_pairs_on_embedding(embedding: np.ndarray, indices: np.ndarray, pairs: List[Dict], max_pairs: int = 20) -> List[Dict]:
    """
    Compute line segments for routing antipodal pairs on UMAP embedding.

    """
    if len(pairs) == 0:
        return []

    # Create feature position mapping
    feature_to_pos = {feat_idx: pos for pos, feat_idx in enumerate(indices)}

    line_segments = []

    # Process pairs up to max_pairs limit
    for pair in pairs[:max_pairs]:
        feat1_idx = pair['feature1_idx']
        feat2_idx = pair['feature2_idx']

        # Check if both features are present in the embedding
        if feat1_idx in feature_to_pos and feat2_idx in feature_to_pos:
            pos1 = feature_to_pos[feat1_idx]
            pos2 = feature_to_pos[feat2_idx]

            # Determine if this is a within-level or cross-level pair
            is_within = pair.get('is_within_level', False)
            if 'is_within_level' not in pair:
                # Fallback: check if levels are equal (if provided)
                feat1_level = pair.get('feature1_level')
                feat2_level = pair.get('feature2_level')
                is_within = feat1_level is not None and feat1_level == feat2_level

            # Style based on level relationship
            color, linewidth, alpha = get_pair_style(is_within)

            line_segment = {
                "x": [embedding[pos1, 0], embedding[pos2, 0]],
                "y": [embedding[pos1, 1], embedding[pos2, 1]],
                "color": color,
                "linewidth": linewidth,
                "alpha": alpha,
                "meta": pair.copy()  # Include all original pair fields
            }
            line_segments.append(line_segment)

    return line_segments


def plot_umap_analysis(payload: Dict, out_path: str) -> None:
    """
    make UMAP analysis figure.

    """
    # Extract data from payload
    enc_embedding = np.asarray(payload['enc_embedding'])
    dec_embedding = np.asarray(payload['dec_embedding'])
    colors_by_level = payload['colors_by_level']
    densities = np.asarray(payload['densities'])
    antipodality_scores_mask = np.asarray(payload['antipodality_scores_mask'])
    antipodality_scores = np.asarray(payload['antipodality_scores'])
    enc_segments = payload['enc_segments']
    dec_segments = payload['dec_segments']
    level_legend = payload['level_legend']
    title_suffix = payload['title_suffix']

    # Create 8-panel figure (2x4 layout)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'UMAP Geometric Analysis{title_suffix}', fontsize=16, fontweight='bold')

    # Flatten axes for easier indexing
    axes = axes.flatten()


    # 1. Encoder UMAP colored by Matryoshka level
    axes[0].scatter(enc_embedding[:, 0], enc_embedding[:, 1], c=colors_by_level,
                   s=30, alpha=0.7, edgecolors='white', linewidth=0.5)
    axes[0].set_title('Encoder UMAP - Colored by Matryoshka Level', fontweight='bold')
    axes[0].set_xlabel('UMAP 1', fontweight='bold')
    axes[0].set_ylabel('UMAP 2', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Add legend for levels
    if level_legend:
        for label, color in level_legend:
            axes[0].scatter([], [], c=color, label=label, s=50)
        axes[0].legend(loc='upper right', bbox_to_anchor=(1.14, 1), fontsize=7,
                      framealpha=0.9, markerscale=0.8)

    # 2. Encoder UMAP colored by density
    scatter2 = axes[1].scatter(enc_embedding[:, 0], enc_embedding[:, 1],
                              c=densities, s=30, alpha=0.7, cmap='viridis',
                              edgecolors='white', linewidth=0.5)
    plt.colorbar(scatter2, ax=axes[1], label='Activation Density', shrink=0.8)
    axes[1].set_title('Encoder UMAP - Colored by Density', fontweight='bold')
    axes[1].set_xlabel('UMAP 1', fontweight='bold')
    axes[1].set_ylabel('UMAP 2', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # 3. Encoder UMAP colored by antipodality score
    valid_embedding = enc_embedding[antipodality_scores_mask]
    scatter3 = axes[2].scatter(valid_embedding[:, 0], valid_embedding[:, 1],
                              c=antipodality_scores, s=30, alpha=0.7, cmap='plasma',
                              edgecolors='white', linewidth=0.5)
    plt.colorbar(scatter3, ax=axes[2], label='Antipodality Score', shrink=0.8)
    axes[2].set_title('Encoder UMAP - Colored by Antipodality', fontweight='bold')
    axes[2].set_xlabel('UMAP 1', fontweight='bold')
    axes[2].set_ylabel('UMAP 2', fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    # 4. Encoder UMAP with antipodal pair routing
    # Show all features as background
    axes[3].scatter(enc_embedding[:, 0], enc_embedding[:, 1],
                   c=BACKGROUND_COLOR, s=15, alpha=0.4, label='All Features')

    # Plot line segments
    within_plotted = False
    cross_plotted = False
    for segment in enc_segments:
        color = segment['color']
        if color == WITHIN_LEVEL_COLOR and not within_plotted:
            axes[3].plot(segment['x'], segment['y'], color=color,
                       alpha=segment['alpha'], linewidth=segment['linewidth'],
                       label='Within-level pairs')
            within_plotted = True
        elif color == CROSS_LEVEL_COLOR and not cross_plotted:
            axes[3].plot(segment['x'], segment['y'], color=color,
                       alpha=segment['alpha'], linewidth=segment['linewidth'],
                       label='Cross-level pairs')
            cross_plotted = True
        else:
            axes[3].plot(segment['x'], segment['y'], color=color,
                       alpha=segment['alpha'], linewidth=segment['linewidth'])

    axes[3].set_title(f'Encoder UMAP - Confirmed Pairs: {len(enc_segments)} (>=0.8)', fontweight='bold')
    axes[3].legend(fontsize=8)

    axes[3].set_xlabel('UMAP 1', fontweight='bold')
    axes[3].set_ylabel('UMAP 2', fontweight='bold')
    axes[3].grid(True, alpha=0.3)

    # 5. Decoder UMAP colored by Matryoshka level
    axes[4].scatter(dec_embedding[:, 0], dec_embedding[:, 1], c=colors_by_level,
                   s=30, alpha=0.7, edgecolors='white', linewidth=0.5)
    axes[4].set_title('Decoder UMAP - Colored by Matryoshka Level', fontweight='bold')
    axes[4].set_xlabel('UMAP 1', fontweight='bold')
    axes[4].set_ylabel('UMAP 2', fontweight='bold')
    axes[4].grid(True, alpha=0.3)

    # 6. Decoder UMAP colored by density
    scatter6 = axes[5].scatter(dec_embedding[:, 0], dec_embedding[:, 1],
                              c=densities, s=30, alpha=0.7, cmap='viridis',
                              edgecolors='white', linewidth=0.5)
    plt.colorbar(scatter6, ax=axes[5], label='Activation Density', shrink=0.8)
    axes[5].set_title('Decoder UMAP - Colored by Density', fontweight='bold')
    axes[5].set_xlabel('UMAP 1', fontweight='bold')
    axes[5].set_ylabel('UMAP 2', fontweight='bold')
    axes[5].grid(True, alpha=0.3)

    # 7. Decoder UMAP colored by antipodality score
    valid_embedding = dec_embedding[antipodality_scores_mask]
    scatter7 = axes[6].scatter(valid_embedding[:, 0], valid_embedding[:, 1],
                              c=antipodality_scores, s=30, alpha=0.7, cmap='plasma',
                              edgecolors='white', linewidth=0.5)
    plt.colorbar(scatter7, ax=axes[6], label='Antipodality Score', shrink=0.8)
    axes[6].set_title('Decoder UMAP - Colored by Antipodality', fontweight='bold')
    axes[6].set_xlabel('UMAP 1', fontweight='bold')
    axes[6].set_ylabel('UMAP 2', fontweight='bold')
    axes[6].grid(True, alpha=0.3)

    # 8. Decoder UMAP with antipodal pair routing
    # Show all features as background
    axes[7].scatter(dec_embedding[:, 0], dec_embedding[:, 1],
                   c=BACKGROUND_COLOR, s=15, alpha=0.4, label='All Features')

    # Plot line segments
    within_plotted = False
    cross_plotted = False
    for segment in dec_segments:
        color = segment['color']
        if color == WITHIN_LEVEL_COLOR and not within_plotted:
            axes[7].plot(segment['x'], segment['y'], color=color,
                       alpha=segment['alpha'], linewidth=segment['linewidth'],
                       label='Within-level pairs')
            within_plotted = True
        elif color == CROSS_LEVEL_COLOR and not cross_plotted:
            axes[7].plot(segment['x'], segment['y'], color=color,
                       alpha=segment['alpha'], linewidth=segment['linewidth'],
                       label='Cross-level pairs')
            cross_plotted = True
        else:
            axes[7].plot(segment['x'], segment['y'], color=color,
                       alpha=segment['alpha'], linewidth=segment['linewidth'])

    axes[7].set_title(f'Decoder UMAP - Confirmed Pairs: {len(dec_segments)} (>=0.8)', fontweight='bold')
    axes[7].legend(fontsize=8)

    axes[7].set_xlabel('UMAP 1', fontweight='bold')
    axes[7].set_ylabel('UMAP 2', fontweight='bold')
    axes[7].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close()