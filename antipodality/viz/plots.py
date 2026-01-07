"""
Visualizations
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Sequence
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import seaborn as sns

from antipodality.constants import MATRYOSHKA_LEVELS, LEVEL_COLORS
from antipodality.types import EncDecScatterPayload, WithinCrossPayload


def _require(payload: dict, keys: list[str]):
    """Validate that payload contains all required keys."""
    missing = [k for k in keys if k not in payload]
    if missing:
        raise KeyError(f"Missing payload keys: {missing}")

# Set matplotlib for headless operation
plt.switch_backend("Agg")

# Apply house style with seaborn
sns.set_theme(
    context="notebook",
    style="whitegrid",
    font_scale=1.0,
    rc={
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.15,
        "figure.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    },
)

# Color constants for cross-level transitions
TRANSITION_COLORS = {
    '128->512': '#3498DB',    # Blue - small to medium
    '128->2048': '#5DADE2',   # Light blue - small to large
    '128->8192': '#85C1E9',   # Very light blue - small to very large
    '512->2048': '#F39C12',   # Orange - medium to large
    '512->8192': '#F7DC6F',   # Light orange - medium to very large
    '2048->8192': '#27AE60',  # Green - large to very large
    '2048->32768': '#58D68D', # Light green - large to max
    '8192->32768': '#E74C3C', # Red - very large to max
}

# Fallback colors by source level
LEVEL_FALLBACK_COLORS = {
    128: '#85C1E9',   # Light blue fallback
    512: '#F8C471',   # Light orange fallback
    2048: '#82E0AA',  # Light green fallback
    8192: '#F1948A',  # Light red fallback
}

def get_transition_color(feature1_level: int, feature2_level: int) -> str:
    """Get color for cross-level transition."""
    transition = f'{feature1_level}->{feature2_level}'
    if transition in TRANSITION_COLORS:
        return TRANSITION_COLORS[transition]
    # Fallback based on source level
    return LEVEL_FALLBACK_COLORS.get(feature1_level, '#D5DBDB')  # Gray fallback

def get_level_color(level: int) -> str:
    """Get color for Matryoshka level."""
    try:
        level_idx = MATRYOSHKA_LEVELS.index(level)
        return LEVEL_COLORS[level_idx]
    except (ValueError, IndexError):
        return 'gray'


# Seaborn helper functions
@dataclass
class HeatmapSpec:
    """Configuration for seaborn heatmaps."""
    cmap: str = "RdBu_r"
    vmin: Optional[float] = -1.0
    vmax: Optional[float] = 1.0
    annot: bool = False
    square: bool = True
    cbar: bool = True


def draw_heatmap(ax: Axes, M, spec: HeatmapSpec):
    """seaborn heatmaps"""
    hm = sns.heatmap(
        M, ax=ax,
        cmap=spec.cmap, vmin=spec.vmin, vmax=spec.vmax,
        annot=spec.annot, square=spec.square, cbar=spec.cbar,
        linewidths=.5, linecolor='lightgray'
    )
    # seaborn attaches the colorbar to collections[0].colorbar if cbar=True
    cbar = getattr(hm.collections[0], "colorbar", None) if spec.cbar else None
    return hm, cbar


def add_shared_colorbar(fig: Figure, mappable, rect=(0.92, 0.15, 0.02, 0.7), label="Cosine Similarity"):
    """Use one colorbar"""
    cax = fig.add_axes(rect)
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(label, fontsize=12, fontweight="bold")
    return cb


def violin(ax: Axes, data: Sequence[Sequence[float]], labels: Sequence[str]):
    """Violin plot"""
    parts = ax.violinplot(data, positions=range(1, len(data) + 1))
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontweight="bold")

    # Color the violin plots to match original style
    if len(parts['bodies']) >= 2:
        parts['bodies'][0].set_facecolor('lightcoral')  # Sparse
        parts['bodies'][1].set_facecolor('lightgreen')  # Dense
        parts['bodies'][0].set_alpha(0.7)
        parts['bodies'][1].set_alpha(0.7)

    return parts


def hist(ax: Axes, values, bins=50, label_mean=True, label_median=True):
    """Histogram"""
    ax.hist(values, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
    if label_mean:
        m = float(np.mean(values))
        ax.axvline(m, color="red", lw=2, label=f"Mean = {m:.3f}")
    if label_median:
        med = float(np.median(values))
        ax.axvline(med, color="orange", ls="--", lw=2, label=f"Median = {med:.3f}")


# Heatmap overlay helpers
def add_pair_rect(ax: Axes, r: int, c: int, color="gold", lw=3, symmetric=True):
    """Add rectangle overlay for antipodal pairs."""
    ax.add_patch(Rectangle((c, r), 1, 1, fill=False, edgecolor=color, lw=lw))
    if symmetric and r != c:
        ax.add_patch(Rectangle((r, c), 1, 1, fill=False, edgecolor=color, lw=lw))


def add_dense_dots(ax: Axes, r: int, c: int, size=50):
    """Add dots for dense features."""
    ax.scatter([c, r], [r, c], c="black", s=size, marker="o")


def plot_global_antipodality(payload: dict, out_path: str) -> None:
    """
    Create matplotlib visualization from global antipodality payload.
    """
    _require(payload, [
        "layer", "threshold", "level_ids", "valid_densities", "valid_scores",
        "correlation_results", "valid_scores_all", "dense_scores_all",
        "sparse_scores_all", "level_counts"
    ])

    layer = payload["layer"]
    thr = payload["threshold"]
    level_ids = payload["level_ids"]
    valid_densities = payload["valid_densities"]
    valid_scores = payload["valid_scores"]
    corr = payload["correlation_results"]
    valid_scores_all = payload["valid_scores_all"]
    dense_scores_all = payload["dense_scores_all"]
    sparse_scores_all = payload["sparse_scores_all"]
    level_counts = payload["level_counts"]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(f"Global Antipodality Analysis - Layer {layer}", fontsize=16, fontweight="bold")

    # scatter (still matplotlib)
    for i, level in enumerate(MATRYOSHKA_LEVELS):
        m = (level_ids == i)
        if np.any(m):
            ax1.scatter(valid_densities[m], valid_scores[m], s=25, alpha=0.7,
                        c=LEVEL_COLORS[i], edgecolors="white", linewidth=0.5, label=f"L{level}")
    ax1.axvline(thr, color="red", ls="--", lw=2, alpha=0.7, label=f"Dense threshold ({thr:g})")
    ax1.set_xlabel("Activation Density", fontweight="bold")
    ax1.set_ylabel("Antipodality Score", fontweight="bold")
    ax1.set_title(f"Density vs Antipodality by Matryoshka Level\n"
                  f"Spearman rho = {corr['spearman_r']:.3f} (p = {corr['spearman_p']:.4f})",
                  fontweight="bold")
    ax1.legend(loc="upper right", bbox_to_anchor=(1.12, 1), fontsize=8, framealpha=0.9)

    # histogram (seaborn)
    hist(ax2, np.asarray(valid_scores_all), bins=50)
    ax2.set_xlabel("Antipodality Score", fontweight="bold")
    ax2.set_ylabel("Frequency (log scale)", fontweight="bold")
    ax2.set_title("Antipodality Score Distribution (All Features)", fontweight="bold")
    ax2.set_yscale("log")
    ax2.legend()

    # violin (seaborn)
    violin(ax3, [np.asarray(sparse_scores_all), np.asarray(dense_scores_all)], ["Sparse", "Dense"])
    ax3.set_yscale("log")
    ax3.set_ylabel("Antipodality Score (log scale)", fontweight="bold")
    ax3.set_title("Antipodality Distribution: Dense vs Sparse Features", fontweight="bold")

    # bar (matplotlib, but simple)
    xs = list(range(len(MATRYOSHKA_LEVELS)))
    counts = [level_counts[level] for level in MATRYOSHKA_LEVELS]
    colors = [LEVEL_COLORS[i] for i in range(len(MATRYOSHKA_LEVELS))]
    bars = ax4.bar(xs, counts, color=colors, alpha=0.8, edgecolor="black", linewidth=1)
    ax4.set_xlabel("Matryoshka Level", fontweight="bold")
    ax4.set_ylabel("Number of Dense Features", fontweight="bold")
    ax4.set_title("Dense Features by Matryoshka Level", fontweight="bold")
    ax4.set_xticks(xs)
    ax4.set_xticklabels([str(l) for l in MATRYOSHKA_LEVELS])
    for i, c in enumerate(counts):
        if c > 0:
            ax4.text(xs[i], c + max(0.5, 0.02 * max(counts)), str(c), ha="center", va="bottom", fontweight="bold")

    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close()


def plot_matryoshka_hierarchy(payload: dict, out_path: str) -> None:
    """
    Create matplotlib visualization from Matryoshka hierarchy payload.


    """
    _require(payload, ["layer", "level_score_groups", "level_labels", "level_colors"])

    layer = payload["layer"]
    level_score_groups = payload["level_score_groups"]
    level_labels = payload["level_labels"]
    level_colors = payload["level_colors"]

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(f"Matryoshka Hierarchy Analysis - Layer {layer}", fontsize=16, fontweight="bold")

    box_plot = ax1.boxplot(level_score_groups, labels=level_labels, patch_artist=True)
    for patch, color in zip(box_plot["boxes"], level_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_xlabel("Matryoshka Level", fontweight="bold")
    ax1.set_ylabel("Antipodality Score (log scale)", fontweight="bold")
    ax1.set_title("Antipodality Distribution by Matryoshka Level", fontweight="bold")
    ax1.set_yscale("log")

    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close()


def plot_enc_dec_scatter(payload: EncDecScatterPayload, out_path: str) -> None:
    """
    Create encoder-decoder scatter plot from payload.


    """
    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Consistent color scale and sequential colormap
    norm = Normalize(vmin=0.0, vmax=1.0)

    # Create scatter plot with colors based on relationship strength
    scatter = ax.scatter(payload.enc_sim, payload.dec_sim,
                    c=payload.pair_score, cmap='plasma', norm=norm,
                    alpha=0.8, s=18, edgecolors='none')

    # Add reference lines at x=0 and y=0
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Set axis properties
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel(r'sim(W$_{\mathrm{enc}}^{(i)}$, W$_{\mathrm{enc}}^{(j)}$)', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'sim(W$_{\mathrm{dec}}^{(i)}$, W$_{\mathrm{dec}}^{(j)}$)', fontsize=12, fontweight='bold')

    layer = payload.layer
    ax.set_title(f'Layer {layer}: Enc vs Dec Antipodalities', fontsize=14, fontweight='bold')

    # Add grid and set aspect
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', shrink=0.8, location='right')
    cbar.set_label('Relationship Strength (Antipodal & Synonym)', fontsize=12, fontweight='bold')
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(labelsize=10)

    # Layout and save
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close()


def plot_within_cross(payload: WithinCrossPayload, out_path: str) -> None:
    """
    create within vs cross-level comparison bar charts from payload.
    """
    # Create dual bar chart visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    layer = payload.layer
    fig.suptitle(f'Within-Level vs Cross-Level Comparison (Pairs & Candidates) Layer {layer}', fontsize=16, fontweight='bold')

    # Plot 1: Top Within-Level Pairs
    within_labels = [pair['pair_label'] for pair in payload.top_within]
    within_scores = [pair['antipodality_score'] for pair in payload.top_within]
    within_colors = [get_level_color(pair['feature1_level']) for pair in payload.top_within]

    bars1 = ax1.bar(range(len(within_scores)), within_scores, color=within_colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_xlabel('Feature Index Pairs', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Antipodality Score', fontsize=12, fontweight='bold')

    within_confirmed_count = sum(1 for p in payload.top_within if p['antipodality_score'] >= 0.8)
    within_candidate_count = len(payload.top_within) - within_confirmed_count
    ax1.set_title(f'Top {len(payload.top_within)} Within-Level: {within_confirmed_count} pairs (>=0.8), {within_candidate_count} candidates', fontsize=12, fontweight='bold')

    ax1.set_xticks(range(len(within_labels)))
    ax1.set_xticklabels(within_labels, rotation=0, ha='center')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, score in zip(bars1, within_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)

    # Fix y-axis padding to prevent value labels from hitting grid lines
    current_ylim = ax1.get_ylim()
    ax1.set_ylim(current_ylim[0], current_ylim[1] * 1.1)

    # Add legend for Matryoshka levels
    level_handles = []
    for level in MATRYOSHKA_LEVELS:
        if any(pair['feature1_level'] == level for pair in payload.top_within):
            level_handles.append(plt.Rectangle((0,0),1,1, color=get_level_color(level), alpha=0.8, label=f'Level {level}'))
    if level_handles:
        ax1.legend(handles=level_handles, loc='upper right', fontsize=10)

    # Plot 2: Top Cross-Level Pairs
    cross_labels = [pair['pair_label'] for pair in payload.top_cross]
    cross_scores = [pair['antipodality_score'] for pair in payload.top_cross]

    # Build transition labels and colors for each pair
    cross_colors = []
    transition_legend_items = {}

    for pair in payload.top_cross:
        color = get_transition_color(pair['feature1_level'], pair['feature2_level'])
        cross_colors.append(color)

        # Track unique transitions for legend
        transition = f"{pair['feature1_level']}->{pair['feature2_level']}"
        if transition not in transition_legend_items:
            transition_legend_items[transition] = color

    bars2 = ax2.bar(range(len(cross_scores)), cross_scores, color=cross_colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_xlabel('Feature Index Pairs', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Antipodality Score', fontsize=12, fontweight='bold')

    cross_confirmed_count = sum(1 for p in payload.top_cross if p['antipodality_score'] >= 0.8)
    cross_candidate_count = len(payload.top_cross) - cross_confirmed_count
    ax2.set_title(f'Top {len(payload.top_cross)} Cross-Level: {cross_confirmed_count} pairs (>=0.8), {cross_candidate_count} candidates', fontsize=12, fontweight='bold')

    ax2.set_xticks(range(len(cross_labels)))
    ax2.set_xticklabels(cross_labels, rotation=0, ha='center')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, score in zip(bars2, cross_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)

    # Fix y-axis padding
    current_ylim = ax2.get_ylim()
    ax2.set_ylim(current_ylim[0], current_ylim[1] * 1.1)

    # Add legend for transition types
    transition_handles = []
    for transition, color in transition_legend_items.items():
        transition_handles.append(plt.Rectangle((0,0),1,1, color=color, alpha=0.8, label=transition))
    if transition_handles:
        ax2.legend(handles=transition_handles, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    plt.close()




def plot_unbiased_antipodal_analysis(payload: dict, out_path: str) -> None:
    """
    Create unbiased antipodal analysis visualization from payload.

    Args:
        payload: Data payload from build_unbiased_antipodal_payload
        out_path: Output file path
    """
    _require(payload, [
        "layer", "C_enc", "C_dec", "ordered_indices", "high_scoring_pairs",
        "densities", "threshold", "n_pairs", "n_features"
    ])

    layer = payload["layer"]
    C_enc = payload["C_enc"]
    C_dec = payload["C_dec"]
    ordered_indices = payload["ordered_indices"]
    high_scoring_pairs = payload["high_scoring_pairs"]
    densities = payload["densities"]
    threshold = payload["threshold"]
    n_pairs = payload["n_pairs"]
    n_features = payload["n_features"]

    # Create mapping from original feature index to matrix position
    feature_to_pos = {feat_idx: pos for pos, feat_idx in enumerate(ordered_indices)}

    # Create enhanced figure with improved layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f"Complete Antipodal Structure Analysis - Layer {layer}\n" +
                f"{n_pairs} pairs with scores >= 0.8",
                fontsize=18, fontweight='bold', y=0.95)

    # Plot similarity matrices
    im1 = ax1.imshow(C_enc, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="equal")
    ax1.set_title("Encoder Cosine Similarity Matrix", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Feature Index", fontsize=14)
    ax1.set_ylabel("Feature Index", fontsize=14)

    im2 = ax2.imshow(C_dec, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="equal")
    ax2.set_title("Decoder Cosine Similarity Matrix", fontsize=16, fontweight='bold')
    ax2.set_xlabel("Feature Index", fontsize=14)
    ax2.set_ylabel("Feature Index", fontsize=14)

    # Add feature index labels (using clustered order)
    feature_labels = [str(idx) for idx in ordered_indices]
    for ax in [ax1, ax2]:
        ax.set_xticks(range(n_features))
        ax.set_yticks(range(n_features))
        ax.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(feature_labels, fontsize=8)

    # Classify features by density for dense marking
    feature_densities = densities[ordered_indices]
    dense_features = set(ordered_indices[feature_densities > threshold])

    # Add overlays for antipodal pairs
    antipodal_pair_count = 0
    dense_dense_pair_count = 0

    for pair in high_scoring_pairs:
        feat1_idx = pair['feature1_idx']
        feat2_idx = pair['feature2_idx']

        if feat1_idx in feature_to_pos and feat2_idx in feature_to_pos:
            pos1 = feature_to_pos[feat1_idx]
            pos2 = feature_to_pos[feat2_idx]

            # Add yellow squares for all antipodal pairs
            for ax in [ax1, ax2]:
                ax.add_patch(Rectangle((pos1-0.5, pos2-0.5), 1, 1, fill=False,
                                     edgecolor='yellow', linewidth=3))
                ax.add_patch(Rectangle((pos2-0.5, pos1-0.5), 1, 1, fill=False,
                                     edgecolor='yellow', linewidth=3))

            # Add black dots for dense-dense pairs
            if feat1_idx in dense_features and feat2_idx in dense_features:
                for ax in [ax1, ax2]:
                    ax.scatter([pos1, pos2], [pos2, pos1], c='black', s=50, marker='o')
                dense_dense_pair_count += 1

            antipodal_pair_count += 1

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.10, 0.02, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label('Cosine Similarity', fontsize=14, fontweight='bold')

    # Custom legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, fill=False, edgecolor='yellow', linewidth=3,
                label=f'Antipodal Pairs'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8,
            label=f'Dense Pairs')
    ]

    ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0, 1), fontsize=12)

    plt.tight_layout(rect=[0, 0, 0.9, 0.94])
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
    plt.close()


def plot_dense_focused_matrix(payload: dict, out_path: str) -> None:
    """
    Create dense-focused matrix visualization from payload.

    Args:
        payload: Data payload from build_dense_focused_matrix_payload
        out_path: Output file path
    """
    _require(payload, ["layer", "actual_k", "C_enc", "C_dec", "antipodal_pairs", "antipodal_threshold", "n_antipodal_pairs"])

    layer = payload["layer"]
    k = payload["actual_k"]
    C_enc = payload["C_enc"]
    C_dec = payload["C_dec"]
    pairs = payload["antipodal_pairs"]
    thr = payload["antipodal_threshold"]
    n_antipodal_pairs = payload["n_antipodal_pairs"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Geometric Analysis of Top {k} Dense Features - Layer {layer}",
                 fontsize=16, fontweight="bold", y=1.02)

    spec = HeatmapSpec(cmap="RdBu_r", vmin=-1, vmax=1, square=True, cbar=False)
    hm1, _ = draw_heatmap(ax1, C_enc, spec)
    hm2, _ = draw_heatmap(ax2, C_dec, spec)

    ax1.set_title("Encoder Cosine Similarity Matrix", fontsize=12, fontweight="bold")
    ax2.set_title("Decoder Cosine Similarity Matrix", fontsize=12, fontweight="bold")
    for ax in (ax1, ax2):
        ax.set_xlabel("Feature Index (Ranked by Density)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Feature Index (Ranked by Density)", fontsize=10, fontweight="bold")

    # overlays
    for p in pairs:
        r, c = p["matrix_i"], p["matrix_j"]
        add_pair_rect(ax1, r, c, color="gold", lw=3)
        add_pair_rect(ax2, r, c, color="gold", lw=3)

    # one shared colorbar using the second mappable
    add_shared_colorbar(fig, hm2.collections[0], label="Cosine Similarity")

    # legend
    fig.legend(
        handles=[Rectangle((0,0),1,1, fill=False, edgecolor='gold', lw=3,
                           label=f'Antipodal Pairs (score >= {thr})')],
        loc="upper left", bbox_to_anchor=(0.02, 0.95), fontsize=10, frameon=True
    )

    # Use tight_layout to adjust subplot params
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    plt.close()

