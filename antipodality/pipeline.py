"""
Thin orchestration layer for the antipodality analysis pipeline.

This module wires together loading -> analysis -> viz-payload build -> rendering -> JSON save.
All logic lives in other modules; this is pure orchestration/glue code.
"""

from __future__ import annotations
import csv
import json
import random
import traceback
from pathlib import Path
from typing import Dict, Optional
from .viz import umap as umap_viz

import numpy as np


# Local modules
from . import io
from . import analysis
from .viz import payloads
from .viz import plots
from .constants import DENSE_THRESHOLD, MATRYOSHKA_LEVELS
from .utils import prepare_for_json, MatryoshkaUtils


def export_antipodal_pairs_csv(pairs, densities, csv_path):
    """
    Export antipodal pairs data to csv
    """

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'pair_id', 'feature1_idx', 'feature2_idx',
            'feature1_density', 'feature2_density',
            'encoder_sim', 'decoder_sim', 'antipodal_score',
            'feature1_level', 'feature2_level'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, pair in enumerate(pairs, 1):
            # Get Matryoshka levels
            feat1_level = MatryoshkaUtils.get_level(pair['feature1_idx'])
            feat2_level = MatryoshkaUtils.get_level(pair['feature2_idx'])

            writer.writerow({
                'pair_id': i,
                'feature1_idx': pair['feature1_idx'],
                'feature2_idx': pair['feature2_idx'],
                'feature1_density': float(densities[pair['feature1_idx']]),
                'feature2_density': float(densities[pair['feature2_idx']]),
                'encoder_sim': pair.get('encoder_similarity', 'N/A'),
                'decoder_sim': pair.get('decoder_similarity', 'N/A'),
                'antipodal_score': float(pair['antipodality_score']),
                'feature1_level': feat1_level,
                'feature2_level': feat2_level
            })




def run(
    npz_path: str,
    layer: int,
    *,
    sae_repo: str = "gemma-2-2b-res-matryoshka-dc",
    out_dir: str = "antipodality_analysis",
    density_threshold: Optional[float] = None,
    top_k_pairs: int = 10,
    block_size: int = 2048,
    antipodal_only: bool = True,
    build_within_cross: bool = True,
    make_umap: bool = True,
    umap_neighbors: int = 15,
    rng_seed: int = 42
) -> Dict:
    """
    Run the full antipodality analysis pipeline.

    Generates comprehensive analysis with PNG visualizations, CSV data export,
    and JSON results. Output includes:
    - Global antipodality analysis (PNG)
    - Matryoshka hierarchy analysis (PNG)
    - Encoder-decoder scatter plot (PNG)
    - Within/cross-level comparison (PNG, optional)
    - UMAP geometric analysis (PNG, optional)
    - Antipodal pairs data (CSV)
    - Complete analysis results (JSON)

    Args:
        npz_path: Path to NPZ file with density data
        layer: Transformer layer number
        sae_repo: SAE repository name
        out_dir: Output directory for results
        density_threshold: Dense threshold (uses DENSE_THRESHOLD if None)
        top_k_pairs: Number of top pairs to analyze
        block_size: Block size for similarity computation
        antipodal_only: Whether to compute only antipodal scores
        build_within_cross: Whether to build within/cross-level analysis
        make_umap: Whether to generate UMAP analysis
        umap_neighbors: UMAP n_neighbors parameter
        rng_seed: Random seed for determinism

    Returns:
        dict: Summary with output paths and key statistics
    """
    try:
        # Use default threshold if not provided
        threshold = density_threshold if density_threshold is not None else DENSE_THRESHOLD 

        # Create output directory
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Stage 1: Deterministic seeding
        random.seed(rng_seed)
        np.random.seed(rng_seed)

        # Stage 2: I/O operations
        densities = io.load_density_data(npz_path)
        W_enc, W_dec, cfg = io.load_sae_weights(sae_repo, layer)

        print(f"Loaded SAE: d_sae={cfg['d_sae']}, d_in={cfg['d_in']}")
        print(f"Density data: {len(densities)} features, threshold={threshold}")

        # Stage 3: Analysis

        # Dense feature identification
        dense_indices, dense_stats = analysis.dense_feature_indices(densities, threshold)
        print(f"{len(dense_indices)} dense features ({len(dense_indices)/len(densities)*100:.1f}%)")

        # Antipodality scores for all features
        all_results = analysis.compute_antipodality_scores(
            W_enc, W_dec,
            feature_indices=None,
            top_k=top_k_pairs,
            block_size=block_size,
            antipodal_only=antipodal_only
        )

        # Antipodality scores for dense features only
        dense_results = analysis.compute_antipodality_scores(
            W_enc, W_dec,
            feature_indices=dense_indices,
            top_k=top_k_pairs,
            block_size=block_size,
            antipodal_only=antipodal_only
        )

        # Correlation analysis
        all_indices = all_results['feature_indices']
        all_scores = all_results['antipodality_scores']
        corr_results = analysis.spearman_corr(
            densities[all_indices], all_scores, threshold
        )

        # Matryoshka hierarchy analysis
        matry_results = analysis.analyze_matryoshka_hierarchy(
            dense_results['feature_indices'],
            dense_results['antipodality_scores'],
            dense_results['antipodal_partners']
        )

        # Top pairs analysis
        top_pairs = analysis.find_top_pairs(
            all_results['feature_indices'],
            all_results['antipodality_scores'],
            all_results['antipodal_partners'],
            top_k=10000,
            W_enc=W_enc,
            W_dec=W_dec
        )

        print(f"{len(all_indices)} total features, "
              f"correlation rho={corr_results['spearman_r']:.3f}")

        # Stage 4: Assemble results dictionary
        results = {
            'analysis_metadata': {
                'layer': layer,
                'sae_repo': sae_repo,
                'd_sae': cfg['d_sae'],
                'd_in': cfg['d_in'],
                'density_threshold': threshold,
                'densities': densities,  
                'W_enc': W_enc,  
                'W_dec': W_dec,  
                'total_features': len(densities),
                'dense_feature_count': len(dense_indices),
                'dense_feature_ratio': len(dense_indices) / len(densities)
            },
            'antipodality_analysis': {
                'all_features': {
                    'feature_indices': all_results['feature_indices'],
                    'antipodality_scores': all_results['antipodality_scores'],
                    'antipodal_partners': all_results['antipodal_partners'],
                    'stats': all_results['summary_stats']
                },
                'dense_features': {
                    'feature_indices': dense_results['feature_indices'],
                    'antipodality_scores': dense_results['antipodality_scores'],
                    'antipodal_partners': dense_results['antipodal_partners'],
                    'stats': dense_results['summary_stats']
                }
            },
            'correlation_analysis': corr_results,
            'matryoshka_analysis': matry_results,
            'top_antipodal_pairs': top_pairs[:100]  # Limit for JSON serialization
        }

        # Stage 5: Generate visualization payloads
        print("Building visualization payloads...")

        global_payload = payloads.antipodality_extraction(results)
        matry_payload = payloads.ext_dense_features(results)
        encdec_payload = payloads.build_enc_dec_scatter_payload(
            results, n_top_pairs=5000, n_random_pairs=5000
        )

        wc_payload = None
        if build_within_cross:
            wc_payload = payloads.build_within_cross_payload(results, top_k=10)

        # Stage 6: Render plots

        prefix = f"layer_{layer}"
        output_files = {}

        # Core plots
        global_path = out_path / f"{prefix}_global_antipodality.png"
        plots.plot_global_antipodality(global_payload, str(global_path))
        output_files['global_antipodality'] = str(global_path)

        matry_path = out_path / f"{prefix}_matryoshka_hierarchy.png"
        plots.plot_matryoshka_hierarchy(matry_payload, str(matry_path))
        output_files['matryoshka_hierarchy'] = str(matry_path)

        encdec_path = out_path / f"{prefix}_enc_dec_scatter.png"
        plots.plot_enc_dec_scatter(encdec_payload, str(encdec_path))
        output_files['enc_dec_scatter'] = str(encdec_path)

        if build_within_cross and wc_payload:
            wc_path = out_path / f"{prefix}_within_cross_level_comparison.png"
            plots.plot_within_cross(wc_payload, str(wc_path))
            output_files['within_cross_comparison'] = str(wc_path)

        # Additional matrix visualizations
        unbiased_payload = payloads.build_unbiased_antipodal_payload(results)
        unbiased_path = out_path / f"{prefix}_unbiased_antipodal_analysis.png"
        plots.plot_unbiased_antipodal_analysis(unbiased_payload, str(unbiased_path))
        output_files['unbiased_antipodal_analysis'] = str(unbiased_path)

        dense_matrix_payload = payloads.build_dense_focused_matrix_payload(results, top_k=50)
        dense_matrix_path = out_path / f"{prefix}_dense_focused_matrix.png"
        plots.plot_dense_focused_matrix(dense_matrix_payload, str(dense_matrix_path))
        output_files['dense_focused_matrix'] = str(dense_matrix_path)
        # Stage 7: Optional UMAP analysis
        if make_umap:
            # Build UMAP payload
            umap_payload = payloads.build_umap_payload(
                results, W_enc, W_dec, dense_indices, top_pairs, umap_neighbors
            )

            # Render UMAP analysis
            umap_path = out_path / f"{prefix}_umap_geometric_analysis.png"
            umap_viz.plot_umap_analysis(umap_payload, str(umap_path))
            output_files['umap_analysis'] = str(umap_path)

            print(f"{len(umap_payload['enc_segments'])} enc pairs routed, {len(umap_payload['dec_segments'])} dec pairs routed")


        # Stage 8: Export antipodal pairs to CSV (only high-quality pairs >=0.8)
        csv_path = out_path / f"antipodal_pairs_layer_{layer}.csv"
        high_quality_pairs_csv = [p for p in top_pairs if p.get('antipodality_score', 0) >= 0.8]
        export_antipodal_pairs_csv(high_quality_pairs_csv, densities, str(csv_path))
        output_files['antipodal_pairs_csv'] = str(csv_path)

        # Stage 9: Save JSON results

        # Remove heavy arrays for JSON serialization
        json_results = results.copy()
        json_results['analysis_metadata'] = {k: v for k, v in results['analysis_metadata'].items()
                                           if k not in ['densities', 'W_enc', 'W_dec']}

        json_path = out_path / f"antipodality_analysis_layer_{layer}.json"
        with open(json_path, 'w') as f:
            json.dump(prepare_for_json(json_results), f, indent=2)
        output_files['json_results'] = str(json_path)

        # Stage 10: Build summary
        summary = {
            'layer': layer,
            'output_files': output_files,
            'key_statistics': {
                'total_features': len(densities),
                'dense_features': len(dense_indices),
                'dense_ratio': len(dense_indices) / len(densities),
                'correlation_rho': corr_results['spearman_r'],
                'correlation_p': corr_results['spearman_p'],
                'mean_antipodality_all': float(np.mean(all_scores[np.isfinite(all_scores)])) if np.any(np.isfinite(all_scores)) else 0.0,
                'mean_antipodality_dense': float(np.mean(dense_results['antipodality_scores'][np.isfinite(dense_results['antipodality_scores'])])) if np.any(np.isfinite(dense_results['antipodality_scores'])) else 0.0,
                'top_antipodal_pairs': len([p for p in top_pairs if p.get('antipodality_score', 0) >= 0.8])
            },
            'counts': {
                'matryoshka_levels': {level: matry_results['level_analysis'][level]['count']
                                    for level in MATRYOSHKA_LEVELS if level in matry_results['level_analysis']}
            }
        }

        return summary

    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        print(traceback.format_exc())
        raise RuntimeError(f"Antipodality analysis pipeline failed: {e}") from e