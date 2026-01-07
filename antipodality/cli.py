"""
Command-line interface for the antipodality analysis pipeline.
"""

import argparse
import sys
from pathlib import Path

from . import pipeline


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run antipodality analysis on dense SAE features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Positional arguments
    parser.add_argument(
        "npz_path",
        help="Path to NPZ file containing activation density data"
    )

    # Required arguments
    parser.add_argument(
        "--layer", "-l",
        type=int,
        required=True,
        help="Transformer layer number to analyze"
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--sae-repo",
        default="gemma-2-2b-res-matryoshka-dc",
        help="SAE repository name"
    )

    parser.add_argument(
        "--out-dir",
        default="antipodality_analysis",
        help="Output directory for results"
    )

    parser.add_argument(
        "--density-threshold",
        type=float,
        help="Dense feature threshold (uses default from constants if not specified)"
    )

    parser.add_argument(
        "--top-k-pairs",
        type=int,
        default=10,
        help="Number of top antipodal pairs to analyze"
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=2048,
        help="Block size for similarity computation"
    )

    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic results"
    )

    # Boolean flags
    parser.add_argument(
        "--no-antipodal-only",
        action="store_true",
        help="Compute both antipodal and synonym scores (default: antipodal only)"
    )

    parser.add_argument(
        "--no-within-cross",
        action="store_true",
        help="Skip within-level vs cross-level comparison analysis"
    )

    parser.add_argument(
        "--no-umap",
        action="store_true",
        help="Skip UMAP geometric analysis"
    )

    args = parser.parse_args()

    # Validate inputs
    npz_path = Path(args.npz_path)
    if not npz_path.exists():
        print(f"Error: NPZ file not found: {npz_path}")
        sys.exit(1)

    # Convert boolean flags
    antipodal_only = not args.no_antipodal_only
    build_within_cross = not args.no_within_cross
    make_umap = not args.no_umap

    # Run the pipeline
    summary = pipeline.run(
        npz_path=str(npz_path),
        layer=args.layer,
        sae_repo=args.sae_repo,
        out_dir=args.out_dir,
        density_threshold=args.density_threshold,
        top_k_pairs=args.top_k_pairs,
        block_size=args.block_size,
        antipodal_only=antipodal_only,
        build_within_cross=build_within_cross,
        make_umap=make_umap,
        umap_neighbors=args.umap_neighbors,
        rng_seed=args.seed
        )

    # Print results summary

    for name, path in summary['output_files'].items():
        print(f"  {name}: {path}")

    stats = summary['key_statistics']
    print(f"\nKey Statistics:")
    print(f"  Total features: {stats['total_features']:,}")
    print(f"  Dense features: {stats['dense_features']:,} ({stats['dense_ratio']:.1%})")
    print(f"  Density-antipodality correlation: rho = {stats['correlation_rho']:.3f} (p = {stats['correlation_p']:.4f})")
    print(f"  Mean antipodality (all): {stats['mean_antipodality_all']:.4f}")
    print(f"  Mean antipodality (dense): {stats['mean_antipodality_dense']:.4f}")
    print(f"  Top antipodal pairs (>=0.8): {stats['top_antipodal_pairs']}")

    if 'matryoshka_levels' in summary['counts']:
        print(f"\nMatryoshka Level Distribution:")
        for level, count in summary['counts']['matryoshka_levels'].items():
            print(f"  Level {level}: {count} dense features")

    print(f"\nAll results saved to: {args.out_dir}/")
    print("Analysis complete!")

if __name__ == "__main__":
    main()