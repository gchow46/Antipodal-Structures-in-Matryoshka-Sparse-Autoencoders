"""Command-line entry point for the analysis pipeline."""

import argparse
import sys
from pathlib import Path

from . import pipeline


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Run antipodality analysis on dense SAE features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("npz_path", help="Path to NPZ file containing activation density data")
    parser.add_argument(
        "--layer", "-l", type=int, required=True,
        help="Transformer layer number to analyze"
    )
    parser.add_argument(
        "--sae-repo", default="gemma-2-2b-res-matryoshka-dc",
        help="SAE repository name"
    )
    parser.add_argument("--out-dir", default="antipodality_analysis", help="Output directory for results")
    parser.add_argument(
        "--density-threshold", type=float,
        help="Dense feature threshold (uses default from constants if not specified)"
    )
    parser.add_argument(
        "--top-k-pairs", type=int, default=10,
        help="Number of top antipodal pairs to analyze"
    )
    parser.add_argument(
        "--block-size", type=int, default=2048,
        help="Block size for similarity computation"
    )
    parser.add_argument("--umap-neighbors", type=int, default=15, help="UMAP n_neighbors parameter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic results")
    parser.add_argument(
        "--no-antipodal-only", action="store_true",
        help="Compute both antipodal and synonym scores (default: antipodal only)"
    )
    parser.add_argument(
        "--no-within-cross", action="store_true",
        help="Skip within-level vs cross-level comparison analysis"
    )
    parser.add_argument("--no-umap", action="store_true", help="Skip UMAP geometric analysis")
    return parser


def _print_summary(summary, out_dir):
    for name, path in summary['output_files'].items():
        print(f"  {name}: {path}")

    stats = summary['key_statistics']
    print("\nKey Statistics:")
    print(f"  Total features: {stats['total_features']:,}")
    print(f"  Dense features: {stats['dense_features']:,} ({stats['dense_ratio']:.1%})")
    print(f"  Density-antipodality correlation: rho = {stats['correlation_rho']:.3f} (p = {stats['correlation_p']:.4f})")
    print(f"  Mean antipodality (all): {stats['mean_antipodality_all']:.4f}")
    print(f"  Mean antipodality (dense): {stats['mean_antipodality_dense']:.4f}")
    print(f"  Top antipodal pairs (>=0.8): {stats['top_antipodal_pairs']}")

    if 'matryoshka_levels' in summary['counts']:
        print("\nMatryoshka Level Distribution:")
        for level, count in summary['counts']['matryoshka_levels'].items():
            print(f"  Level {level}: {count} dense features")

    print(f"\nAll results saved to: {out_dir}/")
    print("Analysis complete!")


def main():
    args = _build_parser().parse_args()
    npz_path = Path(args.npz_path)
    if not npz_path.exists():
        print(f"Error: NPZ file not found: {npz_path}")
        sys.exit(1)

    summary = pipeline.run(
        npz_path=str(npz_path),
        layer=args.layer,
        sae_repo=args.sae_repo,
        out_dir=args.out_dir,
        density_threshold=args.density_threshold,
        top_k_pairs=args.top_k_pairs,
        block_size=args.block_size,
        antipodal_only=not args.no_antipodal_only,
        build_within_cross=not args.no_within_cross,
        make_umap=not args.no_umap,
        umap_neighbors=args.umap_neighbors,
        rng_seed=args.seed
    )
    _print_summary(summary, args.out_dir)


if __name__ == "__main__":
    main()
