#!/usr/bin/env python3
"""
Simple Threshold Justification Script

Generates histograms to justify the 0.05 density threshold used in
unified_antipodality_analyzer.py for identifying dense features.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Set matplotlib for headless operation
plt.switch_backend('Agg')

def main():
    layers = [2, 12, 22]
    results = {}

    for layer in layers:
        npz_file = f'dense_activations_layer_{layer}.npz'

        try:
            print(f"Processing Layer {layer}...")

            # Load density data
            data = np.load(npz_file)
            densities = data['densities']
            total_features = len(densities)

            print(f"  Total features: {total_features:,}")
            print(f"  Density range: {densities.min():.6f} - {densities.max():.6f}")
            print(f"  Mean density: {densities.mean():.6f}")
            print(f"  Median density: {np.median(densities):.6f}")
            print(f"  95th percentile: {np.percentile(densities, 95):.6f}")
            print(f"  99th percentile: {np.percentile(densities, 99):.6f}")

            # Calculate threshold statistics
            count_05 = (densities > 0.05).sum()
            count_10 = (densities > 0.10).sum()
            percent_05 = 100 * count_05 / total_features
            percent_10 = 100 * count_10 / total_features

            print(f"  Threshold = 0.05: {count_05:4d} features ({percent_05:5.2f}%)")
            print(f"  Threshold = 0.10: {count_10:4d} features ({percent_10:5.2f}%)")

        
            # Store results
            results[layer] = {
                'total': total_features,
                'count_05': count_05,
                'count_10': count_10,
                'percent_05': percent_05,
                'percent_10': percent_10,
                'mean_density': densities.mean(),
                'max_density': densities.max()
            }

            # Create histogram
            plt.figure(figsize=(12, 8))

            # Main histogram
            counts, bins, patches = plt.hist(densities, bins=100, alpha=0.7,
                                           color='skyblue', edgecolor='black')

            # Add threshold lines
            plt.axvline(0.05, color='red', linestyle='--', linewidth=3,
                       label=f'tau = 0.05 ({count_05} features)')
            plt.axvline(0.10, color='orange', linestyle='--', linewidth=3,
                       label=f'tau = 0.10 ({count_10} features)')

            # Formatting
            plt.title(f'Layer {layer}: Density Distribution & Threshold Justification',
                     fontsize=16, fontweight='bold')
            plt.xlabel('Activation Density', fontsize=14)
            plt.ylabel('Number of Features (log scale)', fontsize=14)
            plt.yscale('log')  # Use log scale for better visibility
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)

            # Save figure
            output_file = f'threshold_layer_{layer}.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  Saved: {output_file}")
            print()

        except FileNotFoundError:
            print()
        except Exception as e:
            print()

    # Summary analysis
    if results:
        total_features_all = sum(r['total'] for r in results.values())
        total_dense_05 = sum(r['count_05'] for r in results.values())
        avg_percent_05 = np.mean([r['percent_05'] for r in results.values()])

        print(f"- {total_dense_05:,} features across all layers")
        print(f"- {avg_percent_05:.2f}% of features per layer")

        print("Cross-layer consistency:")
        for layer in sorted(results.keys()):
            r = results[layer]
            print(f"Layer {layer:2d}: {r['percent_05']:5.2f}% dense features "
                  f"(mean density: {r['mean_density']:.4f})")

if __name__ == "__main__":
    main()