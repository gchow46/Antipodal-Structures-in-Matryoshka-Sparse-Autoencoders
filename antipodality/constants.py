"""
Constants for Matryoshka SAE hierarchy analysis.

This module contains the core constants used throughout the antipodality analysis,
including the Matryoshka level definitions, ranges, and visualization colors.
"""

# Matryoshka hierarchy constants
MATRYOSHKA_LEVELS = [128, 512, 2048, 8192, 32768]
LEVEL_RANGES = [(0, 128), (128, 512), (512, 2048), (2048, 8192), (8192, 32768)]
LEVEL_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Analysis threshold
DENSE_THRESHOLD = 0.05