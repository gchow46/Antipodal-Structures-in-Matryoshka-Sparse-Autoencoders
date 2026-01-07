"""
Type definitions for antipodality analysis.

This module contains dataclass definitions for payload objects used in the
functional core, imperative shell pattern for visualization generation.
"""

from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Any


@dataclass
class EncDecScatterPayload:
    """Data payload for encoder-decoder scatter plot visualization"""
    enc_sim: np.ndarray      # encoder similarities
    dec_sim: np.ndarray      # decoder similarities
    pair_score: np.ndarray   # antipodality scores (color metric)
    counts: Dict[str, int]   # quadrant counts
    layer: int               # layer number
    meta: Dict[str, Any]     # metadata (ranges, sizes, etc.)


@dataclass
class WithinCrossPayload:
    """Data payload for within-level vs cross-level comparison visualization"""
    top_within: List[Dict]   # within-level pairs with scores and labels
    top_cross: List[Dict]    # cross-level pairs with scores and labels
    layer: int               # layer number
    stats: Dict[str, Any]    # summary statistics (means, counts, etc.)
