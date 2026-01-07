"""
util functions for antipodality analysis
"""

import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Tuple, Any

from .constants import MATRYOSHKA_LEVELS, LEVEL_RANGES, LEVEL_COLORS


def prepare_for_json(obj):
    """convert objects to json"""
    if isinstance(obj, dict):
        return {k: prepare_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [prepare_for_json(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    # numpy arrays
    if isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.floating) or np.issubdtype(obj.dtype, np.integer) or np.issubdtype(obj.dtype, np.bool_):
            # replace non-finite with None to be JSON-safe
            if np.issubdtype(obj.dtype, np.floating):
                o = obj.astype(object)
                o[~np.isfinite(obj)] = None
                return o.tolist()
            return obj.tolist()
        # fallback for object arrays
        return [prepare_for_json(x) for x in obj.tolist()]
    # numpy scalars
    if isinstance(obj, np.generic):
        v = obj.item()
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        return v
    # plain floats with non-finite values
    if isinstance(obj, float):
        return None if (obj != obj or obj == float('inf') or obj == float('-inf')) else obj
    return obj


class MatryoshkaUtils:
    """util class for matryoshka hierarchy analysis"""

    @staticmethod
    def get_level_info(feature_idx: int) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[str]]:
        """map feature index to its exclusive Matryoshka level"""
        for i, (start, end) in enumerate(LEVEL_RANGES):
            if start <= feature_idx < end:
                level_size = end - start  # Actual size of this level
                return i, level_size, MATRYOSHKA_LEVELS[i], LEVEL_COLORS[i]
        return None, None, None, None

    @staticmethod
    def get_level(feature_idx: int) -> Optional[int]:
        """get the Matryoshka level for a given feature index (exclusive binning)"""
        for level, (lo, hi) in zip(MATRYOSHKA_LEVELS, LEVEL_RANGES):
            if lo <= feature_idx < hi:
                return level
        return None

    @staticmethod
    def coerce_level(level: Optional[int]) -> int:
        """return given level or the last level if none"""
        return level if level is not None else MATRYOSHKA_LEVELS[-1]


def assign_levels(indices: np.ndarray) -> np.ndarray:
    """map feature indices to level ids"""
    out = np.empty(len(indices), dtype=int)
    for k, idx in enumerate(indices):
        level_idx, *_ = MatryoshkaUtils.get_level_info(int(idx))
        out[k] = level_idx if level_idx is not None else len(MATRYOSHKA_LEVELS) - 1
    return out