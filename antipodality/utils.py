"""JSON conversion and Matryoshka level lookup."""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from .constants import MATRYOSHKA_LEVELS, LEVEL_RANGES, LEVEL_COLORS


def prepare_for_json(obj):
    """Convert NumPy values and paths, replacing nonfinite numbers with nulls."""
    if isinstance(obj, dict):
        return {key: prepare_for_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [prepare_for_json(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.floating) or np.issubdtype(obj.dtype, np.integer) or np.issubdtype(obj.dtype, np.bool_):
            if np.issubdtype(obj.dtype, np.floating):
                values = obj.astype(object)
                values[~np.isfinite(obj)] = None
                return values.tolist()
            return obj.tolist()
        return [prepare_for_json(value) for value in obj.tolist()]
    if isinstance(obj, np.generic):
        value = obj.item()
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return None
        return value
    if isinstance(obj, float):
        return None if (obj != obj or obj == float('inf') or obj == float('-inf')) else obj
    return obj


class MatryoshkaUtils:
    """Assign features to exclusive index ranges, not cumulative prefixes."""

    @staticmethod
    def get_level_info(feature_idx: int) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[str]]:
        for i, (start, end) in enumerate(LEVEL_RANGES):
            if start <= feature_idx < end:
                level_size = end - start
                return i, level_size, MATRYOSHKA_LEVELS[i], LEVEL_COLORS[i]
        return None, None, None, None

    @staticmethod
    def get_level(feature_idx: int) -> Optional[int]:
        for level, (lo, hi) in zip(MATRYOSHKA_LEVELS, LEVEL_RANGES):
            if lo <= feature_idx < hi:
                return level
        return None


def assign_levels(indices: np.ndarray) -> np.ndarray:
    """Return level positions, using the last level for unmatched indices."""
    out = np.empty(len(indices), dtype=int)
    for k, idx in enumerate(indices):
        level_idx, *_ = MatryoshkaUtils.get_level_info(int(idx))
        out[k] = level_idx if level_idx is not None else len(MATRYOSHKA_LEVELS) - 1
    return out
