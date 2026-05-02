"""
Similarity computation utilities for antipodality analysis
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Union


def pair_score_matrix(C_enc: torch.Tensor, C_dec: torch.Tensor, antipodal_only: bool) -> torch.Tensor:
    """
    Compute per-pair scores from cosine matrices.
    If antipodal_only=True, only (neg,neg) pairs get a score; others are -inf.
    Returns a score matrix S with the diagonal = -inf.
    """
    with torch.no_grad():
        # C_*: (n, m) tensors
        if antipodal_only:
            mask = (C_enc < 0) & (C_dec < 0)
            S = torch.where(mask, (-C_enc) * (-C_dec), torch.full_like(C_enc, float("-inf")))
        else:
            S = C_enc * C_dec
        return S


def normalize_weights(
    W_enc: np.ndarray,
    W_dec: np.ndarray,
    indices: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract and normalize encoder/decoder weights for selected features
    """
    # Input validation
    indices = np.asarray(indices)
    # Normalize weights for cosine similarity computation
    E = F.normalize(torch.from_numpy(W_enc[indices]).float(), dim=1)   # (n, d)
    D = F.normalize(torch.from_numpy(W_dec[indices]).float(), dim=1)   # (n, d)

    return E, D


def blocked_pair_scores(
    E: torch.Tensor,
    D: torch.Tensor,
    top_k: int,
    block_size: int,
    antipodal_only: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute top-k antipodality scores using blocked matrix computation.

    """
    with torch.no_grad():
        n = E.shape[0]

        # Initialize top-k 
        top_vals = torch.full((n, top_k), float("-inf"))
        top_idx  = torch.full((n, top_k), -1, dtype=torch.long)
        rows = torch.arange(n).view(-1, 1)

        n_blocks = (n + block_size - 1) // block_size

        # Blocked computation to avoid memory bottlenecks with progress tracking
        print(f"Processing {n_blocks} blocks of size {block_size}...")
        for block_idx, start in enumerate(range(0, n, block_size)):
            end = min(start + block_size, n)
            C_enc = E @ E[start:end].T   # (n, b)
            C_dec = D @ D[start:end].T   # (n, b)

            S = pair_score_matrix(C_enc, C_dec, antipodal_only)
            # remove self-pairs where row and col refer to the same feature
            cols = torch.arange(start, end).view(1, -1)
            S[rows == cols] = float("-inf")

            k_here = min(top_k, S.shape[1])
            cand_vals, cand_pos_local = torch.topk(S, k=k_here, dim=1)
            cand_idx = cand_pos_local + start

            top_vals, _pos = torch.topk(torch.cat([top_vals, cand_vals], dim=1), k=top_k, dim=1)
            top_idx = torch.gather(torch.cat([top_idx, cand_idx], dim=1), 1, _pos)

        scores   = top_vals[:, 0].cpu().numpy()
        partners = top_idx[:,  0].cpu().numpy()

        return scores, partners


def cosine_matrix(X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Compute cosine similarity matrix with diagonal masked a

    """
    # Handle both tensor and numpy array inputs
    if torch.is_tensor(X):
        X_tensor = X.detach().cpu().float()
    else:
        X_tensor = torch.from_numpy(X).float()
    X_tensor = F.normalize(X_tensor, dim=1)
    C = (X_tensor @ X_tensor.T).numpy()
    np.fill_diagonal(C, np.nan)       
    return C
