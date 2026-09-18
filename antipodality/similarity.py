"""Cosine similarities and blocked pair scoring."""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Union


def pair_score_matrix(C_enc: torch.Tensor, C_dec: torch.Tensor, antipodal_only: bool) -> torch.Tensor:
    """Multiply encoder/decoder cosines, optionally excluding non-antipodal pairs."""
    with torch.no_grad():
        if antipodal_only:
            mask = (C_enc < 0) & (C_dec < 0)
            scores = torch.where(mask, (-C_enc) * (-C_dec), torch.full_like(C_enc, float("-inf")))
        else:
            scores = C_enc * C_dec
        return scores


def normalize_weights(
    W_enc: np.ndarray,
    W_dec: np.ndarray,
    indices: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select features and normalize their encoder/decoder rows."""
    indices = np.asarray(indices)
    E = F.normalize(torch.from_numpy(W_enc[indices]).float(), dim=1)
    D = F.normalize(torch.from_numpy(W_dec[indices]).float(), dim=1)
    return E, D


def blocked_pair_scores(
    E: torch.Tensor,
    D: torch.Tensor,
    top_k: int,
    block_size: int,
    antipodal_only: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Track top-k candidates in blocks and return each feature's best score and partner."""
    with torch.no_grad():
        n = E.shape[0]
        top_values = torch.full((n, top_k), float("-inf"))
        top_indices = torch.full((n, top_k), -1, dtype=torch.long)
        rows = torch.arange(n).view(-1, 1)
        n_blocks = (n + block_size - 1) // block_size

        print(f"Processing {n_blocks} blocks of size {block_size}...")
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            C_enc = E @ E[start:end].T
            C_dec = D @ D[start:end].T
            scores = pair_score_matrix(C_enc, C_dec, antipodal_only)

            # Block columns use local positions; mask self-pairs in global coordinates.
            cols = torch.arange(start, end).view(1, -1)
            scores[rows == cols] = float("-inf")

            k_here = min(top_k, scores.shape[1])
            candidate_values, candidate_positions = torch.topk(scores, k=k_here, dim=1)
            candidate_indices = candidate_positions + start
            top_values, selected = torch.topk(
                torch.cat([top_values, candidate_values], dim=1), k=top_k, dim=1
            )
            top_indices = torch.gather(torch.cat([top_indices, candidate_indices], dim=1), 1, selected)

        scores = top_values[:, 0].cpu().numpy()
        partners = top_indices[:, 0].cpu().numpy()
        return scores, partners


def cosine_matrix(X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Compute row-wise cosine similarities with a NaN diagonal."""
    if torch.is_tensor(X):
        X_tensor = X.detach().cpu().float()
    else:
        X_tensor = torch.from_numpy(X).float()
    X_tensor = F.normalize(X_tensor, dim=1)
    similarities = (X_tensor @ X_tensor.T).numpy()
    np.fill_diagonal(similarities, np.nan)
    return similarities
