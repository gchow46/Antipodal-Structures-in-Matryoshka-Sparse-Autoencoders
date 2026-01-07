"""
Input/Output utilities for SAE data loading.

This module is the only place that imports sae_lens, keeping the rest of the
package import-light by isolating heavy dependencies.
"""

import numpy as np
import torch
from sae_lens import SAE
from typing import Tuple, Dict


def load_density_data(npz_path: str) -> np.ndarray:
    """
    Load activation density data from NPZ 
    """
  
    data = np.load(npz_path)


    densities = data["densities"].astype(np.float32)


    # Print summary
    print(f"Densities loaded for {len(densities)} latents")
    print(f"Density statistics: mean={densities.mean():.4f}, max={densities.max():.4f}")

    return densities


def load_sae_weights(sae_repo: str, layer: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load SAE encoder/decoder weights with shape validation.


    """
    sae_id = f"blocks.{layer}.hook_resid_post"
    print(f"Loading SAE for layer {layer}: {sae_id}")

    # Load SAE on CPU only
    sae = SAE.from_pretrained(sae_repo, sae_id, device="cpu")
    sae.eval()

    # Extract weights and convert to CPU float tensors
    W_enc_tensor = sae.W_enc.detach().cpu().float()
    W_dec_tensor = sae.W_dec.detach().cpu().float()

    # Expected final shape
    expected_shape = (sae.cfg.d_sae, sae.cfg.d_in)

    # Transpose 
    W_enc = W_enc_tensor.T.numpy().astype(np.float32)
    W_dec = W_dec_tensor.numpy().astype(np.float32)
    

    # Create config dict
    cfg = {
        "repo": sae_repo,
        "layer": layer,
        "sae_id": sae_id,
        "d_sae": int(sae.cfg.d_sae),
        "d_in": int(sae.cfg.d_in)
    }

    print(f"Extracted SAE weights: W_enc {W_enc.shape}, W_dec {W_dec.shape}")
    print(f"SAE config: d_sae={cfg['d_sae']}, d_in={cfg['d_in']}")

    # Clean up SAE object to free memory
    del sae

    return W_enc, W_dec, cfg