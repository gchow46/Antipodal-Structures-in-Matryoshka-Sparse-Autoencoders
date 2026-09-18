"""Load activation densities and pretrained SAE weights."""

import numpy as np
from sae_lens import SAE
from typing import Tuple, Dict


def load_density_data(npz_path: str) -> np.ndarray:
    """Read the density array as float32."""
    data = np.load(npz_path)
    densities = data["densities"].astype(np.float32)
    print(f"Densities loaded for {len(densities)} latents")
    print(f"Density statistics: mean={densities.mean():.4f}, max={densities.max():.4f}")
    return densities


def load_sae_weights(sae_repo: str, layer: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load CPU weights with one feature per row, plus the SAE dimensions and identity."""
    sae_id = f"blocks.{layer}.hook_resid_post"
    print(f"Loading SAE for layer {layer}: {sae_id}")
    sae = SAE.from_pretrained(sae_repo, sae_id, device="cpu")
    sae.eval()

    W_enc_tensor = sae.W_enc.detach().cpu().float()
    W_dec_tensor = sae.W_dec.detach().cpu().float()
    W_enc = W_enc_tensor.T.numpy().astype(np.float32)
    W_dec = W_dec_tensor.numpy().astype(np.float32)
    cfg = {
        "repo": sae_repo,
        "layer": layer,
        "sae_id": sae_id,
        "d_sae": int(sae.cfg.d_sae),
        "d_in": int(sae.cfg.d_in)
    }
    print(f"Extracted SAE weights: W_enc {W_enc.shape}, W_dec {W_dec.shape}")
    print(f"SAE config: d_sae={cfg['d_sae']}, d_in={cfg['d_in']}")
    del sae
    return W_enc, W_dec, cfg
