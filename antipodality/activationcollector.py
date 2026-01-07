#!/usr/bin/env python3


import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer
from sae_lens import SAE, HookedSAETransformer
from datasets import load_dataset, DownloadMode
import random


def load_texts(max_entries=150000, min_length=50):
    dataset = load_dataset(
        "Skylion007/openwebtext", 
        split="train", 
        streaming=True, 
        trust_remote_code=True, 
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS
    )   
    texts=[]
    for i, example in enumerate(dataset):
        if i >= max_entries:
            break
        text = example.get("text", "")
        if len(text.strip()) > min_length:
            texts.append(text)
        if i % 5000 == 0 and i > 0:
            print(f" Total texts loaded: {len(texts)} texts")
    print(f" Total texts loaded:{len(texts)}")
    return texts
        
def setup_environment():
    """Set up PyTorch optimizations and random seeds."""
    # Seeds for reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Enable optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def main():
    """Main pipeline execution."""
    # Get configuration from environment variables
    layer = int(os.environ["LAYER"])
    token_budget = int(os.environ["TOKEN_BUDGET"])
    max_length = int(os.environ["MAX_LENGTH"])
    batch_size = int(os.environ["BATCH_SIZE"])
    
    print(f"Processing Layer {layer}")
    print(f"Config: tokens={token_budget}, max_len={max_length}, batch={batch_size}")
    
    # Setup environment
    setup_environment()
    torch.set_grad_enabled(False)

    sae_id = f"blocks.{layer}.hook_resid_post"
    
    # Load SAE on CPU and keep it there (avoids GPU OOM on large Matryoshka SAEs)
    print(f"SAE configuration will be loaded for layer {layer}")
    sae = SAE.from_pretrained(
        "gemma-2-2b-res-matryoshka-dc",
        sae_id,
        device="cpu"
    )
    sae.eval()
    
    # Load model with SAE's model configuration for compatibility
    
    # Default kwargs
    model_kwargs = {}
    
    # Improved CPU loading strategy: load in fp32 on CPU, cast on GPU
    
    model = HookedSAETransformer.from_pretrained_no_processing(
        "google/gemma-2-2b",
        device="cpu",
        torch_dtype=torch.float32,  # Always load in fp32 on CPU
        **model_kwargs
    )
    
    # Move to GPU and cast to appropriate precision
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_properties(0).major
        target_dtype = torch.bfloat16 if cap >= 8 else torch.float16
        model = model.to("cuda")
        model = model.to(target_dtype)
        
        # Enable optimizations after GPU placement
        if hasattr(model.cfg, 'attn_implementation'):
            model.cfg.attn_implementation = "sdpa"
        if hasattr(model.cfg, 'n_ctx'):
            model.cfg.n_ctx = max_length  # Clamp context to actual sequence length
    else:
        print("  Keeping model on CPU in fp32")
    
    model = model.eval()

    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear any fragmentation
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    tokenizer.pad_token = tokenizer.eos_token
    # Load texts
    texts = load_texts()
    print(f"Loaded {len(texts)} texts")
    
    # Print SAE dimensions
    d_sae = int(getattr(sae, "d_sae", sae.W_dec.shape[0]))
    print(f"SAE d_sae: {d_sae}")
    
 

    # Output json, npz
    acts_json = os.environ.get("ACTS_JSON", f"activations_layer{layer}.json")
    acts_npz  = os.environ.get("ACTS_NPZ",  f"activations_layer{layer}.npz")


    # We'll grab resid_post at this layer, encode with the SAE, and tally density.
    def _capture(store):
        def _hook(tensor, hook):
            # Keep residuals on same device as model for efficiency
            store["resid"] = tensor.detach()
            return tensor
        return _hook

    # Stop at 1.5M
    def _iter_batches(texts, bsz, max_len):
        for i in range(0, len(texts), bsz):
            yield texts[i:i+bsz], tokenizer(
                texts[i:i+bsz],
                truncation=True, padding=True,
                max_length=max_len, return_tensors="pt"
            )

    pos_counts = np.zeros(d_sae, dtype=np.int64)
    sum_vals   = np.zeros(d_sae, dtype=np.float64)
    total_valid_tokens = 0



    # Warmup (helps SDPA kernels)
    with torch.inference_mode():
        device = next(model.parameters()).device  # Get device from model parameters
        w = tokenizer(["warmup"], max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        w = {k: v.to(device, non_blocking=True) for k, v in w.items()}  # Dict comprehension for guaranteed non_blocking
        _ = model(w["input_ids"], attention_mask=w["attention_mask"])

    seen_tokens = 0
    stop_at = min(getattr(model.cfg, "n_layers", layer+1), layer+1)
    
    # Cache device/dtype outside loop for efficiency
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    
    # Initialize progress counter outside loop
    next_report = 50000


    for _, toks in _iter_batches(texts, batch_size, max_length):
        if seen_tokens >= token_budget:
            break

        # Use dict comprehension for guaranteed non_blocking support
        toks = {k: v.to(model_device, non_blocking=True) for k, v in toks.items()}
        # Keep attention mask on same device as model for GPU optimization
        attn_bool = toks["attention_mask"].bool()

        # Clear any residual tensors before processing
        store = {}
        hooks = [(f"blocks.{layer}.hook_resid_post", _capture(store))]
        with torch.inference_mode():
         
            
            if model_device.type == "cuda" and model_dtype in (torch.float16, torch.bfloat16):
                with torch.autocast(device_type="cuda", dtype=model_dtype):
                    model.run_with_hooks(
                        toks["input_ids"], attention_mask=toks["attention_mask"],
                        fwd_hooks=hooks, stop_at_layer=stop_at
                    )
            else:
                model.run_with_hooks(
                    toks["input_ids"], attention_mask=toks["attention_mask"],
                    fwd_hooks=hooks, stop_at_layer=stop_at
                )
    

        resid = store["resid"]                            # [B,T,d_model] on same device as model
        B, T, Dm = resid.shape
        valid = attn_bool.view(-1).to(resid.device)       # [B*T] move to same device as resid
        flat = resid.view(B*T, Dm)[valid].contiguous()    # [n_valid, d_model]

        # SAE encode on CPU (hybrid approach: model on GPU, SAE on CPU)

        flat_cpu = flat.to('cpu', non_blocking=True) if flat.is_cuda else flat
        acts = sae.encode(flat_cpu)                   # [n_valid, d_sae] SAE encoding on CPU
    
        # Density stats
        pos = acts > 0
        pos_counts += pos.sum(dim=0).cpu().numpy()
        sum_vals   += acts.sum(dim=0).cpu().numpy()

        n_valid = int(valid.sum().item())
        total_valid_tokens += n_valid
        seen_tokens        += n_valid
            
        
        del resid, flat, acts, pos
        
        
        # Progress reporting using cached device
        if seen_tokens >= next_report:
            if model_device.type == "cuda":
                print(f"  Processed {seen_tokens:,} tokens. GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")
                # Empty cache every 8 reports (400k tokens)
                if (next_report // 50000) % 8 == 0:
                    torch.cuda.empty_cache()
            else:
                print(f"  Processed {seen_tokens:,} tokens (CPU)")
            next_report += 50000

    # Densities & mean activation when active
    densities = (pos_counts / max(1, total_valid_tokens)).astype(np.float64)
    means_when_active = np.divide(sum_vals, np.maximum(1, pos_counts), dtype=np.float64)

    summary = {
        "layer": layer,
        "d_sae": int(d_sae),
        "tokens_processed": int(total_valid_tokens),
        "mean_density_all": float(densities.mean()),
        "max_density_all": float(densities.max()),
    }


    with open(acts_json, "w") as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(
        acts_npz,
        densities=densities,
        pos_counts=pos_counts,
        sum_vals=sum_vals,
        means_when_active=means_when_active
    )

    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {acts_json}\nWrote: {acts_npz}")


if __name__ == "__main__":
    main()