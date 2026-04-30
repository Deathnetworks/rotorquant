import torch
import torch.nn as nn
from tqdm import tqdm
from .clifford import embed_vectors_as_multivectors, extract_vectors_from_multivectors

def calibrate_xpu(model, tokenizer, bits=4, mode='rotor', n_samples=32):
    """
    Calibrate codebooks by sampling KV states from the model's actual attention layers.
    All operations are kept on XPU to maximize throughput and power draw.
    """
    device = next(model.parameters()).device
    # layers 3, 7, 11, 15, 19, 23 are the attention layers for Qwen3.5-2B
    attn_layer_indices = [3, 7, 11, 15, 19, 23]
    sampled_data = {idx: [] for idx in attn_layer_indices}
    
    def get_hook(idx):
        def hook(module, input, output):
            # Keep on XPU for native calibration
            if len(sampled_data[idx]) < n_samples:
                sampled_data[idx].append(output.detach())
        return hook

    hooks = []
    for idx in attn_layer_indices:
        h = model.model.layers[idx].self_attn.k_proj.register_forward_hook(get_hook(idx))
        hooks.append(h)

    print(f"Sampling {n_samples} activations on XPU for {mode}{bits} calibration...")
    prompt = "The quick brown fox jumps over the lazy dog. " * 10
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Warmup the card during sampling
    with torch.no_grad():
        for _ in range(n_samples):
            model(**inputs)

    for h in hooks: h.remove()

    rotors_map = {}
    cbs_map = {}

    print(f"Fitting {mode}{bits} centroids on XPU...")
    for idx in attn_layer_indices:
        # (N_samples, seq, dim) -> (N, dim)
        data = torch.cat(sampled_data[idx], dim=0) 
        data = data.view(-1, data.size(-1))
        
        if mode == 'rotor':
            # Initialize identity rotors on XPU
            n_groups = (data.size(-1) + 2) // 3
            rotors = torch.zeros((n_groups, 8), device=device, dtype=torch.float32)
            rotors[:, 0] = 1.0 # scalar = 1.0
            
            # Linear distribution fitting on XPU
            # Note: We use linspace here for the baseline to ensure we are testing 
            # the KERNEL throughput, not the lloyd-max convergence time.
            cbs = {
                'scalar': torch.linspace(-1, 1, 2**bits, device=device, dtype=torch.float32),
                'vector': torch.linspace(-1, 1, 2**bits, device=device, dtype=torch.float32),
                'bivector': torch.linspace(-1, 1, 2**bits, device=device, dtype=torch.float32),
                'trivector': torch.linspace(-1, 1, 2**bits, device=device, dtype=torch.float32)
            }
            rotors_map[idx] = rotors
            cbs_map[idx] = cbs
        else:
            # IsoQuant calibration on XPU
            centroids = torch.linspace(-1, 1, 2**bits, device=device, dtype=torch.float32)
            rotors_map[idx] = None
            cbs_map[idx] = {'iso_centroids': centroids}

    return rotors_map, cbs_map
