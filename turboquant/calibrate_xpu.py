import torch
import numpy as np
from transformers import DynamicCache
from . import xpu_backend

def _fit_centroids_1d(samples: np.ndarray, n_centroids: int) -> np.ndarray:
    """Lloyd-Max centroids for 1D."""
    if len(samples) < n_centroids:
        return np.linspace(-0.1, 0.1, n_centroids).astype(np.float32)
    
    # Initialize from quantiles
    q = np.linspace(0, 1, n_centroids + 2)[1:-1]
    centroids = np.quantile(samples, q)
    
    for _ in range(20):
        midpoints = (centroids[:-1] + centroids[1:]) / 2.0
        indices = np.searchsorted(midpoints, samples)
        new_centroids = np.zeros_like(centroids)
        for i in range(n_centroids):
            mask = (indices == i)
            if mask.any():
                new_centroids[i] = samples[mask].mean()
            else:
                new_centroids[i] = centroids[i]
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids
    return centroids.astype(np.float32)

def _fit_centroids_3d(samples: np.ndarray, n_centroids: int) -> np.ndarray:
    """K-Means for 3D centroids."""
    from sklearn.cluster import MiniBatchKMeans
    if len(samples) < n_centroids:
        return np.random.randn(n_centroids, 3).astype(np.float32)
    
    # Use MiniBatchKMeans for speed
    kmeans = MiniBatchKMeans(n_clusters=n_centroids, n_init=3, max_iter=50, batch_size=4096).fit(samples)
    return kmeans.cluster_centers_.astype(np.float32)

@torch.no_grad()
def calibrate_xpu(model, tokenizer, n_samples=8, bits=4, mode='iso'):
    print(f"Calibrating XPU Codebooks (Mode: {mode}, {bits}-bit)...")
    device = next(model.parameters()).device
    
    # 1. Capture KV-cache distributions
    activations = []
    _orig_update = DynamicCache.update
    def hook(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx == 0: # Sample from first layer for simplicity
            activations.append(key_states.detach().cpu())
        return _orig_update(self, key_states, value_states, layer_idx, cache_kwargs)
    
    DynamicCache.update = hook
    for _ in range(n_samples):
        input_ids = torch.randint(0, model.config.vocab_size, (1, 128), device=device)
        model(input_ids, use_cache=True)
    DynamicCache.update = _orig_update

    # 2. Fit centroids
    X = torch.cat(activations, dim=0).to(torch.float32) # (N, H, L, D)
    D = X.size(-1)
    
    if mode == 'iso':
        from sklearn.cluster import MiniBatchKMeans
        # 4D blocks
        X_blocks = X.view(-1, 4)
        # Normalize blocks to match kernel behavior
        norms = torch.norm(X_blocks, dim=-1, keepdim=True).clamp(min=1e-8)
        X_unit = X_blocks / norms
        
        print(f"Fitting 4D centroids to {X_unit.size(0)} unit-normalized samples...")
        kmeans = MiniBatchKMeans(n_clusters=2**bits, n_init=3, batch_size=1024).fit(X_unit.numpy())
        centroids = torch.from_numpy(kmeans.cluster_centers_).to(torch.float32).to(device)
        
        # Binary search requires sorted centroids (for each component)
        # For 4D, we can sort by the first component or just sort all elements if it's a 1D scalar fallback.
        # Wait, IsoQuant uses 4D blocks but SCALAR quantization per component.
        # So we need a 1D sorted codebook for all components.
        
        # Flatten and sort to get a 1D scalar codebook from the 4D clusters
        centroids_1d, _ = torch.sort(centroids.flatten())
        # We need exactly 2**bits levels.
        # If we had (2**bits, 4), we have 4 * 2**bits values.
        # Let's just take the unique-ish 2**bits values or just fit a 1D kmeans.
        
        print(f"Fitting 1D scalar codebook for components...")
        kmeans_1d = MiniBatchKMeans(n_clusters=2**bits, n_init=3).fit(X_unit.flatten().numpy().reshape(-1, 1))
        cb_1d, _ = torch.sort(torch.from_numpy(kmeans_1d.cluster_centers_).flatten())
        return {'iso_centroids': cb_1d.to(device)}
    
    n_layers = model.config.num_hidden_layers
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    n_groups = (head_dim + 2) // 3
    n_centroids = 2**bits
    
    # Stats for all components
    stats = {i: {'scalar': [], 'vector': [], 'bivector': [], 'trivector': []} for i in range(n_layers)}
    rotors_dict = {}
    for i in range(n_layers):
        r = torch.randn(n_groups, 4, device=device, dtype=torch.float32)
        rotors_dict[i] = r / torch.norm(r, dim=-1, keepdim=True)

    _orig_update = DynamicCache.update
    
    def hook(self, key_states, value_states, layer_idx, cache_kwargs=None):
        r = rotors_dict[layer_idx]
        mv = xpu_backend.rotor_sandwich(key_states, r)
        
        # 1. Vector (1,2,3)
        vec_parts = mv[..., 1:4].reshape(-1, 3)
        v_norms = vec_parts.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        stats[layer_idx]['vector'].append((vec_parts / v_norms).cpu().float().numpy())
        
        # 2. Scalar (0)
        s_parts = mv[..., 0].reshape(-1)
        stats[layer_idx]['scalar'].append(s_parts.cpu().float().numpy())
        
        # 3. Bivector (4,5,6)
        b_parts = mv[..., 4:7].reshape(-1)
        stats[layer_idx]['bivector'].append(b_parts.cpu().float().numpy())
        
        # 4. Trivector (7)
        t_parts = mv[..., 7].reshape(-1)
        stats[layer_idx]['trivector'].append(t_parts.cpu().float().numpy())
        
        return _orig_update(self, key_states, value_states, layer_idx, cache_kwargs)

    DynamicCache.update = hook
    for _ in range(n_samples):
        # 128 tokens per sample
        input_ids = torch.randint(0, model.config.vocab_size, (1, 128), device=device)
        model(input_ids, use_cache=True)
    DynamicCache.update = _orig_update

    codebooks = {}
    for i in range(n_layers):
        c_v = _fit_centroids_3d(np.concatenate(stats[i]['vector']), n_centroids)
        c_s = _fit_centroids_1d(np.concatenate(stats[i]['scalar']), n_centroids)
        c_b = _fit_centroids_1d(np.concatenate(stats[i]['bivector']), n_centroids)
        c_t = _fit_centroids_1d(np.concatenate(stats[i]['trivector']), n_centroids)
        
        codebooks[i] = {
            'scalar': torch.from_numpy(c_s).to(device),
            'vector': torch.from_numpy(c_v.flatten()).to(device),
            'bivector': torch.from_numpy(c_b).to(device),
            'trivector': torch.from_numpy(c_t).to(device),
        }
        
    print("Calibration complete.")
    return rotors_dict, codebooks
