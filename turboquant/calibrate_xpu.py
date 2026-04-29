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
def calibrate_xpu(model, tokenizer, n_samples=4, bits=4):
    print(f"Calibrating XPU Codebooks (Full 8-component, {bits}-bit)...")
    device = next(model.parameters()).device
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
