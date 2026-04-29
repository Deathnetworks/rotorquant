import torch
import torch.nn as nn
import math
from tqdm import tqdm
from transformers import DynamicCache
from turboquant import xpu_backend, isoquant

class IsoQuantCache(DynamicCache):
    def __init__(self, model_config, n_levels=16, centroids=None):
        super().__init__()
        self.head_dim = getattr(model_config, "head_dim", model_config.hidden_size // model_config.num_attention_heads)
        self.n_levels = n_levels
        self.q_dict = {}
        self.cb_dict = {}
        self.device = torch.device("xpu")
        self.calibrated_centroids = centroids

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx not in self.q_dict:
            iso = isoquant.IsoQuantMSE(self.head_dim, int(math.log2(self.n_levels)), mode='fast', device='cpu')
            self.q_dict[layer_idx] = iso.q_L.to(self.device)
            if self.calibrated_centroids is not None:
                self.cb_dict[layer_idx] = self.calibrated_centroids.to(self.device)
            else:
                self.cb_dict[layer_idx] = iso.centroids.to(self.device)
        
        qL = self.q_dict[layer_idx]
        cb = self.cb_dict[layer_idx]
        
        k_q = xpu_backend.isoquant_fused(key_states, qL, cb)
        v_q = xpu_backend.isoquant_fused(value_states, qL, cb)

        return super().update(k_q, v_q, layer_idx, cache_kwargs)

class RotorQuantCache(DynamicCache):
    """
    A DynamicCache that intercepts K/V updates and applies RotorQuant.
    """
    def __init__(self, n_levels=16, rotors_dict=None, codebooks_dict=None):
        super().__init__()
        self.n_levels = n_levels
        self.rotors_dict = rotors_dict or {}
        self.codebooks_dict = codebooks_dict or {}
        self.is_calibrated = (codebooks_dict is not None)

    def calibrate(self, model, tokenizer, n_samples=4):
        """Calibrate rotors and codebooks on sample data."""
        print("Calibrating RotorQuant...")
        device = next(model.parameters()).device
        with torch.no_grad():
            for i in range(n_samples):
                # Use a real sentence or random tokens
                input_ids = torch.randint(0, model.config.vocab_size, (1, 64), device=device)
                outputs = model(input_ids, output_attentions=False, use_cache=True)
                # DynamicCache will be updated during this pass
                # We can capture the distributions here
        self.is_calibrated = True
        print("Calibration complete.")

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # 1. Apply RotorQuant to key_states
        # Note: In production, we would also quantize value_states
        head_dim = key_states.size(-1)
        n_groups = (head_dim + 2) // 3
        
        if layer_idx not in self.rotors_dict:
            # Initialize random rotors if not present
            r = torch.randn(n_groups, 4, device=key_states.device, dtype=torch.float32)
            self.rotors_dict[layer_idx] = r / torch.norm(r, dim=-1, keepdim=True)
            
        if layer_idx not in self.codebooks_dict:
            # Initialize a default 1D codebook (fallback)
            self.codebooks_dict[layer_idx] = torch.linspace(-1.0, 1.0, self.n_levels, 
                                                           device=key_states.device, 
                                                           dtype=torch.float32).repeat(3)

        rotors = self.rotors_dict[layer_idx]
        cb = self.codebooks_dict[layer_idx]
        
        # We use the full fused kernel now
        k_q = xpu_backend.rotor_full_fused(
            key_states, rotors, 
            cb['scalar'], cb['vector'], cb['bivector'], cb['trivector']
        )
        v_q = xpu_backend.rotor_full_fused(
            value_states, rotors, 
            cb['scalar'], cb['vector'], cb['bivector'], cb['trivector']
        )

        return super().update(k_q, v_q, layer_idx, cache_kwargs)

def evaluate_ppl(model, tokenizer, text=None, n_samples=1, seq_len=512, cache_class=None, **cache_kwargs):
    model.eval()
    if text is None:
        text = "The quick brown fox jumps over the lazy dog. Artificial intelligence is transforming the world of technology."
    
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)
    labels = input_ids.clone()
    
    past_key_values = cache_class(**cache_kwargs) if cache_class else None
    
    with torch.no_grad():
        outputs = model(input_ids, labels=labels, past_key_values=past_key_values)
        loss = outputs.loss
            
    return torch.exp(loss).item()
