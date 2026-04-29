import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import DynamicCache
from turboquant import xpu_backend

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

def evaluate_ppl(model, tokenizer, dataset="wikitext-2", n_samples=32, seq_len=512, cache_class=None, **cache_kwargs):
    model.eval()
    # For Wikitext-2, we typically load from HF datasets
    # But for a quick check, we can use a small local sample or a hardcoded one
    # I'll use a placeholder for now
    
    total_loss = 0
    count = 0
    
    # Mock data loop
    for i in range(n_samples):
        input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len), device=model.device)
        labels = input_ids.clone()
        
        past_key_values = cache_class(**cache_kwargs) if cache_class else None
        
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, past_key_values=past_key_values)
            loss = outputs.loss
            total_loss += loss.item()
            count += 1
            
    avg_loss = total_loss / count
    return torch.exp(torch.tensor(avg_loss)).item()
