import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
import io

# Ensure UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_coherency(model_id="Qwen/Qwen2.5-3B-Instruct"):
    print(f"Testing Coherency for {model_id} on XPU...")
    device = torch.device("xpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load in FP16 to check base model
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    model.eval()
    
    prompts = [
        "What is the capital of France?",
        "Explain the importance of quantum computing in one sentence.",
        "Solve: 25 * 4 + 10 ="
    ]
    
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            response = tokenizer.decode(output_ids[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            print(f"\nQ: {prompt}")
            print(f"A: {response.strip()}")

if __name__ == "__main__":
    test_coherency()
