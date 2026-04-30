import subprocess
import os
import re

ONEAPI_PATH = r"C:\Program Files (x86)\Intel\oneAPI\2025.3\bin"
LLAMACPP_BIN = r"d:\User Files\Desktop\RotorQuant\llamacpp\build\bin"
MODEL_PATH = r"d:\User Files\Desktop\RotorQuant\models\Qwen3.5-2B-BF16.gguf"

os.environ["PATH"] = f"{ONEAPI_PATH};{LLAMACPP_BIN};" + os.environ["PATH"]

def stress_test(ctx):
    print(f">> Stress testing context {ctx} to maximize power...")
    prompt_file = f"stress_{ctx}.txt"
    with open(prompt_file, "w") as f:
        f.write("POWER TEST " * (ctx // 2))

    cmd = [
        os.path.join(LLAMACPP_BIN, "llama-completion.exe"),
        "-m", MODEL_PATH,
        "-f", prompt_file,
        "-n", "256",
        "-ngl", "99",
        "-b", "512",
        "-ub", "512",
        "--no-display-prompt",
        "-no-cnv"
    ]
    
    try:
        # Run and print output live to see timings
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        
        if os.path.exists(prompt_file):
            os.remove(prompt_file)
    except Exception as e:
        print(f"Error: {e}")
        if os.path.exists(prompt_file):
            os.remove(prompt_file)

if __name__ == "__main__":
    # Test 64k context to pull maximum power
    stress_test(65536)
