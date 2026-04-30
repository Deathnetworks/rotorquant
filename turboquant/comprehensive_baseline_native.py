import subprocess
import os
import re

ONEAPI_SETVARS = r"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
LLAMA_BENCH = r"d:\User Files\Desktop\RotorQuant\llamacpp\build\bin\llama-bench.exe"
MODEL_PATH = r"d:\User Files\Desktop\RotorQuant\models\Qwen3.5-2B-BF16.gguf"
REPORT_PATH = r"d:\User Files\Desktop\RotorQuant\baseline_results_native.md"

def main():
    contexts = [512, 1024, 2048, 4096, 8192, 16384, 32768, 64000]
    
    with open(REPORT_PATH, "w") as f:
        f.write("# Native llama.cpp Performance Matrix (llama-bench)\n\n")
        f.write("| Mode | Context | Prefill (t/s) | Decode (t/s) |\n")
        f.write("|---|---|---|---|\n")

    for ctx in contexts:
        print(f">> Benchmarking context {ctx}...")
        
        # Use shell=True and let cmd.exe handle 'call'
        cmd = f'call "{ONEAPI_SETVARS}" && "{LLAMA_BENCH}" -m "{MODEL_PATH}" -p {ctx} -n 32 -ngl 99 -fa 1'
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True, encoding="utf-8", errors="ignore")
            output = result.stdout + result.stderr
            
            prefill = "0.0"
            decode = "0.0"
            
            rows = output.split('\n')
            for row in rows:
                if f"pp{ctx}" in row:
                    parts = [p.strip() for p in row.split('|') if p.strip()]
                    if len(parts) >= 8:
                        val = parts[7].split(' ')[0]
                        prefill = val
                if "tg32" in row:
                    parts = [p.strip() for p in row.split('|') if p.strip()]
                    if len(parts) >= 8:
                        val = parts[7].split(' ')[0]
                        decode = val

            if prefill == "0.0" and decode == "0.0":
                 print(f"DEBUG: Output for {ctx}:\n{output}")

            with open(REPORT_PATH, "a") as f:
                f.write(f"| FP16 (Native) | {ctx} | {prefill} | {decode} |\n")
            print(f"Done {ctx}: {prefill} / {decode}")
            
        except Exception as e:
            print(f"Error at {ctx}: {e}")

if __name__ == "__main__":
    main()
