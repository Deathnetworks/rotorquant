@echo off
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" > nul
.\llamacpp\build\bin\llama-bench.exe -m .\models\Qwen3.5-2B-Q8_0.gguf -p 512 -n 128 -fa 1 -ngl 99
