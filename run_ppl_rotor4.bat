@echo off
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" > nul
.\llamacpp\build\bin\llama-perplexity.exe -m .\models\Qwen3.5-2B-ROTOR4.gguf -f wikitext_test.txt -fa 1 -ngl 99 -c 512
