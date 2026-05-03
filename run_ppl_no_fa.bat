@echo off
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" > nul
.\llamacpp\build\bin\llama-perplexity.exe -m .\models\Qwen3.5-2B-ISO4.gguf -f wikitext_test.txt -fa 0 -ngl 99 -c 512
