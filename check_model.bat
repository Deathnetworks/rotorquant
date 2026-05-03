@echo off
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" > nul
.\llamacpp\build\bin\llama-cli.exe --model .\models\Qwen3.5-2B-ISO4.gguf --verbose -n 1 -p "Hello"
