@echo off
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" > nul
.\llamacpp\build\bin\llama-bench.exe -h
