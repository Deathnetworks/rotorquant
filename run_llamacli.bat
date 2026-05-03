@echo off
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
.\llamacpp\build\bin\llama-cli.exe %*
