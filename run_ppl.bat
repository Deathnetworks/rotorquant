@echo off
setlocal enabledelayedexpansion

:: initializing oneAPI environment...
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 --force > nul 2>&1

:: set PATH
set "PATH=%~dp0llamacpp\build\bin;%PATH%"

:: run llama-perplexity
llama-perplexity.exe %*
