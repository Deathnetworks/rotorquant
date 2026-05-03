@echo off
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" > nul
cd llamacpp\build
cmake --build . --config Release -j 12 --target ggml-sycl
