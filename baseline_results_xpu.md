# XPU Baseline Results - 2026-05-01 07:43

**Model**: Qwen/Qwen3.5-2B | **Hardware**: Intel Arc Pro B70 (32GB)

| Cache K | Cache V | Mode | Context | Prefill (tk/s) | Decode (tk/s) | VRAM (MB) | PPL | Needle | Coherency |
|---|---|---|---|---|---|---|---|---|---|
| FP16 | FP16 | Sync | 512 | 853.0 | 7.9 | 3614.0 | 13.71 | PASS | OK |
| FP16 | FP16 | Sync | 1024 | 1482.2 | 9.3 | 3628.0 | 13.71 | PASS | OK |
| FP16 | FP16 | Sync | 2048 | 2137.0 | 9.4 | 3656.0 | 13.71 | PASS | OK |
| FP16 | FP16 | Sync | 4096 | 2606.8 | 9.3 | 3712.0 | 13.71 | PASS | OK |
| FP16 | FP16 | Sync | 8192 | 3097.7 | 9.3 | 3824.0 | 13.71 | PASS | OK |
| FP16 | FP16 | Sync | 16384 | 2939.2 | 7.7 | 4048.0 | 13.71 | PASS | OK |
| FP16 | FP16 | Sync | 32768 | 2279.4 | 8.2 | 4496.0 | 13.71 | PASS | OK |
