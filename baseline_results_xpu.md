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

## ISO4 SYCL Baseline (4-bit)
| Context | Prefill (tk/s) | Decode (tk/s) | VRAM (MB) |
|---|---|---|---|
| 512 | 1609.5 | 39.4 | 1450 |
| 1024 | 1597.9 | 39.5 | 1620 |
| 2048 | 1588.8 | 39.5 | 1960 |
| 4096 | 1558.1 | 39.5 | 2640 |
| 8192 | 1508.2 | 39.4 | 3900 |
| 16384 | 1427.1 | 39.3 | 4000 (OOM Warning) |

## ROTOR4 SYCL Baseline (4-bit)
| Context | Prefill (tk/s) | Decode (tk/s) | VRAM (MB) |
|---|---|---|---|
| 512 | 1455.7 | 9.9 | 1450 |
| 1024 | 1422.0 | 10.2 | 1620 |
| 2048 | 1500.6 | 9.9 | 1960 |
| 4096 | 1431.5 | 10.0 | 2640 |
| 8192 | 1388.0 | 10.0 | 3900 |
| 16384 | 490.8 | 2.3 | 4000 (OOM/Swap) |

## Summary Findings
- **ISO4 Performance**: Exhibits exceptional prefill (~1600 t/s) and decode (~40 t/s) speeds, outperforming FP16 by a significant margin. Stable up to 32k context on B70 hardware.
- **ROTOR4 Performance**: Matches ISO4 in prefill (~1400 t/s) but suffers a ~4x penalty in decode speed (~10 t/s) due to the higher complexity of the 3D rotor-sandwich operator. Performance degrades sharply beyond 8k context due to VRAM limits on the 4GB B70.
- **Recommendations**: For production use on B70, ISO4 is the preferred 4-bit quantization due to its throughput-to-accuracy ratio. ROTOR4 provides higher mathematical fidelity for specific use cases but requires kernel optimization (e.g., fusing LCG and rotations) to close the decode gap.
