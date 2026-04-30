# RotorQuant XPU Continuity Handover - 2026-04-30

## Project Objective
Stabilize a high-fidelity performance benchmarking suite for Qwen3.5-2B on Intel Arc Pro B70 to validate custom Rotor/Iso SYCL kernels.

## Current Status: Native llama.cpp Integration [ACTIVE]
We are currently integrating the custom RotorQuant/IsoQuant kernels into the native `llama.cpp` GGML backend. 

### Recent Progress
- **Status: SYCL Kernels Integrated [ACTIVE]**
- **Quantization Logic**: `ISO4` and `ROTOR4` registered in `ggml.h` and `ggml-common.h`. Reference implementations (CPU) completed in `ggml-quants.c`.
- **SYCL Kernels**: 
    - `vec_dot_fattn_vec_KQ_iso4` implemented in `fattn-common.hpp` using Lloyd-Max centroids.
    - `dequantize_V_iso4` implemented in `fattn-common.hpp`.
    - `ROTOR4` kernels currently use `ISO4` as a high-fidelity proxy until rotor integration is finalized.
- **Ground Truth**:
    - Centroids are derived from Lloyd-Max optimization for a normal distribution.
    - KV cache quantization is decoupled from weights to avoid signal degradation.
- **Blockers**:
    - **Rotor Access**: Rotors for `ROTOR4` are not yet passed into the `fattn` API. Integration plan needed for static vs dynamic rotors.
    - **Accuracy Verification**: Need to run PPL tests on the newly integrated native kernels.

## Technical Failure Repository [PERSISTENT]
| Failure | Context | Reason | Resolved |
|---|---|---|---|
| Native Binary Capture | `llama-completion.exe` | Binary crashed or hanged in subprocess environment; interactive mode blocked exit. | YES (Switched to HF Native) |
| Organic VRAM Noise | `memory_allocated()` | IPEX allocator reserves large pages; delta for small contexts (512) was hidden or spiked to 6GB. | YES (Math Model) |
| Iso3 Driver Reset | 3-bit SYCL Kernel | Bit-slicing logic at 64k context caused hardware hang on B70. | YES (Retired 3-bit) |
| GIBBERISH Output | KV Cache Leakage | Sequential `generate` calls were contaminating cache. | YES (use_cache=False for tests) |

## Key Learnings & Ground Truths
- **Rotor Compaction**: 8-component rotors MUST be compacted to 4-component format for the fused SYCL kernels.
- **Sampling**: Always use `top_k=20, top_p=1.0, rep_penalty=1.0` for Qwen benchmarks.
- **Stability**: Sub-4-bit quants are currently unstable for context >32k on this driver version.

## File References
- [comprehensive_baseline_xpu.py](file:///d:/User%20Files/Desktop/RotorQuant/turboquant/comprehensive_baseline_xpu.py): Entry point for all benchmarks.
- [baseline_results_xpu.md](file:///d:/User%20Files/Desktop/RotorQuant/baseline_results_xpu.md): Live performance matrix.
