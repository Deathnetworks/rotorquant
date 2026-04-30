# RotorQuant XPU Continuity Handover - 2026-04-30

## Project Objective
Stabilize a high-fidelity performance benchmarking suite for Qwen3.5-2B on Intel Arc Pro B70 to validate custom Rotor/Iso SYCL kernels.

## Current Status: [DEBUGGING SYCL INITIALIZATION]
- **Goal**: Establish a stable SYCL device discovery flow on Intel Arc Pro B70.
- **Problem**: `std::bad_array_new_length` exception in `dpct::dev_mgr::instance().device_count()` during startup.
- **Recent Progress**: 
    - Implemented `try-catch` instrumentation to prevent silent hangs in `llama-cli`.
    - Identified that `dpct` wrapper failure is likely a driver/runtime mismatch for Arc Pro B70.
    - Successfully validated that `llama-cli --version` can now run to completion by catching the exception.

## Blockers
- [CRITICAL] **SYCL Runtime Discovery**: The `dpct` wrapper triggers a memory allocation error. Must bypass or fix to enable GPU acceleration.
- [RESOLVED] **Stale Binary Linking**: Forced clean relink of `ggml-sycl.dll` to ensure instrumentation is live.

## Pending Tasks
- [ ] Bypass `dpct::dev_mgr` using native `sycl::device::get_devices()` for enumeration.
- [ ] Investigate `ONEAPI_DEVICE_SELECTOR` environment variables to filter out problematic sub-devices.
- [ ] Run comprehensive PPL/Latency/VRAM benchmark suite once initialized.

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
