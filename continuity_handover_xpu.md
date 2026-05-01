# RotorQuant XPU Continuity Handover - 2026-05-01

## Project Objective
Finalizing RotorQuant SYCL Kernels (ISO4/ROTOR4) for parity with CPU reference and enabling performance benchmarking on Intel Arc Pro hardware.

## Current Status: [SYCL KERNELS IMPLEMENTED - LINKING BLOCKED]
- **Accomplishments**: 
    - `ISO4` and `ROTOR4` kernels fully implemented in `vecdotq.hpp`.
    - `vec_dot` signature updated to include `ibx` across all SYCL types (IQ, NVFP4, Q4/Q5/Q6/Q8).
    - MoE and MMVQ dispatchers updated for `ISO4`/`ROTOR4`.
    - Mathematical helpers (Deterministic Rotors/Quaternions) migrated to `vecdotq.hpp`.
- **Issue**: `mmvq.cpp` and `vecdotq.hpp` compile successfully, but final linking of `llama-cli` fails with `LNK1104: cannot open file 'libircmt.lib'`.
- **Resolution**: Need to verify Intel oneAPI library paths or re-source `setvars.bat` in the environment. Logic-wise, the SYCL integration is complete and verified against syntax/template errors.

## Blockers
- [RESOLVED] **Template Remainder by Zero**: Fixed `VDR` constants for `ISO4`/`ROTOR4` from 8 to 4 to match `QI4_0=4`.
- [RESOLVED] **Missing ibx in IQ Kernels**: Updated all `IQ` and `NVFP4` signatures and calls in `mmvq.cpp`.
- [IN PROGRESS] **Linker Error**: `libircmt.lib` missing during `llama-cli` link.

## Pending Tasks
- [ ] Resolve `libircmt.lib` linking error.
- [ ] Verify `ISO4`/`ROTOR4` bit-accuracy using `llama-cli`.
- [ ] Execute performance benchmarking for Qwen3.5-2B.
- [ ] Record final 36x8 performance matrix in `baseline_results_xpu.md`.

## Technical Failure Repository [PERSISTENT]
| Failure | Context | Reason | Resolved |
|---|---|---|---|
| Remainder by Zero | `mmvq.cpp` | `QI / VDR` was 0 for 4-byte blocks when VDR=8. | YES (Set VDR=4) |
| Undefined Function | SYCL Kernels | Signature mismatch (missing `ibx`) in `vec_dot` calls. | YES (Synced all signatures) |
| LNK1104: libircmt.lib | `llama-cli` link | Intel runtime library missing from linker path. | NO (Environment issue) |
| Organic VRAM Noise | `memory_allocated()` | IPEX allocator reserves large pages; delta for small contexts (512) was hidden or spiked to 6GB. | YES (Math Model) |
| Debug CRT Mismatch | `ggml-sycl.dll` | `CMAKE_BUILD_TYPE=Debug` caused deadlock/hang in SYCL runtime on B70. | YES (Switched to RelWithDebInfo) |

## Key Learnings & Ground Truths
- **VDR Alignment**: For `Q4_0` style layouts (32 weights, 16 bytes), `QI` is 4. `VDR` must be <= 4 to avoid division by zero in indexing logic.
- **Unified Signatures**: SYCL `vec_dot` templates require a strictly uniform signature (`vbq, bq8_1, iqs, ibx`) for the `mul_mat_vec_q` dispatcher.
- **Rotor Generation**: Rotors are deterministic based on `ibx`; logic must match `ggml-quants.c`.

## File References
- [implementation_plan.md](file:///C:/Users/deathnetworks/.gemini/antigravity/brain/69d885c6-b13b-4dac-b571-e4154dd001b1/implementation_plan.md): Technical roadmap for ROTOR4/ISO4.
- [vecdotq.hpp](file:///d:/User%20Files/Desktop/RotorQuant/llamacpp/ggml/src/ggml-sycl/vecdotq.hpp): Implementation of new kernels.
- [mmvq.cpp](file:///d:/User%20Files/Desktop/RotorQuant/llamacpp/ggml/src/ggml-sycl/mmvq.cpp): Dispatch logic.
- [baseline_results_xpu.md](file:///d:/User%20Files/Desktop/RotorQuant/baseline_results_xpu.md): Performance data.
- [RotorQuant Paper](/paper/rotorquant.md)
- [IsoQuant Paper 1](/paper/isoquant%20paper%201.md)
- [IsoQuant Paper 2](/paper/isoquant%20paper%202.md)
