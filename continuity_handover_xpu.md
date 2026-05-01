# RotorQuant XPU Continuity Handover - 2026-05-01

## Status Update (2026-05-01)
- **SYCL Backend Success**: ISO4 and ROTOR4 are fully integrated and verified on Intel Arc Pro B70.
- **Conversion Support**: Added dequantize kernels and registered types in `convert.cpp`, resolving model loading crashes.
- **Architecture**: Refactored rotation math into `rotor-math.hpp` for shared access between dot-product and dequantization kernels.
- **Performance**:
    - **ISO4**: ~37 t/s generation, ~84 t/s prompt (Qwen3.5-2B).
    - **ROTOR4**: ~10 t/s generation, ~12 t/s prompt.
- **Verification**: Bit-accurate loading and inference confirmed.

## Technical Failure Repository [RESOLVED]
- **Failure**: `LNK1104: libircmt.lib` during build.
- **Reason**: Missing OneAPI environment variables in the build shell.
- **Fix**: Explicitly call `setvars.bat` within the build command or `.bat` script.

- **Failure**: `unsupport data type=iso4` at runtime.
- **Reason**: SYCL conversion dispatcher in `convert.cpp` lacked entries for custom types.
- **Fix**: Implemented `dequantize_row_iso4_sycl` and `dequantize_row_rotor4_sycl` and added them to `ggml_get_to_fp16_sycl` and `ggml_get_to_fp32_sycl`.

## Next Steps
- [ ] Finalize 36x8 performance matrix for high-context windows.
- [ ] Optimize ROTOR4 kernel (currently ~3x slower than ISO4).
- [ ] Port to Xe2 (Lunar Lake) for mobile benchmarking.

## Ground Truth References
- **Kernels**: `llamacpp/ggml/src/ggml-sycl/vecdotq.hpp`
- **Math Helpers**: `llamacpp/ggml/src/ggml-sycl/rotor-math.hpp`
- **CPU Reference**: `llamacpp/ggml/src/ggml-quants.c` (Lines 313-405)
- **Registration**: `llamacpp/src/llama-model-loader.cpp`
- **RotorQuant Paper**: [RotorQuant](/paper/rotorquant.md)
- **IsoQuant Paper 1**: [IsoQuant](/paper/isoquant%20paper%201.md)
- **IsoQuant Paper 2**: [IsoQuant](/paper/isoquant%20paper%202.md)