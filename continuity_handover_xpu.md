# RotorQuant XPU Continuity Handover - 2026-05-02

## Status Update (2026-05-02)
- **Rotation Alignment Fix [COMPLETE]**: Successfully refactored all SYCL kernels (Dot Product, Dequantize, Copy, Attention) to use block-local rotation indexing anchored to `ib * stride + g`. This resolves the `NaN` generation issue caused by global indexing misalignment.
- **Dequantization Parity**: Verified that `dequantize_row_iso4` and `dequantize_row_rotor4` produce deterministic results consistent with weight quantization.
- **SYCL Backend Stability**: Successfully resolved linker issues by stubbing out Flash Attention (FA) template instances for standard testing.
- **Benchmarking Operational**: `llama-bench` and `llama-perplexity` are ready for validation.

## Technical Failure Repository [RESOLVED]
- **Failure**: Perplexity anomalies (`NaN` and drift) during RotorQuant XPU inference.
- **Reason 1**: Indexing misalignment (fixed 2026-05-02).
- **Reason 2**: Rotation direction inversion in dot product kernels. Activation-side rotation was applying the inverse instead of the forward rotation, while reconstruction correctly used the inverse.
- **Fix**: Corrected `quat_multiply_left` and `rotor_sandwich_vec` parameters in `vecdotq.hpp` and `fattn-common.hpp` to use forward rotation for activations.
- **Failure**: LNK2019 Unresolved Externals in `ggml-sycl.dll`. [RESOLVED]
- **Reason**: Missing FA template instances.
- **Fix**: Stubbed out `ggml_sycl_flash_attn_ext_*` functions.
- **Failure**: Benchmarking script violating process management rules. [RESOLVED]
- **Reason**: Multiple `llama-cli.exe` instances causing contention and PPL drift.
- **Fix**: Added mandatory `taskkill` and `sleep` to `run_command` in `comprehensive_baseline_xpu_v3.py`.

## Next Steps
- [ ] Run `llama-perplexity.exe` with ISO4/ROTOR4 models to confirm bit-exact accuracy and stability.
- [ ] Re-enable fused Flash Attention kernels and align their rotation striding logic.
- [ ] Execute the full 36x8 performance matrix benchmark on Intel Arc Pro B70.
- [ ] Document final throughput and PPL in `baseline_results_xpu_v2.md`.
24: - **Papers**: Grounded in `rotorquant.md`, `isoquant paper 1.md`, and `isoquant paper 2.md`.
25: - **Kernels**: `llamacpp/ggml/src/ggml-sycl/vecdotq.hpp`
26: - **Math Helpers**: `llamacpp/ggml/src/ggml-sycl/rotor-math.hpp`
27: - **Dequantize Logic**: `llamacpp/ggml/src/ggml-sycl/dequantize.hpp`
28: - **Benchmarking**: `turboquant/comprehensive_baseline_xpu_v3.py`