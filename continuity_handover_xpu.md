# RotorQuant XPU Continuity Handover - 2026-05-01

3: ## Status Update (2026-05-01)
4: - **SYCL Backend Success**: ISO4 and ROTOR4 are fully integrated and verified on Intel Arc Pro B70.
5: - **Benchmarking v3 (In-Progress)**: Executing improved benchmark with sliding window PPL, proper Needle test, and factual coherency.
6: - **PPL Correction**: Implemented `--ppl-stride 512` to resolve the "90,000+ PPL" issue caused by zero context overlap.
7: - **Needle/Coherency Fixes**: Transitioned from trivial "1 2 3" tests to factual retrieval and secret key extraction in large context.
8: - **VRAM Metrics**: Enabled `GGML_SYCL_DEBUG=1` to capture detailed memory breakdown (context vs compute).
9: 
10: ## Technical Failure Repository [RESOLVED]
11: - **Failure**: `llama-cli` process hang during automated benchmarking.
12: - **Reason**: Interactive mode defaults.
13: - **Fix**: Use `--single-turn` flag.
14: - **Failure**: Inaccurate PPL (90k+).
15: - **Reason**: Lack of context overlap (stride) and format mismatch.
16: - **Fix**: Use `--ppl-stride 512` and proper instruct templates.
17: 
18: ## Next Steps
19: - [ ] Complete `comprehensive_baseline_xpu_v3.py` execution.
20: - [ ] Finalize the performance matrix in `baseline_results_xpu_v2.md`.
21: - [ ] Analyze the PPL gap between Q8 and ISO4/ROTOR4.
22: 
23: ## Ground Truth References
24: - **Papers**: Grounded in `rotorquant.md`, `isoquant paper 1.md`, and `isoquant paper 2.md`.
25: - **Kernels**: `llamacpp/ggml/src/ggml-sycl/vecdotq.hpp`
26: - **Math Helpers**: `llamacpp/ggml/src/ggml-sycl/rotor-math.hpp`
27: - **Dequantize Logic**: `llamacpp/ggml/src/ggml-sycl/dequantize.hpp`
28: - **Benchmarking**: `turboquant/comprehensive_baseline_xpu_v3.py`