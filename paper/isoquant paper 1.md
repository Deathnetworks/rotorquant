Below is the Markdown conversion of the paper "IsoQuant: Quaternion-based Isoclinic Rotation for Hardware-Aligned KV-Cache Compression" from the provided repository.

***

# IsoQuant: Quaternion-based Isoclinic Rotation for Hardware-Aligned KV-Cache Compression

## Abstract
Large Language Model (LLM) serving is often bottlenecked by the memory bandwidth and capacity of the Key-Value (KV) cache. Recent works like TurboQuant and RotorQuant have proposed using random or structured orthogonal rotations to "spread" the outlier features of model activations, enabling more aggressive scalar quantization. However, TurboQuant's dense rotations are computationally expensive ($O(d^2)$), and RotorQuant's 3D Clifford-algebra blocks do not align well with the power-of-two dimensions (e.g., 64, 128, 256) common in modern LLM architectures.

We introduce **IsoQuant**, a quantization framework based on quaternion algebra and the isoclinic decomposition of $SO(4)$. IsoQuant replaces 3D blocks with hardware-aligned 4D quaternion blocks. This approach provides stronger local mixing than 3D rotations while maintaining significantly lower computational overhead than dense rotations.

---

## 1. Introduction
Efficient LLM inference requires compressing the KV-cache to reduce memory footprint. Standard scalar quantization often fails due to high-magnitude "outliers" in specific feature dimensions. Orthogonal rotations can mitigate this by rotating the activation vector such that the energy is distributed more uniformly across all dimensions.

While dense rotations (TurboQuant) achieve the best distribution, they introduce $O(d^2)$ overhead. RotorQuant improved this by using 3D rotations based on Clifford algebra. However, 3D chunks are "awkward" for standard GPU memory layouts and LLM head dimensions. IsoQuant addresses this by leveraging 4D blocks derived from quaternion rotations.

---

## 2. Methodology: The 4D Construction
The core of IsoQuant is the application of $SO(4)$ rotations parameterized by unit quaternions. Every rotation in 4D space can be decomposed into two isoclinic rotations (left and right).

### 2.1 IsoQuant Variants
We implement three variants of the transformation:

1.  **IsoQuant-Full**: $v \to q_L \cdot v \cdot \bar{q}_R$  
    The most expressive variant, using two unit quaternions per 4D block.
2.  **IsoQuant-Fast**: $v \to q_L \cdot v$  
    A lower-cost variant using only one isoclinic factor (left-multiplication).
3.  **IsoQuant-2D**: $u \to R(\theta)u$  
    A lightweight planar special case used as an auxiliary low-cost point.

### 2.2 Complexity Comparison ($d=128$)
| Method | Block Structure | Parameters | FMAs |
| :--- | :--- | :--- | :--- |
| **TurboQuant** | Dense $128 \times 128$ | 16,384 | 16,384 |
| **RotorQuant** | $43 \times 3D$ blocks | 172 | ~2,408 |
| **IsoQuant-2D** | $64 \times 2D$ blocks | 128 | ~256 |
| **IsoQuant-Full**| $32 \times 4D$ blocks | 256 | 1,024 |
| **IsoQuant-Fast**| $32 \times 4D$ blocks | 128 | 512 |

---

## 3. Implementation
IsoQuant is implemented with fused CUDA kernels to ensure that the rotation does not become a bottleneck during the quantization path.

### 3.1 Kernel Performance
Benchmarks across $d \in \{128, 256, 512\}$ and precision levels (2, 3, 4 bits) show:
*   **Reconstruction Accuracy**: IsoQuant maintains a reconstruction Mean Squared Error (MSE) essentially identical to RotorQuant.
*   **Hardware Alignment**: The 4D block structure aligns perfectly with 128-bit and 256-bit memory transactions, leading to improved cache utilization compared to 3D-based methods.

---

## 4. Conclusion and Future Work
IsoQuant provides a hardware-friendly path for KV-cache compression. By utilizing 4D quaternion blocks, it bridges the gap between the efficiency of structured rotations and the power-of-two alignment required by modern GPU architectures.

**Next Steps:**
*   Validation of end-to-end KV-cache quality on real model activations.
*   Integration with stage-2 residual correction paths (e.g., QJL).
*   Measurement of attention-logit fidelity and perplexity on Llama-3 and Mistral models.

---

## Usage (Quick Start)
To run the validation and CUDA benchmarks:
```bash
# Run validation
PYTHONPATH=. python -m isoquant.validate_isoquant

# Run CUDA benchmarks
PYTHONPATH=. python -m isoquant.benchmark_cuda
```