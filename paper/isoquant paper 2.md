# IsoQuant: Hardware-Aligned $SO(4)$ Isoclinic Rotations for LLM KV-Cache Compression

## Abstract
Orthogonal feature decorrelation is effective for low-bit online vector quantization, but dense random orthogonal transforms incur prohibitive $O(d^2)$ storage and compute costs. RotorQuant reduces this cost with blockwise 3D Clifford rotors, yet the resulting 3D partition is poorly aligned with modern hardware and offers limited local mixing. We propose **IsoQuant**, a blockwise rotation framework based on quaternion algebra and the isoclinic decomposition of $SO(4)$. It represents each 4D block as a quaternion and applies a closed-form transform $T(v) = q_L v \overline{q_R}$. This yields two main variants: *IsoQuant-Full*, which realizes the full $SO(4)$ rotation, and *IsoQuant-Fast*, which keeps only one isoclinic factor for lower cost; the framework also admits a lightweight 2D special case. At $d=128$, IsoQuant-Full reduces forward rotation cost from about 2,408 FMAs in RotorQuant to 1,024, while IsoQuant-Fast further reduces it to 512.

---

## 1. Introduction
KV-cache compression is a central systems bottleneck for long-context LLM inference. A core insight behind online vector quantization methods such as TurboQuant is that decorrelating features before scalar quantization substantially improves rate–distortion behavior. While dense rotations achieve optimal distribution, they are computationally expensive. Moving from 3D Clifford blocks to 4D quaternion blocks leverages the Lie-theoretic decomposition $so(4) \cong su(2) \oplus su(2)$, which implies that every 4D rotation can be represented by a pair of unit quaternions acting from the left and right. This yields a closed-form, low-overhead parameterization of $SO(4)$ that is both mathematically clean and implementation-friendly.

---

## 2. Methodology: The 4D Construction
### 2.1 The Quaternion Transform
Each 4D block of an input vector $v \in \mathbb{R}^4$ is identified with a quaternion:
$$v = x_0 + x_1\mathbf{i} + x_2\mathbf{j} + x_3\mathbf{k} \in \mathbb{H}$$
The transformation is defined by the map:
$$T(v) = q_L v \overline{q_R}$$
where $q_L, q_R \in S^3$ are unit quaternions. This map preserves the Euclidean norm on $\mathbb{H} \cong \mathbb{R}^4$, ensuring the transformation is orthogonal.

### 2.2 Variants
1.  **IsoQuant-Full**: Realizes the full six degrees of freedom of $SO(4)$ using both $q_L$ and $q_R$.
2.  **IsoQuant-Fast**: Retains a single isoclinic factor ($T(v) = q_L v$), providing lower computational overhead.
3.  **IsoQuant-2D**: A lightweight planar special case used as an auxiliary operating point.

---

## 3. Systems and Hardware Alignment
IsoQuant’s 4D partition provides several advantages for systems deployment:

*   **Alignment**: Most transformer head dimensions are powers of two (e.g., 64, 128, 256). A 4D partition avoids the pathological tails induced by 3D chunking found in other methods. At $d=128$, IsoQuant uses exactly 32 blocks.
*   **Vectorization**: Four-wide blocks fit naturally into SIMD-friendly load and store patterns (e.g., `float4`). This reduces boundary checks and helps both CPU SIMD backends and GPU kernels maintain regular control flow.
*   **Kernel Efficiency**: The structure allows for fused CUDA kernel implementations that avoid materializing large dense rotation matrices, operating instead directly on the closed-form quaternion representation.

---

## 4. Performance Comparison ($d=128$)

| Method | Block Structure | Parameters | FMAs (Approx.) |
| :--- | :--- | :--- | :--- |
| **TurboQuant** | Dense $d \times d$ | 16,384 | 16,384 |
| **RotorQuant** | $43 \times 3D$ blocks | 372 | ~2,408 |
| **IsoQuant-Full** | $32 \times 4D$ blocks | 256 | 1,024 |
| **IsoQuant-Fast** | $32 \times 4D$ blocks | 128 | 512 |

---

## 5. Conclusion
IsoQuant bridges the gap between the expressive power of global orthogonal transforms and the efficiency of structured blockwise methods. By utilizing the isoclinic decomposition of $SO(4)$, it provides a hardware-aligned path for KV-cache compression that maintains high reconstruction fidelity while significantly reducing the overhead of the rotation step.