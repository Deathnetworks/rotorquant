#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <c10/xpu/XPUStream.h>
#include <cmath>

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif

template <typename T> inline float convert_to_float(T value) { return (float)value; }
template <> inline float convert_to_float<c10::Half>(c10::Half value) { return (float)value; }
template <> inline float convert_to_float<float>(float value) { return value; }
template <> inline float convert_to_float<at::BFloat16>(at::BFloat16 value) { return (float)value; }

template <typename T> inline T convert_from_float(float value) { return (T)value; }
template <> inline c10::Half convert_from_float<c10::Half>(float value) { return (c10::Half)value; }
template <> inline float convert_from_float<float>(float value) { return value; }
template <> inline at::BFloat16 convert_from_float<at::BFloat16>(float value) { return (at::BFloat16)value; }

#define WARP_SIZE 32

/*
 * Sparse geometric product: rotor * multivector (rotor on LEFT)
 *
 * Computes R * x where R = [s, 0, 0, 0, p12, p13, p23, 0] in Cl(3,0).
 * 28 FMAs total.
 */
inline void gp_rotor_mv(const float s, const float p12, const float p13, const float p23,
                         const float x[8], float r[8]) {
    r[0] = s * x[0] - p12 * x[4] - p13 * x[5] - p23 * x[6];
    r[1] = s * x[1] + p12 * x[2] + p13 * x[3] + p23 * x[7];
    r[2] = s * x[2] - p12 * x[1] + p23 * x[3] - p13 * x[7];
    r[3] = s * x[3] - p13 * x[1] - p23 * x[2] + p12 * x[7];
    r[4] = s * x[4] + p12 * x[0] + p13 * x[6] - p23 * x[5];
    r[5] = s * x[5] + p13 * x[0] - p12 * x[6] + p23 * x[4];
    r[6] = s * x[6] + p23 * x[0] + p12 * x[5] - p13 * x[4];
    r[7] = s * x[7] - p23 * x[1] + p13 * x[2] - p12 * x[3];
}

/*
 * Sparse geometric product: multivector * rotor (rotor on RIGHT)
 *
 * Computes x * R where R = [s, 0, 0, 0, p12, p13, p23, 0] in Cl(3,0).
 * NOTE: This is DIFFERENT from R * x in non-commutative Clifford algebra.
 * 28 FMAs total.
 */
inline void gp_mv_rotor(const float x[8], const float s, const float p12, const float p13, const float p23,
                         float r[8]) {
    r[0] = s * x[0] - p12 * x[4] - p13 * x[5] - p23 * x[6];
    r[1] = s * x[1] - p12 * x[2] - p13 * x[3] + p23 * x[7];
    r[2] = s * x[2] + p12 * x[1] - p23 * x[3] - p13 * x[7];
    r[3] = s * x[3] + p13 * x[1] + p23 * x[2] + p12 * x[7];
    r[4] = s * x[4] + p12 * x[0] + p23 * x[5] - p13 * x[6];
    r[5] = s * x[5] + p13 * x[0] - p23 * x[4] + p12 * x[6];
    r[6] = s * x[6] + p23 * x[0] + p13 * x[4] - p12 * x[5];
    r[7] = s * x[7] + p23 * x[1] - p13 * x[2] + p12 * x[3];
}

inline float quantize_scalar(float val, const float* __restrict__ centroids, int levels) {
    float best = centroids[0];
    float min_d = sycl::fabs(val - best);
    for (int i = 1; i < levels; ++i) {
        float d = sycl::fabs(val - centroids[i]);
        if (d < min_d) { min_d = d; best = centroids[i]; }
    }
    return best;
}

/*
 * Fused RotorQuant kernel:
 *   embed → rotor_sandwich_fwd → quantize → rotor_sandwich_inv → extract
 *
 * One work-group per batch item. Work-items iterate over groups.
 * Rotors stored in COMPACT (n_groups, 4) layout: [s, p12, p13, p23]
 * matching the CUDA reference kernel exactly.
 */
template <typename T>
[[intel::reqd_sub_group_size(32)]]
void rotor_full_fused_kernel(
    const T* __restrict__ input, const float* __restrict__ rotors,
    const float* __restrict__ c_scalar, int n_scalar, const float* __restrict__ c_vector, int n_vector,
    const float* __restrict__ c_bivector, int n_bivector, const float* __restrict__ c_trivector, int n_trivector,
    T* __restrict__ output, int batch_size, int emb_dim, int n_groups,
    float* sh_rotors, float* sh_c_scalar, float* sh_c_vector, float* sh_c_bivector, float* sh_c_trivector) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int bid = item_ct1.get_group(2), tid = item_ct1.get_local_id(2), threads = item_ct1.get_local_range(2);

    // Load compact rotors: (n_groups, 4) = [s, p12, p13, p23]
    for (int i = tid; i < n_groups * 4; i += threads) sh_rotors[i] = rotors[i];
    for (int i = tid; i < n_scalar; i += threads) sh_c_scalar[i] = c_scalar[i];
    for (int i = tid; i < n_vector; i += threads) sh_c_vector[i] = c_vector[i];
    for (int i = tid; i < n_bivector; i += threads) sh_c_bivector[i] = c_bivector[i];
    for (int i = tid; i < n_trivector; i += threads) sh_c_trivector[i] = c_trivector[i];
    item_ct1.barrier(sycl::access::fence_space::local_space);

    const T* in_ptr = input + bid * emb_dim;
    T* out_ptr = output + bid * emb_dim;

    for (int g = tid; g < n_groups; g += threads) {
        // Load compact rotor
        float s   = sh_rotors[g * 4 + 0];
        float p12 = sh_rotors[g * 4 + 1];
        float p13 = sh_rotors[g * 4 + 2];
        float p23 = sh_rotors[g * 4 + 3];

        // Embed: 3 vector dims → multivector (grade-1 only)
        int d0 = g * 3;
        float x_mv[8] = {0.0f};
        if (d0     < emb_dim) x_mv[1] = convert_to_float<T>(in_ptr[d0]);
        if (d0 + 1 < emb_dim) x_mv[2] = convert_to_float<T>(in_ptr[d0 + 1]);
        if (d0 + 2 < emb_dim) x_mv[3] = convert_to_float<T>(in_ptr[d0 + 2]);

        // Forward sandwich: R x R̃ = (R * x) * R̃
        float temp[8], rotated[8];
        gp_rotor_mv(s, p12, p13, p23, x_mv, temp);
        gp_mv_rotor(temp, s, -p12, -p13, -p23, rotated);

        // Grade-aware quantization (matching CUDA: all 8 components)
        float q_mv[8];
        q_mv[0] = quantize_scalar(rotated[0], sh_c_scalar,   n_scalar);
        q_mv[1] = quantize_scalar(rotated[1], sh_c_vector,   n_vector);
        q_mv[2] = quantize_scalar(rotated[2], sh_c_vector,   n_vector);
        q_mv[3] = quantize_scalar(rotated[3], sh_c_vector,   n_vector);
        q_mv[4] = quantize_scalar(rotated[4], sh_c_bivector, n_bivector);
        q_mv[5] = quantize_scalar(rotated[5], sh_c_bivector, n_bivector);
        q_mv[6] = quantize_scalar(rotated[6], sh_c_bivector, n_bivector);
        q_mv[7] = quantize_scalar(rotated[7], sh_c_trivector,n_trivector);

        // Inverse sandwich: R̃ q R = (R̃ * q) * R
        float temp2[8], final_mv[8];
        gp_rotor_mv(s, -p12, -p13, -p23, q_mv, temp2);
        gp_mv_rotor(temp2, s, p12, p13, p23, final_mv);

        // Extract vector grades back to output
        if (d0     < emb_dim) out_ptr[d0]     = convert_from_float<T>(final_mv[1]);
        if (d0 + 1 < emb_dim) out_ptr[d0 + 1] = convert_from_float<T>(final_mv[2]);
        if (d0 + 2 < emb_dim) out_ptr[d0 + 2] = convert_from_float<T>(final_mv[3]);
    }
}

/*
 * Optimized fused kernel for grade-1 inputs (KV cache vectors).
 *
 * Key insight: For grade-1 input, the sandwich R*x*R~ produces non-zero output
 * ONLY in grades 1,2,3 (vector) and grade 7 (trivector).
 * Grades 0 (scalar) and 4,5,6 (bivector) are exactly zero.
 *
 * This kernel quantizes only the 4 meaningful components, cutting:
 *   - Quantization work by 50%
 *   - Storage per group from 8 to 4 values
 *   - Inverse compute (4 zeros skip multiplies)
 */
template <typename T>
[[intel::reqd_sub_group_size(32)]]
void rotor_fused_vec_kernel(
    const T* __restrict__ input, const float* __restrict__ rotors,
    const float* __restrict__ c_vector, int n_vector,
    const float* __restrict__ c_trivector, int n_trivector,
    T* __restrict__ output, int batch_size, int emb_dim, int n_groups,
    float* sh_rotors, float* sh_c_vector, float* sh_c_trivector) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int bid = item_ct1.get_group(2), tid = item_ct1.get_local_id(2), threads = item_ct1.get_local_range(2);

    for (int i = tid; i < n_groups * 4; i += threads) sh_rotors[i] = rotors[i];
    for (int i = tid; i < n_vector; i += threads) sh_c_vector[i] = c_vector[i];
    for (int i = tid; i < n_trivector; i += threads) sh_c_trivector[i] = c_trivector[i];
    item_ct1.barrier(sycl::access::fence_space::local_space);

    const T* in_ptr = input + bid * emb_dim;
    T* out_ptr = output + bid * emb_dim;

    for (int g = tid; g < n_groups; g += threads) {
        float s   = sh_rotors[g * 4 + 0];
        float p12 = sh_rotors[g * 4 + 1];
        float p13 = sh_rotors[g * 4 + 2];
        float p23 = sh_rotors[g * 4 + 3];

        // Embed grade-1
        int d0 = g * 3;
        float x_mv[8] = {0.0f};
        if (d0     < emb_dim) x_mv[1] = convert_to_float<T>(in_ptr[d0]);
        if (d0 + 1 < emb_dim) x_mv[2] = convert_to_float<T>(in_ptr[d0 + 1]);
        if (d0 + 2 < emb_dim) x_mv[3] = convert_to_float<T>(in_ptr[d0 + 2]);

        // Forward sandwich
        float temp[8], rotated[8];
        gp_rotor_mv(s, p12, p13, p23, x_mv, temp);
        gp_mv_rotor(temp, s, -p12, -p13, -p23, rotated);

        // Quantize ONLY non-zero grades: vector (1,2,3) and trivector (7)
        // Grades 0,4,5,6 are exactly zero for grade-1 inputs
        float q_mv[8] = {0.0f};
        q_mv[1] = quantize_scalar(rotated[1], sh_c_vector, n_vector);
        q_mv[2] = quantize_scalar(rotated[2], sh_c_vector, n_vector);
        q_mv[3] = quantize_scalar(rotated[3], sh_c_vector, n_vector);
        q_mv[7] = quantize_scalar(rotated[7], sh_c_trivector, n_trivector);

        // Inverse sandwich
        float temp2[8], final_mv[8];
        gp_rotor_mv(s, -p12, -p13, -p23, q_mv, temp2);
        gp_mv_rotor(temp2, s, p12, p13, p23, final_mv);

        // Extract
        if (d0     < emb_dim) out_ptr[d0]     = convert_from_float<T>(final_mv[1]);
        if (d0 + 1 < emb_dim) out_ptr[d0 + 1] = convert_from_float<T>(final_mv[2]);
        if (d0 + 2 < emb_dim) out_ptr[d0 + 2] = convert_from_float<T>(final_mv[3]);
    }
}



/*
 * Standalone rotor sandwich (no quantization).
 * Uses COMPACT (n_groups, 4) rotor layout.
 */
template <typename T>
[[intel::reqd_sub_group_size(32)]]
void rotor_sandwich_kernel(const T* __restrict__ input, const float* __restrict__ rotors, T* __restrict__ output, int batch_size, int emb_dim, int n_groups, float* sh_rotors) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int bid = item_ct1.get_group(2), tid = item_ct1.get_local_id(2), threads = item_ct1.get_local_range(2);
    for (int i = tid; i < n_groups * 4; i += threads) sh_rotors[i] = rotors[i];
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const T* in_ptr = input + bid * emb_dim;
    T* out_ptr = output + bid * n_groups * 8;
    for (int g = tid; g < n_groups; g += threads) {
        float x_mv[8] = {0,0,0,0,0,0,0,0};
        int d0 = g * 3;
        if (d0     < emb_dim) x_mv[1] = convert_to_float<T>(in_ptr[d0]);
        if (d0 + 1 < emb_dim) x_mv[2] = convert_to_float<T>(in_ptr[d0 + 1]);
        if (d0 + 2 < emb_dim) x_mv[3] = convert_to_float<T>(in_ptr[d0 + 2]);

        float s   = sh_rotors[g * 4 + 0];
        float p12 = sh_rotors[g * 4 + 1];
        float p13 = sh_rotors[g * 4 + 2];
        float p23 = sh_rotors[g * 4 + 3];

        float temp[8], rotated[8];
        gp_rotor_mv(s, p12, p13, p23, x_mv, temp);
        gp_mv_rotor(temp, s, -p12, -p13, -p23, rotated);

        int base = g * 8;
        for (int c = 0; c < 8; ++c) out_ptr[base + c] = convert_from_float<T>(rotated[c]);
    }
}

/*
 * Inverse rotor sandwich: reconstruct vectors from multivectors.
 * Uses COMPACT (n_groups, 4) rotor layout.
 */
template <typename T>
[[intel::reqd_sub_group_size(32)]]
void rotor_inverse_sandwich_kernel(const T* __restrict__ input, const float* __restrict__ rotors, T* __restrict__ output, int batch_size, int emb_dim, int n_groups, float* sh_rotors) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int bid = item_ct1.get_group(2), tid = item_ct1.get_local_id(2), threads = item_ct1.get_local_range(2);
    for (int i = tid; i < n_groups * 4; i += threads) sh_rotors[i] = rotors[i];
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const T* in_ptr = input + bid * n_groups * 8;
    T* out_ptr = output + bid * emb_dim;
    for (int g = tid; g < n_groups; g += threads) {
        float x_mv[8];
        int base = g * 8;
        for (int c = 0; c < 8; ++c) x_mv[c] = convert_to_float<T>(in_ptr[base + c]);

        float s   = sh_rotors[g * 4 + 0];
        float p12 = sh_rotors[g * 4 + 1];
        float p13 = sh_rotors[g * 4 + 2];
        float p23 = sh_rotors[g * 4 + 3];

        float temp[8], rotated[8];
        // Inverse sandwich: R̃ * x * R
        gp_rotor_mv(s, -p12, -p13, -p23, x_mv, temp);
        gp_mv_rotor(temp, s, p12, p13, p23, rotated);

        int d0 = g * 3;
        if (d0     < emb_dim) out_ptr[d0]     = convert_from_float<T>(rotated[1]);
        if (d0 + 1 < emb_dim) out_ptr[d0 + 1] = convert_from_float<T>(rotated[2]);
        if (d0 + 2 < emb_dim) out_ptr[d0 + 2] = convert_from_float<T>(rotated[3]);
    }
}


template <typename T>
torch::Tensor rotor_full_fused_impl(torch::Tensor input, torch::Tensor rotors, torch::Tensor c_scalar, int n_scalar, torch::Tensor c_vector, int n_vector, torch::Tensor c_bivector, int n_bivector, torch::Tensor c_trivector, int n_trivector) {
    int emb_dim = input.size(-1);
    int batch_size = input.numel() / emb_dim;
    int n_groups = (emb_dim + 2) / 3;
    auto output = torch::empty_like(input);
    int threads = std::min(256, std::max(n_groups, WARP_SIZE));
    c10::xpu::getCurrentXPUStream().queue().submit([&](sycl::handler &cgh) {
        // Compact rotor SLM: 4 floats per group (halved from 8)
        sycl::local_accessor<float, 1> sh_rotors(sycl::range<1>(n_groups * 4), cgh), sh_c_scalar(sycl::range<1>(n_scalar), cgh), sh_c_vector(sycl::range<1>(n_vector), cgh), sh_c_bivector(sycl::range<1>(n_bivector), cgh), sh_c_trivector(sycl::range<1>(n_trivector), cgh);
        auto in_p = input.data_ptr<T>();
        auto r_p = rotors.data_ptr<float>();
        auto cs_p = c_scalar.data_ptr<float>();
        auto cv_p = c_vector.data_ptr<float>();
        auto cb_p = c_bivector.data_ptr<float>();
        auto ct_p = c_trivector.data_ptr<float>();
        auto out_p = output.data_ptr<T>();
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, batch_size) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)), [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
            rotor_full_fused_kernel<T>(in_p, r_p, cs_p, n_scalar, cv_p, n_vector, cb_p, n_bivector, ct_p, n_trivector, out_p, batch_size, emb_dim, n_groups, sh_rotors.get_pointer(), sh_c_scalar.get_pointer(), sh_c_vector.get_pointer(), sh_c_bivector.get_pointer(), sh_c_trivector.get_pointer());
        });
    });
    return output;
}

template <typename T>
torch::Tensor rotor_sandwich_impl(torch::Tensor input, torch::Tensor rotors) {
    int emb_dim = input.size(-1);
    int batch_size = input.numel() / emb_dim;
    int n_groups = (emb_dim + 2) / 3;
    std::vector<int64_t> out_shape;
    for (int i=0; i<input.dim()-1; i++) out_shape.push_back(input.size(i));
    out_shape.push_back(n_groups);
    out_shape.push_back(8);
    auto output = torch::empty(out_shape, input.options());
    int threads = std::min(256, std::max(n_groups, WARP_SIZE));
    c10::xpu::getCurrentXPUStream().queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sh_rotors(sycl::range<1>(n_groups * 4), cgh);
        auto in_p = input.data_ptr<T>();
        auto r_p = rotors.data_ptr<float>();
        auto out_p = output.data_ptr<T>();
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, batch_size) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)), [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
            rotor_sandwich_kernel<T>(in_p, r_p, out_p, batch_size, emb_dim, n_groups, sh_rotors.get_pointer());
        });
    });
    return output;
}

template <typename T>
torch::Tensor rotor_inverse_sandwich_impl(torch::Tensor input, torch::Tensor rotors, int emb_dim) {
    int n_groups = input.size(-2);
    int batch_size = input.numel() / (n_groups * 8);
    std::vector<int64_t> out_shape;
    for (int i=0; i<input.dim()-2; i++) out_shape.push_back(input.size(i));
    out_shape.push_back(emb_dim);
    auto output = torch::empty(out_shape, input.options());
    int threads = std::min(256, std::max(n_groups, WARP_SIZE));
    c10::xpu::getCurrentXPUStream().queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sh_rotors(sycl::range<1>(n_groups * 4), cgh);
        auto in_p = input.data_ptr<T>();
        auto r_p = rotors.data_ptr<float>();
        auto out_p = output.data_ptr<T>();
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, batch_size) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)), [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
            rotor_inverse_sandwich_kernel<T>(in_p, r_p, out_p, batch_size, emb_dim, n_groups, sh_rotors.get_pointer());
        });
    });
    return output;
}

/*
 * Fused sandwich roundtrip: forward + inverse in ONE kernel launch.
 * Input: (batch, emb_dim) -> Output: (batch, emb_dim)
 * Does: embed -> R*x*R~ -> R~*y*R -> extract
 * This is the identity transform (minus FP rounding).
 * Used for parity testing and as a template for the quantized pipeline.
 */
template <typename T>
[[intel::reqd_sub_group_size(32)]]
void rotor_sandwich_roundtrip_kernel(const T* __restrict__ input, const float* __restrict__ rotors,
    T* __restrict__ output, int batch_size, int emb_dim, int n_groups, float* sh_rotors) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int bid = item_ct1.get_group(2), tid = item_ct1.get_local_id(2), threads = item_ct1.get_local_range(2);
    for (int i = tid; i < n_groups * 4; i += threads) sh_rotors[i] = rotors[i];
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const T* in_ptr = input + bid * emb_dim;
    T* out_ptr = output + bid * emb_dim;
    for (int g = tid; g < n_groups; g += threads) {
        // Load input as grade-1 multivector
        float x_mv[8] = {0,0,0,0,0,0,0,0};
        int d0 = g * 3;
        if (d0     < emb_dim) x_mv[1] = convert_to_float<T>(in_ptr[d0]);
        if (d0 + 1 < emb_dim) x_mv[2] = convert_to_float<T>(in_ptr[d0 + 1]);
        if (d0 + 2 < emb_dim) x_mv[3] = convert_to_float<T>(in_ptr[d0 + 2]);

        float s   = sh_rotors[g * 4 + 0];
        float p12 = sh_rotors[g * 4 + 1];
        float p13 = sh_rotors[g * 4 + 2];
        float p23 = sh_rotors[g * 4 + 3];

        // Forward sandwich: R * x * R~
        float t1[8], fwd[8];
        gp_rotor_mv(s, p12, p13, p23, x_mv, t1);
        gp_mv_rotor(t1, s, -p12, -p13, -p23, fwd);

        // Inverse sandwich: R~ * fwd * R
        float t2[8], inv[8];
        gp_rotor_mv(s, -p12, -p13, -p23, fwd, t2);
        gp_mv_rotor(t2, s, p12, p13, p23, inv);

        // Extract vector grades
        if (d0     < emb_dim) out_ptr[d0]     = convert_from_float<T>(inv[1]);
        if (d0 + 1 < emb_dim) out_ptr[d0 + 1] = convert_from_float<T>(inv[2]);
        if (d0 + 2 < emb_dim) out_ptr[d0 + 2] = convert_from_float<T>(inv[3]);
    }
}

template <typename T>
torch::Tensor rotor_sandwich_roundtrip_impl(torch::Tensor input, torch::Tensor rotors) {
    int emb_dim = input.size(-1);
    int batch_size = input.numel() / emb_dim;
    int n_groups = (emb_dim + 2) / 3;
    auto output = torch::empty_like(input);
    int threads = std::min(256, std::max(n_groups, WARP_SIZE));
    c10::xpu::getCurrentXPUStream().queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sh_rotors(sycl::range<1>(n_groups * 4), cgh);
        auto in_p = input.data_ptr<T>();
        auto r_p = rotors.data_ptr<float>();
        auto out_p = output.data_ptr<T>();
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, batch_size) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)), [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
            rotor_sandwich_roundtrip_kernel<T>(in_p, r_p, out_p, batch_size, emb_dim, n_groups, sh_rotors.get_pointer());
        });
    });
    return output;
}

template <typename T>
torch::Tensor rotor_fused_vec_impl(torch::Tensor input, torch::Tensor rotors,
    torch::Tensor c_vector, int n_vector, torch::Tensor c_trivector, int n_trivector) {
    int emb_dim = input.size(-1);
    int batch_size = input.numel() / emb_dim;
    int n_groups = (emb_dim + 2) / 3;
    auto output = torch::empty_like(input);
    int threads = std::min(256, std::max(n_groups, WARP_SIZE));
    c10::xpu::getCurrentXPUStream().queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sh_rotors(sycl::range<1>(n_groups * 4), cgh);
        sycl::local_accessor<float, 1> sh_c_vector(sycl::range<1>(n_vector), cgh);
        sycl::local_accessor<float, 1> sh_c_trivector(sycl::range<1>(n_trivector), cgh);
        auto in_p = input.data_ptr<T>();
        auto r_p = rotors.data_ptr<float>();
        auto cv_p = c_vector.data_ptr<float>();
        auto ct_p = c_trivector.data_ptr<float>();
        auto out_p = output.data_ptr<T>();
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, batch_size) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)), [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
            rotor_fused_vec_kernel<T>(in_p, r_p, cv_p, n_vector, ct_p, n_trivector,
                out_p, batch_size, emb_dim, n_groups,
                sh_rotors.get_pointer(), sh_c_vector.get_pointer(), sh_c_trivector.get_pointer());
        });
    });
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rotor_full_fused_half", &rotor_full_fused_impl<c10::Half>);
    m.def("rotor_full_fused_float", &rotor_full_fused_impl<float>);
    m.def("rotor_full_fused_bf16", &rotor_full_fused_impl<at::BFloat16>);
    m.def("rotor_sandwich_half", &rotor_sandwich_impl<c10::Half>);
    m.def("rotor_sandwich_float", &rotor_sandwich_impl<float>);
    m.def("rotor_sandwich_bf16", &rotor_sandwich_impl<at::BFloat16>);
    m.def("rotor_inverse_half", &rotor_inverse_sandwich_impl<c10::Half>);
    m.def("rotor_inverse_float", &rotor_inverse_sandwich_impl<float>);
    m.def("rotor_inverse_bf16", &rotor_inverse_sandwich_impl<at::BFloat16>);
    m.def("rotor_roundtrip_half", &rotor_sandwich_roundtrip_impl<c10::Half>);
    m.def("rotor_roundtrip_float", &rotor_sandwich_roundtrip_impl<float>);
    m.def("rotor_roundtrip_bf16", &rotor_sandwich_roundtrip_impl<at::BFloat16>);
    m.def("rotor_fused_vec_half", &rotor_fused_vec_impl<c10::Half>);
    m.def("rotor_fused_vec_float", &rotor_fused_vec_impl<float>);
    m.def("rotor_fused_vec_bf16", &rotor_fused_vec_impl<at::BFloat16>);
}
