#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <c10/xpu/XPUStream.h>
#include <cmath>

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif

template <typename T> inline float convert_to_float(T value) { return (float)value; }
template <> inline float convert_to_float<c10::Half>(c10::Half value) { return (float)value; }
template <> inline float convert_to_float<at::BFloat16>(at::BFloat16 value) { return (float)value; }

template <typename T> inline T convert_from_float(float value) { return (T)value; }
template <> inline c10::Half convert_from_float<c10::Half>(float value) { return (c10::Half)value; }
template <> inline at::BFloat16 convert_from_float<at::BFloat16>(float value) { return (at::BFloat16)value; }

#define WARP_SIZE 32

/* ── CLIFFORD 3D MATH ────────────────────────────────────────────────── */

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

/* ── QUANTIZATION UTILS ─────────────────────────────────────────────── */

inline float quantize_scalar_bs(float val, const float* __restrict__ centroids, int levels) {
    if (levels <= 0) return val;
    int low = 0, high = levels - 1;
    while (low < high) {
        int mid = (low + high) / 2;
        float mid_val = (centroids[mid] + centroids[mid+1]) * 0.5f;
        if (val < mid_val) high = mid;
        else low = mid + 1;
    }
    return centroids[low];
}

inline int quantize_scalar_idx(float val, const float* __restrict__ centroids, int levels) {
    if (levels <= 0) return 0;
    int low = 0, high = levels - 1;
    while (low < high) {
        int mid = (low + high) / 2;
        float mid_val = (centroids[mid] + centroids[mid+1]) * 0.5f;
        if (val < mid_val) high = mid;
        else low = mid + 1;
    }
    return low;
}

inline int quantize_vector3_idx(float v0, float v1, float v2, float norm, const float* centroids, int n_clusters) {
    int best = 0;
    float min_d = 1e30f;
    float iv0 = v0 / norm;
    float iv1 = v1 / norm;
    float iv2 = v2 / norm;
    for (int i = 0; i < n_clusters; ++i) {
        float d0 = iv0 - centroids[i * 3 + 0];
        float d1 = iv1 - centroids[i * 3 + 1];
        float d2 = iv2 - centroids[i * 3 + 2];
        float d = d0 * d0 + d1 * d1 + d2 * d2;
        if (d < min_d) { min_d = d; best = i; }
    }
    return best;
}

/* ── ROTOR COMPRESS KERNEL ──────────────────────────────────────────── */

template <typename T>
torch::Tensor rotor_compress_impl(torch::Tensor input, torch::Tensor rotors, 
                                 torch::Tensor c_scalar, int n_scalar, 
                                 torch::Tensor c_vector, int n_vector, 
                                 torch::Tensor c_bivector, int n_bivector, 
                                 torch::Tensor c_trivector, int n_trivector) {
    int emb_dim = input.size(-1);
    int batch_size = input.numel() / emb_dim;
    int n_groups = (emb_dim + 2) / 3;
    
    // Output: indices (uint8) + norms (float)
    // We'll pack 8 components per group into a flat uint8 tensor
    auto indices = torch::empty({batch_size, n_groups, 8}, input.options().dtype(torch::kUInt8));
    auto norms = torch::empty({batch_size, n_groups}, input.options().dtype(torch::kFloat32));
    
    int threads = std::min(256, std::max(n_groups, WARP_SIZE));
    c10::xpu::getCurrentXPUStream().queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sh_rotors(sycl::range<1>(n_groups * 4), cgh);
        sycl::local_accessor<float, 1> sh_c_scalar(sycl::range<1>(n_scalar), cgh);
        sycl::local_accessor<float, 1> sh_c_vector(sycl::range<1>(n_vector), cgh);
        sycl::local_accessor<float, 1> sh_c_bivector(sycl::range<1>(n_bivector), cgh);
        sycl::local_accessor<float, 1> sh_c_trivector(sycl::range<1>(n_trivector), cgh);
        auto in_p = input.data_ptr<T>();
        auto r_p = rotors.data_ptr<float>();
        auto cs_p = c_scalar.data_ptr<float>();
        auto cv_p = c_vector.data_ptr<float>();
        auto cb_p = c_bivector.data_ptr<float>();
        auto ct_p = c_trivector.data_ptr<float>();
        auto out_idx = indices.data_ptr<uint8_t>();
        auto out_norms = norms.data_ptr<float>();

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, batch_size) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)), [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
            int bid = item.get_group(2), tid = item.get_local_id(2), threads_l = item.get_local_range(2);
            for (int i = tid; i < n_groups * 4; i += threads_l) sh_rotors[i] = r_p[i];
            for (int i = tid; i < n_scalar; i += threads_l) sh_c_scalar[i] = cs_p[i];
            for (int i = tid; i < n_vector; i += threads_l) sh_c_vector[i] = cv_p[i];
            for (int i = tid; i < n_bivector; i += threads_l) sh_c_bivector[i] = cb_p[i];
            for (int i = tid; i < n_trivector; i += threads_l) sh_c_trivector[i] = ct_p[i];
            item.barrier(sycl::access::fence_space::local_space);

            const T* in_ptr = in_p + bid * emb_dim;
            uint8_t* idx_ptr = out_idx + bid * n_groups * 8;
            float* norm_ptr = out_norms + bid * n_groups;

            for (int g = tid; g < n_groups; g += threads_l) {
                float s = sh_rotors[g * 4 + 0], p12 = sh_rotors[g * 4 + 1], p13 = sh_rotors[g * 4 + 2], p23 = sh_rotors[g * 4 + 3];
                float x_mv[8] = {0,0,0,0,0,0,0,0};
                int d0 = g * 3;
                if (d0     < emb_dim) x_mv[1] = convert_to_float<T>(in_ptr[d0]);
                if (d0 + 1 < emb_dim) x_mv[2] = convert_to_float<T>(in_ptr[d0 + 1]);
                if (d0 + 2 < emb_dim) x_mv[3] = convert_to_float<T>(in_ptr[d0 + 2]);
                float t1[8], fwd[8];
                gp_rotor_mv(s, p12, p13, p23, x_mv, t1);
                gp_mv_rotor(t1, s, -p12, -p13, -p23, fwd);
                
                float local_norm_v = sycl::fmax(sycl::fabs(fwd[1]), sycl::fmax(sycl::fabs(fwd[2]), sycl::fabs(fwd[3])));
                if (local_norm_v < 1e-9f) local_norm_v = 1.0f;
                norm_ptr[g] = local_norm_v;

                idx_ptr[g * 8 + 0] = (uint8_t)quantize_scalar_idx(fwd[0], &sh_c_scalar[0], n_scalar);
                idx_ptr[g * 8 + 1] = (uint8_t)quantize_vector3_idx(fwd[1], fwd[2], fwd[3], local_norm_v, &sh_c_vector[0], n_vector / 3);
                idx_ptr[g * 8 + 2] = 0; // reserved
                idx_ptr[g * 8 + 3] = 0; // reserved
                idx_ptr[g * 8 + 4] = (uint8_t)quantize_scalar_idx(fwd[4], &sh_c_bivector[0], n_bivector);
                idx_ptr[g * 8 + 5] = (uint8_t)quantize_scalar_idx(fwd[5], &sh_c_bivector[0], n_bivector);
                idx_ptr[g * 8 + 6] = (uint8_t)quantize_scalar_idx(fwd[6], &sh_c_bivector[0], n_bivector);
                idx_ptr[g * 8 + 7] = (uint8_t)quantize_scalar_idx(fwd[7], &sh_c_trivector[0], n_trivector);
            }
        });
    });
    return torch::cat({indices.view({batch_size, -1}), norms.view({batch_size, -1})}, -1);
}

/* ── ROTOR DECOMPRESS KERNEL ────────────────────────────────────────── */
// (Skipping for brevity, we can just use full_fused for roundtrip testing)

/* ── FULL FUSED (ROUNDTRIP) ─────────────────────────────────────────── */

template <typename T>
torch::Tensor rotor_full_fused_impl(torch::Tensor input, torch::Tensor rotors, 
                                 torch::Tensor c_scalar, int n_scalar, 
                                 torch::Tensor c_vector, int n_vector, 
                                 torch::Tensor c_bivector, int n_bivector, 
                                 torch::Tensor c_trivector, int n_trivector) {
    int emb_dim = input.size(-1);
    int batch_size = input.numel() / emb_dim;
    int n_groups = (emb_dim + 2) / 3;
    auto output = torch::empty_like(input);
    int threads = std::min(256, std::max(n_groups, WARP_SIZE));
    c10::xpu::getCurrentXPUStream().queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sh_rotors(sycl::range<1>(n_groups * 4), cgh);
        sycl::local_accessor<float, 1> sh_c_scalar(sycl::range<1>(n_scalar), cgh);
        sycl::local_accessor<float, 1> sh_c_vector(sycl::range<1>(n_vector), cgh);
        sycl::local_accessor<float, 1> sh_c_bivector(sycl::range<1>(n_bivector), cgh);
        sycl::local_accessor<float, 1> sh_c_trivector(sycl::range<1>(n_trivector), cgh);
        auto in_p = input.data_ptr<T>();
        auto r_p = rotors.data_ptr<float>();
        auto cs_p = c_scalar.data_ptr<float>();
        auto cv_p = c_vector.data_ptr<float>();
        auto cb_p = c_bivector.data_ptr<float>();
        auto ct_p = c_trivector.data_ptr<float>();
        auto out_p = output.data_ptr<T>();
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, batch_size) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)), [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
            int bid = item.get_group(2), tid = item.get_local_id(2), threads_l = item.get_local_range(2);
            for (int i = tid; i < n_groups * 4; i += threads_l) sh_rotors[i] = r_p[i];
            for (int i = tid; i < n_scalar; i += threads_l) sh_c_scalar[i] = cs_p[i];
            for (int i = tid; i < n_vector; i += threads_l) sh_c_vector[i] = cv_p[i];
            for (int i = tid; i < n_bivector; i += threads_l) sh_c_bivector[i] = cb_p[i];
            for (int i = tid; i < n_trivector; i += threads_l) sh_c_trivector[i] = ct_p[i];
            item.barrier(sycl::access::fence_space::local_space);
            const T* in_ptr = in_p + bid * emb_dim;
            T* out_ptr = out_p + bid * emb_dim;
            for (int g = tid; g < n_groups; g += threads_l) {
                float s = sh_rotors[g * 4 + 0], p12 = sh_rotors[g * 4 + 1], p13 = sh_rotors[g * 4 + 2], p23 = sh_rotors[g * 4 + 3];
                float x_mv[8] = {0,0,0,0,0,0,0,0};
                int d0 = g * 3;
                if (d0     < emb_dim) x_mv[1] = convert_to_float<T>(in_ptr[d0]);
                if (d0 + 1 < emb_dim) x_mv[2] = convert_to_float<T>(in_ptr[d0 + 1]);
                if (d0 + 2 < emb_dim) x_mv[3] = convert_to_float<T>(in_ptr[d0 + 2]);
                float t1[8], fwd[8];
                gp_rotor_mv(s, p12, p13, p23, x_mv, t1);
                gp_mv_rotor(t1, s, -p12, -p13, -p23, fwd);
                
                float local_norm_v = sycl::fmax(sycl::fabs(fwd[1]), sycl::fmax(sycl::fabs(fwd[2]), sycl::fabs(fwd[3])));
                if (local_norm_v < 1e-9f) local_norm_v = 1.0f;

                float q_mv[8];
                q_mv[0] = quantize_scalar_bs(fwd[0], &sh_c_scalar[0], n_scalar);
                
                int best_idx = 0;
                float min_d = 1e30f;
                float iv0 = fwd[1] / local_norm_v, iv1 = fwd[2] / local_norm_v, iv2 = fwd[3] / local_norm_v;
                for (int i = 0; i < n_vector / 3; ++i) {
                    float d0 = iv0 - sh_c_vector[i * 3 + 0];
                    float d1 = iv1 - sh_c_vector[i * 3 + 1];
                    float d2 = iv2 - sh_c_vector[i * 3 + 2];
                    float d = d0 * d0 + d1 * d1 + d2 * d2;
                    if (d < min_d) { min_d = d; best_idx = i; }
                }
                q_mv[1] = sh_c_vector[best_idx * 3 + 0] * local_norm_v;
                q_mv[2] = sh_c_vector[best_idx * 3 + 1] * local_norm_v;
                q_mv[3] = sh_c_vector[best_idx * 3 + 2] * local_norm_v;
                
                q_mv[4] = quantize_scalar_bs(fwd[4], &sh_c_bivector[0], n_bivector);
                q_mv[5] = quantize_scalar_bs(fwd[5], &sh_c_bivector[0], n_bivector);
                q_mv[6] = quantize_scalar_bs(fwd[6], &sh_c_bivector[0], n_bivector);
                q_mv[7] = quantize_scalar_bs(fwd[7], &sh_c_trivector[0], n_trivector);

                float t2[8], inv[8];
                gp_rotor_mv(s, -p12, -p13, -p23, q_mv, t2);
                gp_mv_rotor(t2, s, p12, p13, p23, inv);
                if (d0     < emb_dim) out_ptr[d0]     = convert_from_float<T>(inv[1]);
                if (d0 + 1 < emb_dim) out_ptr[d0 + 1] = convert_from_float<T>(inv[2]);
                if (d0 + 2 < emb_dim) out_ptr[d0 + 2] = convert_from_float<T>(inv[3]);
            }
        });
    });
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rotor_full_fused_half", &rotor_full_fused_impl<c10::Half>);
    m.def("rotor_full_fused_bf16", &rotor_full_fused_impl<at::BFloat16>);
    m.def("rotor_compress_half", &rotor_compress_impl<c10::Half>);
    m.def("rotor_compress_bf16", &rotor_compress_impl<at::BFloat16>);
}
