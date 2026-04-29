#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <c10/xpu/XPUStream.h>
#include <cmath>

template <typename T> inline float convert_to_float(T value) { return (float)value; }
template <> inline float convert_to_float<c10::Half>(c10::Half value) { return (float)value; }
template <> inline float convert_to_float<at::BFloat16>(at::BFloat16 value) { return (float)value; }

template <typename T> inline T convert_from_float(float value) { return (T)value; }
template <> inline c10::Half convert_from_float<c10::Half>(float value) { return (c10::Half)value; }
template <> inline at::BFloat16 convert_from_float<at::BFloat16>(float value) { return (at::BFloat16)value; }

#define WARP_SIZE 32

/* ── QUATERNION MATH (ISOQUANT) ──────────────────────────────────────── */

inline void quat_mul(const float a[4], const float b[4], float r[4]) {
    r[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
    r[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
    r[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
    r[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
}

inline void quat_conj(const float q[4], float r[4]) {
    r[0] = q[0]; r[1] = -q[1]; r[2] = -q[2]; r[3] = -q[3];
}

/* ── PLANAR MATH (GIVENS) ─────────────────────────────────────────── */

inline void rot2_apply(float c, float s, float v0, float v1, float &r0, float &r1) {
    r0 = c * v0 - s * v1;
    r1 = s * v0 + c * v1;
}

inline void rot2_inv(float c, float s, float v0, float v1, float &r0, float &r1) {
    r0 = c * v0 + s * v1;
    r1 = -s * v0 + c * v1;
}

/* ── QUANTIZATION UTILS ─────────────────────────────────────────────── */

inline float quantize_bs(float val, const float* centroids, int levels) {
    int low = 0, high = levels - 1;
    while (low < high) {
        int mid = (low + high) / 2;
        float mid_val = (centroids[mid] + centroids[mid+1]) * 0.5f;
        if (val < mid_val) high = mid;
        else low = mid + 1;
    }
    return centroids[low];
}

/* ── ISOQUANT FUSED KERNEL (4D) ─────────────────────────────────────── */

template <typename T>
torch::Tensor isoquant_fused_impl(torch::Tensor input, torch::Tensor qL, torch::Tensor centroids) {
    int emb_dim = input.size(-1);
    int batch_size = input.numel() / emb_dim;
    int n_groups = (emb_dim + 3) / 4;
    int n_levels = centroids.size(0);
    auto output = torch::empty_like(input);
    
    int threads = std::min(256, std::max(n_groups, WARP_SIZE));
    c10::xpu::getCurrentXPUStream().queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sh_qL(sycl::range<1>(n_groups * 4), cgh);
        sycl::local_accessor<float, 1> sh_cb(sycl::range<1>(n_levels), cgh);
        auto in_p = input.data_ptr<T>();
        auto qL_p = qL.data_ptr<float>();
        auto cb_p = centroids.data_ptr<float>();
        auto out_p = output.data_ptr<T>();

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, batch_size) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)), [=](sycl::nd_item<3> item) {
            int bid = item.get_group(2), tid = item.get_local_id(2), threads_l = item.get_local_range(2);
            for (int i = tid; i < n_groups * 4; i += threads_l) sh_qL[i] = qL_p[i];
            for (int i = tid; i < n_levels; i += threads_l) sh_cb[i] = cb_p[i];
            item.barrier(sycl::access::fence_space::local_space);

            const T* in_ptr = in_p + bid * emb_dim;
            T* out_ptr = out_p + bid * emb_dim;

            for (int g = tid; g < n_groups; g += threads_l) {
                float v[4] = {0,0,0,0};
                int d0 = g * 4;
                for (int i=0; i<4; ++i) if (d0+i < emb_dim) v[i] = convert_to_float<T>(in_ptr[d0+i]);
                
                float norm = 0;
                for (int i=0; i<4; ++i) norm += v[i]*v[i];
                norm = std::sqrt(norm);
                if (norm < 1e-8f) norm = 1.0f;
                for (int i=0; i<4; ++i) v[i] /= norm;

                float q_l[4] = {sh_qL[g*4+0], sh_qL[g*4+1], sh_qL[g*4+2], sh_qL[g*4+3]};
                float v_rot[4];
                quat_mul(q_l, v, v_rot);

                float v_q[4];
                for (int i=0; i<4; ++i) v_q[i] = quantize_bs(v_rot[i], &sh_cb[0], n_levels);

                float q_l_conj[4];
                quat_conj(q_l, q_l_conj);
                float v_unrot[4];
                quat_mul(q_l_conj, v_q, v_unrot);

                for (int i=0; i<4; ++i) if (d0+i < emb_dim) out_ptr[d0+i] = convert_from_float<T>(v_unrot[i] * norm);
            }
        });
    });
    return output;
}

/* ── PLANARQUANT FUSED KERNEL (2D) ────────────────────────────────────── */

template <typename T>
torch::Tensor planar_fused_impl(torch::Tensor input, torch::Tensor cs, torch::Tensor centroids) {
    int emb_dim = input.size(-1);
    int batch_size = input.numel() / emb_dim;
    int n_groups = (emb_dim + 1) / 2;
    int n_levels = centroids.size(0);
    auto output = torch::empty_like(input);
    
    int threads = std::min(256, std::max(n_groups, WARP_SIZE));
    c10::xpu::getCurrentXPUStream().queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sh_cs(sycl::range<1>(n_groups * 2), cgh);
        sycl::local_accessor<float, 1> sh_cb(sycl::range<1>(n_levels), cgh);
        auto in_p = input.data_ptr<T>();
        auto cs_p = cs.data_ptr<float>();
        auto cb_p = centroids.data_ptr<float>();
        auto out_p = output.data_ptr<T>();

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, batch_size) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)), [=](sycl::nd_item<3> item) {
            int bid = item.get_group(2), tid = item.get_local_id(2), threads_l = item.get_local_range(2);
            for (int i = tid; i < n_groups * 2; i += threads_l) sh_cs[i] = cs_p[i];
            for (int i = tid; i < n_levels; i += threads_l) sh_cb[i] = cb_p[i];
            item.barrier(sycl::access::fence_space::local_space);

            const T* in_ptr = in_p + bid * emb_dim;
            T* out_ptr = out_p + bid * emb_dim;

            for (int g = tid; g < n_groups; g += threads_l) {
                float v0 = 0, v1 = 0;
                int d0 = g * 2;
                if (d0 < emb_dim) v0 = convert_to_float<T>(in_ptr[d0]);
                if (d0+1 < emb_dim) v1 = convert_to_float<T>(in_ptr[d0+1]);
                
                float norm = std::sqrt(v0*v0 + v1*v1);
                if (norm < 1e-8f) norm = 1.0f;
                v0 /= norm; v1 /= norm;

                float c = sh_cs[g*2+0], s = sh_cs[g*2+1];
                float vr0, vr1;
                rot2_apply(c, s, v0, v1, vr0, vr1);

                float vq0 = quantize_bs(vr0, &sh_cb[0], n_levels);
                float vq1 = quantize_bs(vr1, &sh_cb[0], n_levels);

                float vu0, vu1;
                rot2_inv(c, s, vq0, vq1, vu0, vu1);

                if (d0 < emb_dim) out_ptr[d0] = convert_from_float<T>(vu0 * norm);
                if (d0+1 < emb_dim) out_ptr[d0+1] = convert_from_float<T>(vu1 * norm);
            }
        });
    });
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("isoquant_fused_half", &isoquant_fused_impl<c10::Half>);
    m.def("isoquant_fused_bf16", &isoquant_fused_impl<at::BFloat16>);
    m.def("planar_fused_half", &planar_fused_impl<c10::Half>);
    m.def("planar_fused_bf16", &planar_fused_impl<at::BFloat16>);
}
