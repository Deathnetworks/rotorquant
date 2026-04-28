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

inline void geometric_product_f32(const float a[8], const float b[8], float r[8]) {
    // S=0, E1=1, E2=2, E3=3, E12=4, E13=5, E23=6, E123=7
    float a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a12 = a[4], a13 = a[5], a23 = a[6], a123 = a[7];
    float b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3], b12 = b[4], b13 = b[5], b23 = b[6], b123 = b[7];

    r[0] = a0*b0 + a1*b1 + a2*b2 + a3*b3 - a12*b12 - a13*b13 - a23*b23 - a123*b123;
    r[1] = a0*b1 + a1*b0 - a2*b12 + a12*b2 - a3*b13 + a13*b3 + a23*b123 + a123*b23;
    r[2] = a0*b2 + a2*b0 + a1*b12 - a12*b1 - a3*b23 + a23*b3 - a13*b123 - a123*b13;
    r[3] = a0*b3 + a3*b0 + a1*b13 - a13*b1 + a2*b23 - a23*b2 + a12*b123 + a123*b12;
    r[4] = a0*b12 + a12*b0 + a1*b2 - a2*b1 + a13*b23 - a23*b13 + a3*b123 - a123*b3;
    r[5] = a0*b13 + a13*b0 + a1*b3 - a3*b1 - a12*b23 + a23*b12 - a2*b123 + a123*b2;
    r[6] = a0*b23 + a23*b0 + a2*b3 - a3*b2 + a12*b13 - a13*b12 + a1*b123 - a123*b1;
    r[7] = a0*b123 + a123*b0 + a1*b23 - a23*b1 - a2*b13 + a13*b2 + a3*b12 - a12*b3;
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

    for (int i = tid; i < n_groups * 8; i += threads) sh_rotors[i] = rotors[i];
    for (int i = tid; i < n_scalar; i += threads) sh_c_scalar[i] = c_scalar[i];
    for (int i = tid; i < n_vector; i += threads) sh_c_vector[i] = c_vector[i];
    for (int i = tid; i < n_bivector; i += threads) sh_c_bivector[i] = c_bivector[i];
    for (int i = tid; i < n_trivector; i += threads) sh_c_trivector[i] = c_trivector[i];
    item_ct1.barrier(sycl::access::fence_space::local_space);

    int levels = n_scalar / n_groups;
    int v_levels = n_vector / (n_groups * 3);
    int b_levels = n_bivector / (n_groups * 3);
    int t_levels = n_trivector / n_groups;

    const T* in_ptr = input + bid * emb_dim;
    T* out_ptr = output + bid * emb_dim;

    for (int grp = tid; grp < n_groups; grp += threads) {
        float x_mv[8] = {0,0,0,0,0,0,0,0};
        int d0 = grp * 3;
        if (d0 < emb_dim) x_mv[1] = convert_to_float<T>(in_ptr[d0]);
        if (d0 + 1 < emb_dim) x_mv[2] = convert_to_float<T>(in_ptr[d0 + 1]);
        if (d0 + 2 < emb_dim) x_mv[3] = convert_to_float<T>(in_ptr[d0 + 2]);

        float r_fwd[8], r_rev[8], temp[8], rotated[8];
        for (int i = 0; i < 8; i++) {
            r_fwd[i] = sh_rotors[grp * 8 + i];
            r_rev[i] = (i < 4) ? r_fwd[i] : -r_fwd[i];
        }
        geometric_product_f32(r_fwd, x_mv, temp);
        geometric_product_f32(temp, r_rev, rotated);

        float q_mv[8] = {0.0f};
        q_mv[0] = quantize_scalar(rotated[0], sh_c_scalar + grp*levels, levels);
        q_mv[1] = quantize_scalar(rotated[1], sh_c_vector + grp*3*v_levels, v_levels);
        q_mv[2] = quantize_scalar(rotated[2], sh_c_vector + (grp*3+1)*v_levels, v_levels);
        q_mv[3] = quantize_scalar(rotated[3], sh_c_vector + (grp*3+2)*v_levels, v_levels);
        q_mv[4] = quantize_scalar(rotated[4], sh_c_bivector + grp*3*b_levels, b_levels);
        q_mv[5] = quantize_scalar(rotated[5], sh_c_bivector + (grp*3+1)*b_levels, b_levels);
        q_mv[6] = quantize_scalar(rotated[6], sh_c_bivector + (grp*3+2)*b_levels, b_levels);
        q_mv[7] = quantize_scalar(rotated[7], sh_c_trivector + grp*t_levels, t_levels);

        float temp2[8], final_mv[8];
        geometric_product_f32(r_rev, q_mv, temp2);
        geometric_product_f32(temp2, r_fwd, final_mv);

        if (d0 < emb_dim) out_ptr[d0] = convert_from_float<T>(final_mv[1]);
        if (d0 + 1 < emb_dim) out_ptr[d0 + 1] = convert_from_float<T>(final_mv[2]);
        if (d0 + 2 < emb_dim) out_ptr[d0 + 2] = convert_from_float<T>(final_mv[3]);
    }
}

template <typename T>
[[intel::reqd_sub_group_size(32)]]
void rotor_sandwich_kernel(const T* __restrict__ input, const float* __restrict__ rotors, T* __restrict__ output, int batch_size, int emb_dim, int n_groups, float* sh_rotors) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int bid = item_ct1.get_group(2), tid = item_ct1.get_local_id(2), threads = item_ct1.get_local_range(2);
    for (int i = tid; i < n_groups * 8; i += threads) sh_rotors[i] = rotors[i];
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const T* in_ptr = input + bid * emb_dim;
    T* out_ptr = output + bid * n_groups * 8;
    for (int grp = tid; grp < n_groups; grp += threads) {
        float x_mv[8] = {0,0,0,0,0,0,0,0};
        int d0 = grp * 3;
        if (d0 < emb_dim) x_mv[1] = convert_to_float<T>(in_ptr[d0]);
        if (d0 + 1 < emb_dim) x_mv[2] = convert_to_float<T>(in_ptr[d0 + 1]);
        if (d0 + 2 < emb_dim) x_mv[3] = convert_to_float<T>(in_ptr[d0 + 2]);
        float r_fwd[8], r_rev[8], temp[8], rotated[8];
        for (int i = 0; i < 8; i++) {
            r_fwd[i] = sh_rotors[grp * 8 + i];
            r_rev[i] = (i < 4) ? r_fwd[i] : -r_fwd[i];
        }
        geometric_product_f32(r_fwd, x_mv, temp);
        geometric_product_f32(temp, r_rev, rotated);
        for (int i = 0; i < 8; i++) out_ptr[grp * 8 + i] = convert_from_float<T>(rotated[i]);
    }
}

template <typename T>
[[intel::reqd_sub_group_size(32)]]
void rotor_inverse_sandwich_kernel(const T* __restrict__ input, const float* __restrict__ rotors, T* __restrict__ output, int batch_size, int emb_dim, int n_groups, float* sh_rotors) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int bid = item_ct1.get_group(2), tid = item_ct1.get_local_id(2), threads = item_ct1.get_local_range(2);
    for (int i = tid; i < n_groups * 8; i += threads) sh_rotors[i] = rotors[i];
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const T* in_ptr = input + bid * n_groups * 8;
    T* out_ptr = output + bid * emb_dim;
    for (int grp = tid; grp < n_groups; grp += threads) {
        float x_mv[8];
        for (int i = 0; i < 8; i++) x_mv[i] = convert_to_float<T>(in_ptr[grp * 8 + i]);
        float r_fwd[8], r_rev[8], temp[8], rotated[8];
        for (int i = 0; i < 8; i++) {
            r_fwd[i] = sh_rotors[grp * 8 + i];
            r_rev[i] = (i < 4) ? r_fwd[i] : -r_fwd[i];
        }
        // Inverse sandwich is R_rev x R
        geometric_product_f32(r_rev, x_mv, temp);
        geometric_product_f32(temp, r_fwd, rotated);
        int d0 = grp * 3;
        if (d0 < emb_dim) out_ptr[d0] = convert_from_float<T>(rotated[1]);
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
        sycl::local_accessor<float, 1> sh_rotors(sycl::range<1>(n_groups * 8), cgh), sh_c_scalar(sycl::range<1>(n_scalar), cgh), sh_c_vector(sycl::range<1>(n_vector), cgh), sh_c_bivector(sycl::range<1>(n_bivector), cgh), sh_c_trivector(sycl::range<1>(n_trivector), cgh);
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
        sycl::local_accessor<float, 1> sh_rotors(sycl::range<1>(n_groups * 8), cgh);
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
        sycl::local_accessor<float, 1> sh_rotors(sycl::range<1>(n_groups * 8), cgh);
        auto in_p = input.data_ptr<T>();
        auto r_p = rotors.data_ptr<float>();
        auto out_p = output.data_ptr<T>();
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, batch_size) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)), [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
            rotor_inverse_sandwich_kernel<T>(in_p, r_p, out_p, batch_size, emb_dim, n_groups, sh_rotors.get_pointer());
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
}
