#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <c10/xpu/XPUStream.h>
#include <cmath>

template <typename T> inline float convert_to_float(T value) { return (float)value; }
template <> inline float convert_to_float<c10::Half>(c10::Half value) { return (float)value; }
template <> inline float convert_to_float<float>(float value) { return value; }
template <> inline float convert_to_float<at::BFloat16>(at::BFloat16 value) { return (float)value; }

template <typename T> inline T convert_from_float(float value) { return (T)value; }
template <> inline c10::Half convert_from_float<c10::Half>(float value) { return (c10::Half)value; }
template <> inline float convert_from_float<float>(float value) { return value; }
template <> inline at::BFloat16 convert_from_float<at::BFloat16>(float value) { return (at::BFloat16)value; }

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 32
#define EMB_DIM 128

template <typename T, typename Tproj>
[[intel::reqd_sub_group_size(32)]]
void quantize_with_outliers_kernel(
    T *key_states, uint8_t *key_quant, uint8_t *key_outlier_quant,
    const uint8_t *outlier_indices, const Tproj *rand_prj, T *outlier_norms,
    int batch_size, int head_size, int n_size, int group_size, int sketch_dim,
    int outlier_sketch_dim, int emb_dim, int outlier_counts,
    uint8_t *shared_mask, float *shared_keys,
    float *shared_outlier_norms,
    uint8_t *shared_key_quant,
    uint8_t *shared_key_outlier_quant) {

    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sg = item_ct1.get_sub_group();
    
    // Original CUDA dims: grid(batch*head*n, blocksPerGroup, sketch_dim/32), block(32, 32)
    size_t bhn = item_ct1.get_group(0);
    size_t gIdx_block = item_ct1.get_group(1);
    size_t pIdx_block = item_ct1.get_group(2);
    
    size_t threadLane = sg.get_local_id()[0]; // 0..31 (lane in warp)
    size_t wIdx = item_ct1.get_local_id(1);    // 0..31 (warp in block)
    
    size_t gIdx = gIdx_block * 32 + threadLane;
    size_t pIdx = pIdx_block * 32 + wIdx;

    int hash_dim = sketch_dim/8;
    int outlier_hash_dim = outlier_sketch_dim/8;

    int base_index_key_quant = (bhn * group_size * hash_dim) + (gIdx * hash_dim);
    int base_index_outlier_quant = (bhn * group_size * outlier_hash_dim) + (gIdx * outlier_hash_dim);

    int base_index_outlier_indices = bhn * outlier_counts;
    const uint8_t* outlier_ind = outlier_indices + base_index_outlier_indices;

    int base_index_key = (bhn * group_size * emb_dim) + (gIdx_block * 32 * emb_dim);
    T* key_base = key_states + base_index_key;

    size_t base_index_rand_prj = (size_t)pIdx * emb_dim;
    const Tproj* sketch = rand_prj + base_index_rand_prj;

    size_t base_index_outlier_norm = (size_t)(bhn * group_size) + gIdx;
    T* key_outlier_norm = outlier_norms + base_index_outlier_norm;

    size_t tIdx = wIdx * 32 + threadLane;
    if (tIdx < (size_t)emb_dim) {
        shared_mask[tIdx] = 0;
    }
    if (wIdx == 0) {
        shared_outlier_norms[threadLane] = 0.0f;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (tIdx < (size_t)outlier_counts) {
        shared_mask[outlier_ind[tIdx]] = 1;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    for (size_t chnl_idx{wIdx}; chnl_idx < (size_t)emb_dim; chnl_idx += 32) {
        shared_keys[chnl_idx * 32 + threadLane] = convert_to_float<T>(key_base[threadLane * (size_t)emb_dim + chnl_idx]);
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    float sketched_keys = 0.0f;
    float sketched_outliers = 0.0f;

    for (size_t chnl_idx{0}; chnl_idx < (size_t)emb_dim; chnl_idx++) {
        float key_val = shared_keys[chnl_idx * 32 + threadLane];
        float sketch_val = convert_to_float<Tproj>(sketch[chnl_idx]);
        if (shared_mask[chnl_idx] == 0) {
            sketched_keys += key_val * sketch_val;
        } else {
            sketched_outliers += key_val * sketch_val;
        }
    }

    shared_key_quant[threadLane * 32 + wIdx] = (1 << (wIdx % 8));
    shared_key_outlier_quant[threadLane * 32 + wIdx] = (1 << (wIdx % 8));
    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (gIdx >= group_size) return;

    if ((wIdx % 8) == 0) {
        uint8_t hashed_key = 0;
#pragma unroll
        for (int shift = 0; shift < 8; shift++) {
            hashed_key += shared_key_quant[threadLane * 32 + (wIdx + shift)];
        }
        key_quant[base_index_key_quant + pIdx / 8] = hashed_key;

        if (pIdx < (size_t)outlier_sketch_dim) {
            uint8_t hashed_outlier = 0;
#pragma unroll
            for (int shift = 0; shift < 8; shift++) {
                hashed_outlier += shared_key_outlier_quant[threadLane * 32 + (wIdx + shift)];
            }
            key_outlier_quant[base_index_outlier_quant + pIdx / 8] = hashed_outlier;
        }
    }
 
    
    if (wIdx == 0 && pIdx_block == 0) {
        float outlier_norm_sum = 0.0;
        for (size_t chnl_idx{0}; chnl_idx < (size_t)emb_dim; chnl_idx++) {
            if (shared_mask[chnl_idx] != 0) {
                float val = shared_keys[chnl_idx * 32 + threadLane];
                outlier_norm_sum += val * val;
            }
        }
        key_outlier_norm[0] = convert_from_float<T>(sycl::sqrt(outlier_norm_sum));
    }
}

template <typename T, typename Tproj>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> QJLQuantCudaTemplate(
    torch::Tensor key_states,
    torch::Tensor outlier_indices,
    torch::Tensor rand_prj,
    int outlier_sketch_dim) {

    int batch = key_states.size(0);
    int head = key_states.size(1);
    int n = key_states.size(2);
    int group_size = key_states.size(3);
    int emb_dim = key_states.size(4);
    int sketch_dim = rand_prj.size(0);
    int hash_dim = sketch_dim / 8;
    int outlier_hash_dim = outlier_sketch_dim / 8;
    int outlier_counts = outlier_indices.size(-1);

    auto options = torch::TensorOptions().device(torch::kXPU, 0).dtype(torch::kUInt8);
    auto key_quant = torch::zeros({batch, head, n, group_size, hash_dim}, options).contiguous();
    auto key_outlier_quant = torch::zeros({batch, head, n, group_size, outlier_hash_dim}, options).contiguous();
    auto outlier_norms = torch::zeros({batch, head, n, group_size}, key_states.options()).contiguous();

    int blocksPerGroup = (group_size + 31) / 32;
    int numProjBlocks = sketch_dim / 32;
    sycl::range<3> numBlocks(batch * head * n, blocksPerGroup, numProjBlocks);
    sycl::range<3> threadsPerBlockDim(32, 32, 1);

    auto key_states_ptr = key_states.data_ptr<T>();
    auto outlier_norms_ptr = outlier_norms.data_ptr<T>();
    auto rand_prj_ptr = rand_prj.data_ptr<Tproj>();

    c10::xpu::getCurrentXPUStream().queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> shared_mask_acc(sycl::range<1>(128), cgh);
        sycl::local_accessor<float, 1> shared_keys_acc(sycl::range<1>(128 * 32), cgh);
        sycl::local_accessor<float, 1> shared_outlier_norms_acc(sycl::range<1>(32), cgh);
        sycl::local_accessor<uint8_t, 1> shared_key_quant_acc(sycl::range<1>(32 * 32), cgh);
        sycl::local_accessor<uint8_t, 1> shared_key_outlier_quant_acc(sycl::range<1>(32 * 32), cgh);

        auto k_quant_ptr = key_quant.data_ptr<uint8_t>();
        auto ko_quant_ptr = key_outlier_quant.data_ptr<uint8_t>();
        auto o_indices_ptr = outlier_indices.data_ptr<uint8_t>();

        cgh.parallel_for(sycl::nd_range<3>(numBlocks * threadsPerBlockDim, threadsPerBlockDim), 
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            quantize_with_outliers_kernel(
                key_states_ptr, k_quant_ptr, ko_quant_ptr, o_indices_ptr, rand_prj_ptr, outlier_norms_ptr,
                batch, head, n, group_size, sketch_dim, outlier_sketch_dim, emb_dim, outlier_counts,
                shared_mask_acc.get_pointer(), (float*)shared_keys_acc.get_pointer(), shared_outlier_norms_acc.get_pointer(),
                (uint8_t*)shared_key_quant_acc.get_pointer(), (uint8_t*)shared_key_outlier_quant_acc.get_pointer());
        });
    });

    return std::make_tuple(key_quant, key_outlier_quant, outlier_norms);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qjl_quant_xpu_half_half", &QJLQuantCudaTemplate<c10::Half, c10::Half>);
    m.def("qjl_quant_xpu_half_float", &QJLQuantCudaTemplate<c10::Half, float>);
    m.def("qjl_quant_xpu_float_float", &QJLQuantCudaTemplate<float, float>);
    m.def("qjl_quant_xpu_bf16_bf16", &QJLQuantCudaTemplate<at::BFloat16, at::BFloat16>);
    m.def("qjl_quant_xpu_bf16_float", &QJLQuantCudaTemplate<at::BFloat16, float>);
}
