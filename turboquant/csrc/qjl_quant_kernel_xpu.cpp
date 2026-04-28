#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cmath>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 32
#define EMB_DIM 128

template <typename T> SYCL_EXTERNAL float convert_to_float(T value) {
    // Return 0 by default, indicating misuse if not specialized correctly.
    return 0.0f;
}

template <>
float convert_to_float<c10::Half>(c10::Half value) {
    return __half2float(value);
}

template <>
float convert_to_float<float>(float value) {
    return value;
}

template <>
float convert_to_float<at::BFloat16>(at::BFloat16 value) {
    return static_cast<float>(value);
}

template <typename T> SYCL_EXTERNAL T convert_from_float(float value) {
    // Return 0 by default, indicating misuse if not specialized correctly.
    return static_cast<T>(0);
}

template <>
c10::Half convert_from_float<c10::Half>(float value) {
    return __float2half(value);
}

template <>
float convert_from_float<float>(float value) {
    return value;
}

template <>
at::BFloat16 convert_from_float<at::BFloat16>(float value) {
    return static_cast<at::BFloat16>(value);
}

template <typename T, typename Tproj>
/*
DPCT1110:6: The total declared local variable size in device function
quantize_with_outliers_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void quantize_with_outliers_kernel(
    T *key_states, uint8_t *key_quant, uint8_t *key_outlier_quant,
    const uint8_t *outlier_indices, const Tproj *rand_prj, T *outlier_norms,
    int batch_size, int head_size, int n_size, int group_size, int sketch_dim,
    int outlier_sketch_dim, int emb_dim, int outlier_counts,
    uint8_t *shared_mask, float shared_keys[128 /*EMB_DIM*/][32 /*WARP_SIZE*/],
    float *shared_outlier_norms,
    uint8_t shared_key_quant[32 /*WARP_SIZE*/][32 /*WARPS_PER_BLOCK*/],
    uint8_t shared_key_outlier_quant[32 /*WARP_SIZE*/]
                                    [32 /*WARPS_PER_BLOCK*/]) {

    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    size_t bhn = item_ct1.get_group(2);
    size_t threadLane = item_ct1.get_local_id(2);
    size_t wIdx = item_ct1.get_local_id(1);
    size_t gIdx = item_ct1.get_group(1) * WARP_SIZE;
    size_t pIdx = item_ct1.get_group(0) * WARPS_PER_BLOCK + wIdx;

    int hash_dim = sketch_dim/8;
    int outlier_hash_dim = outlier_sketch_dim/8;

    int base_index_key_quant = (bhn * group_size * hash_dim) + ((gIdx+threadLane) * hash_dim);
    int base_index_outlier_quant = (bhn * group_size * outlier_hash_dim) + ((gIdx+threadLane) * outlier_hash_dim);

    int base_index_outlier_indices = bhn * outlier_counts;
    const uint8_t* outlier_ind = outlier_indices + base_index_outlier_indices;

    int base_index_key = (bhn * group_size * emb_dim) + (gIdx * emb_dim);
    T* key = key_states + base_index_key;

    int base_index_rand_prj = (pIdx * emb_dim);
    const Tproj* sketch = rand_prj + base_index_rand_prj;

    int base_index_outlier_norm = (bhn * group_size) + gIdx;
    T* key_outlier_norm = outlier_norms + base_index_outlier_norm;

    size_t tIdx = wIdx * WARP_SIZE + threadLane;
#pragma unroll
    for (size_t tile_idx{tIdx}; tile_idx < EMB_DIM; tile_idx += (WARP_SIZE * WARPS_PER_BLOCK)) {
        shared_mask[tile_idx] = 0;
    }
    /*
    DPCT1065:26: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (tIdx < outlier_counts){
        size_t otlr_idx = outlier_ind[tIdx];
        shared_mask[otlr_idx] = 1;
    }
    /*
    DPCT1065:27: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

#pragma unroll
    for (size_t grp_tile{wIdx}; grp_tile < WARP_SIZE; grp_tile += WARPS_PER_BLOCK) {
#pragma unroll
        for (size_t chnl_tile{threadLane}; chnl_tile < EMB_DIM; chnl_tile += WARP_SIZE){
            shared_keys[chnl_tile][grp_tile] = convert_to_float<T>(key[grp_tile*EMB_DIM + chnl_tile]);
        }
    }
    /*
    DPCT1065:28: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    float sketched_keys = 0.0;
    float sketched_outliers = 0.0;
#pragma unroll
    for (size_t chnl_idx{0}; chnl_idx < EMB_DIM; chnl_idx++){
        float key_proj_prod = convert_to_float<Tproj>(sketch[chnl_idx]) * shared_keys[chnl_idx][threadLane];
        if (shared_mask[chnl_idx] == 0){
            sketched_keys += key_proj_prod;
        }
        else{
            sketched_outliers += key_proj_prod;
        }
    }

    if (item_ct1.get_group(0) == 0) {
        if (wIdx == 0){
            shared_outlier_norms[threadLane] = 0.0;
        }
        /*
        DPCT1118:7: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:31: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

#pragma unroll
        for (size_t chnl_idx{wIdx}; chnl_idx < EMB_DIM; chnl_idx += WARPS_PER_BLOCK) {
            if (shared_mask[chnl_idx] != 0) {
                dpct::atomic_fetch_add<
                    sycl::access::address_space::generic_space>(
                    &shared_outlier_norms[threadLane],
                    shared_keys[chnl_idx][threadLane] *
                        shared_keys[chnl_idx][threadLane]);
            }
        }
    }
    /*
    DPCT1065:29: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    shared_key_quant[threadLane][wIdx] = (sketched_keys>0 ? (1<<(wIdx%8)) :0);
    shared_key_outlier_quant[threadLane][wIdx] = (sketched_outliers>0 ? (1<<(wIdx%8)) :0);
    /*
    DPCT1065:30: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (gIdx+threadLane >= group_size) return;

    if ((wIdx%8) == 0) {
        uint8_t hashed_key = 0;
#pragma unroll
        for (int shift = 0; shift < 8; shift ++){
            hashed_key += shared_key_quant[threadLane][wIdx+shift];
        }
        key_quant[base_index_key_quant+pIdx/8] = hashed_key;

        if (pIdx >= outlier_sketch_dim) return;

        uint8_t hashed_outlier = 0;
#pragma unroll
        for (int shift = 0; shift < 8; shift ++){
            hashed_outlier += shared_key_outlier_quant[threadLane][wIdx+shift];
        }
        key_outlier_quant[base_index_outlier_quant+pIdx/8] = hashed_outlier;
    } else if ((wIdx == 1) && (item_ct1.get_group(0) == 0)) {
        key_outlier_norm[threadLane] =
            convert_from_float<T>(sycl::sqrt(shared_outlier_norms[threadLane]));
    }
    return;
}


torch::TensorOptions getOptionsForType(const std::type_info& typeInfo) {
    if (typeInfo == typeid(c10::Half)) {
        return torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kHalf);
    } else if (typeInfo == typeid(float)) {
        return torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat);
    } else if (typeInfo == typeid(at::BFloat16)) {
        return torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kBFloat16);
    } else {
        // Default case for unexpected types
        throw std::runtime_error("Unsupported type for tensor options.");
    }
}

template <typename T, typename Tproj>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> QJLQuantCudaTemplate(
    torch::Tensor key_states,
    torch::Tensor outlier_indices,
    torch::Tensor rand_prj,
    int outlier_sketch_dim) {

    auto options = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kUInt8);
    auto options_outlier_norm = getOptionsForType(typeid(T));

    int batch = key_states.size(0);
    int head = key_states.size(1);
    int n = key_states.size(2);
    int group_size = key_states.size(3);
    int emb_dim = key_states.size(4);
    int sketch_dim = rand_prj.size(0);
    int hash_dim = sketch_dim/8;
    int outlier_hash_dim = outlier_sketch_dim/8;
    int outlier_counts = outlier_indices.size(3);

    auto key_quant = torch::zeros({batch, head, n, group_size, hash_dim}, options).contiguous();
    auto key_outlier_quant = torch::zeros({batch, head, n, group_size, outlier_hash_dim}, options).contiguous();
    auto outlier_norms = torch::zeros({batch, head, n, group_size}, options_outlier_norm).contiguous();

    int blocksPerGroup = (group_size + WARP_SIZE - 1) / WARP_SIZE;
    int numProjBlocks = sketch_dim / WARPS_PER_BLOCK;
    dpct::dim3 numBlocks(batch * head * n, blocksPerGroup, numProjBlocks);
    dpct::dim3 threadsPerBlockDim(WARP_SIZE, WARPS_PER_BLOCK, 1);

    auto key_states_ptr = key_states.data_ptr<T>();
    auto outlier_norms_ptr = outlier_norms.data_ptr<T>();
    auto rand_prj_ptr = rand_prj.data_ptr<Tproj>();


//     Compiler hints for using L2 Persistent Cache
    dpct::queue_ptr stream;
    stream = dpct::get_current_device().create_queue(); // Create CUDA stream
    int device_id{0};
    device_id = dpct::get_current_device_id(); // Device ID

    dpct::device_info prop; // CUDA device properties variable
    dpct::get_device(device_id).get_device_info(prop); // Query GPU properties
    size_t size = std::min(1024 * 1024, prop.persistingL2CacheMaxSize);
    /*
    DPCT1026:32: The call to cudaDeviceSetLimit was removed because SYCL
    currently does not support setting resource limits on devices.
    */
; // set-aside 1 Mbytes of L2 cache for persisting accesses or the max allowed

    size_t num_bytes = sketch_dim * emb_dim * sizeof(T);
    size_t window_size =
        std::min(static_cast<size_t>(prop.accessPolicyMaxWindowSize),
                 num_bytes); // Select minimum of user defined num_bytes and max
                             // window size.

    int stream_attribute;        // Stream level attributes data structure
;                            // Global Memory data pointer
indow_size;                  // Number of bytes for persistence access
.0;                          // Hint for cache hit ratio
udaAccessPropertyPersisting; // Persistence Property
udaAccessPropertyStreaming;  // Type of access property on cache miss

    /*
    DPCT1026:33: The call to cudaStreamSetAttribute was removed because SYCL
    currently does not support setting cache config on devices.
    */
; // Set the attributes to a CUDA Stream

    /*
    DPCT1049:8: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    stream->submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:61: 'EMB_DIM' expression was replaced with a value. Modify the
        code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<uint8_t, 1> shared_mask_acc_ct1(
            sycl::range<1>(128 /*EMB_DIM*/), cgh);
        /*
        DPCT1101:62: 'EMB_DIM' expression was replaced with a value. Modify the
        code to use the original expression, provided in comments, if it is
        correct.
        */
        /*
        DPCT1101:63: 'WARP_SIZE' expression was replaced with a value. Modify
        the code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float[128 /*EMB_DIM*/][32 /*WARP_SIZE*/], 0>
            shared_keys_acc_ct1(cgh);
        /*
        DPCT1101:64: 'WARP_SIZE' expression was replaced with a value. Modify
        the code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float, 1> shared_outlier_norms_acc_ct1(
            sycl::range<1>(32 /*WARP_SIZE*/), cgh);
        /*
        DPCT1101:65: 'WARP_SIZE' expression was replaced with a value. Modify
        the code to use the original expression, provided in comments, if it is
        correct.
        */
        /*
        DPCT1101:66: 'WARPS_PER_BLOCK' expression was replaced with a value.
        Modify the code to use the original expression, provided in comments, if
        it is correct.
        */
        sycl::local_accessor<uint8_t[32 /*WARP_SIZE*/][32 /*WARPS_PER_BLOCK*/],
                             0>
            shared_key_quant_acc_ct1(cgh);
        /*
        DPCT1101:67: 'WARP_SIZE' expression was replaced with a value. Modify
        the code to use the original expression, provided in comments, if it is
        correct.
        */
        /*
        DPCT1101:68: 'WARPS_PER_BLOCK' expression was replaced with a value.
        Modify the code to use the original expression, provided in comments, if
        it is correct.
        */
        sycl::local_accessor<uint8_t[32 /*WARP_SIZE*/][32 /*WARPS_PER_BLOCK*/],
                             0>
            shared_key_outlier_quant_acc_ct1(cgh);

        auto key_quant_data_ptr_uint8_t_ct1 = key_quant.data_ptr<uint8_t>();
        auto key_outlier_quant_data_ptr_uint8_t_ct2 =
            key_outlier_quant.data_ptr<uint8_t>();
        auto outlier_indices_data_ptr_uint8_t_ct3 =
            outlier_indices.data_ptr<uint8_t>();

        cgh.parallel_for(
            sycl::nd_range<3>(numBlocks * threadsPerBlockDim,
                              threadsPerBlockDim),
            [=](sycl::nd_item<3> item_ct1) {
                quantize_with_outliers_kernel(
                    key_states_ptr, key_quant_data_ptr_uint8_t_ct1,
                    key_outlier_quant_data_ptr_uint8_t_ct2,
                    outlier_indices_data_ptr_uint8_t_ct3, rand_prj_ptr,
                    outlier_norms_ptr, batch, head, n, group_size, sketch_dim,
                    outlier_sketch_dim, emb_dim, outlier_counts,
                    shared_mask_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    shared_keys_acc_ct1,
                    shared_outlier_norms_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    shared_key_quant_acc_ct1, shared_key_outlier_quant_acc_ct1);
            });
    });

; // Setting the window size to 0 disable it
    /*
    DPCT1026:34: The call to cudaStreamSetAttribute was removed because SYCL
    currently does not support setting cache config on devices.
    */
; // Overwrite the access policy attribute to a CUDA Stream
    /*
    DPCT1026:35: The call to cudaCtxResetPersistingL2Cache was removed because
    SYCL currently does not support setting cache config on devices.
    */
; // Remove any persistent lines in L2

    return std::make_tuple(key_quant, key_outlier_quant, outlier_norms);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qjl_quant_half_half", &QJLQuantCudaTemplate<c10::Half, c10::Half>, "Quantize using Half precision",
    py::arg("key_states"),
    py::arg("outlier_indices"),
    py::arg("rand_prj"),
    py::arg("outlier_sketch_dim"));

    m.def("qjl_quant_half_float", &QJLQuantCudaTemplate<c10::Half, float>, "Quantize using Half to Float precision",
    py::arg("key_states"),
    py::arg("outlier_indices"),
    py::arg("rand_prj"),
    py::arg("outlier_sketch_dim"));

    m.def("qjl_quant_float_float", &QJLQuantCudaTemplate<float, float>, "Quantize using Float precision",
    py::arg("key_states"),
    py::arg("outlier_indices"),
    py::arg("rand_prj"),
    py::arg("outlier_sketch_dim"));

    m.def("qjl_quant_bf16_bf16", &QJLQuantCudaTemplate<at::BFloat16, at::BFloat16>, "Quantize using BF16 precision",
    py::arg("key_states"),
    py::arg("outlier_indices"),
    py::arg("rand_prj"),
    py::arg("outlier_sketch_dim"));

    m.def("qjl_quant_bf16_float", &QJLQuantCudaTemplate<at::BFloat16, float>, "Quantize using BF16 to Float precision",
    py::arg("key_states"),
    py::arg("outlier_indices"),
    py::arg("rand_prj"),
    py::arg("outlier_sketch_dim"));
}
