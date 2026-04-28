#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cmath>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define EMB_DIM 128
#define GQA_GROUP_SIZE 4
#define FULL_MASK 0xffffffff

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

SYCL_EXTERNAL __inline__ float warpReduceSum(float val) {
  for (int offset = sycl::ext::oneapi::this_work_item::get_sub_group()
                        .get_local_range()
                        .get(0) /
                    2;
       offset > 0; offset >>= 1)
    /*
    DPCT1121:15: Make sure that the "val" which is used in the SYCL group
    function/algorithm is initialized.
    */
    val += dpct::shift_sub_group_left(
        sycl::ext::oneapi::this_work_item::get_sub_group(), val, offset);
  return val;
}

template <typename T, typename Tproj>
/*
DPCT1110:16: The total declared local variable size in device function
calc_gqa_score_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void calc_gqa_score_kernel(
    T *query_states, const uint8_t *key_quant, const uint8_t *key_outlier_quant,
    T *key_norm, T *key_outlier_norm, const uint8_t *outlier_indices,
    const float *query_sketch, const Tproj *rand_prj, float *scores,
    int batch_size, int key_head_size, int n_size, int group_size,
    int gqa_group_size, int sketch_dim, int outlier_sketch_dim, int emb_dim,
    int outlier_counts,
    float shared_query[4 /*GQA_GROUP_SIZE*/][128 /*EMB_DIM*/],
    uint8_t *shared_outlier_ind,
    float shared_innprod[4 /*GQA_GROUP_SIZE*/][32 /*WARP_SIZE*/],
    float shared_outlier_innprod[4 /*GQA_GROUP_SIZE*/][32 /*WARP_SIZE*/],
    float shared_q_sketch[4 /*GQA_GROUP_SIZE*/][32 /*WARP_SIZE*/][8],
    float shared_q_outliers_sketch[4 /*GQA_GROUP_SIZE*/][32 /*WARP_SIZE*/][8]) {

    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    size_t k_bh = item_ct1.get_group(2);
    size_t n = item_ct1.get_group(1);
    size_t q_bh = gqa_group_size * k_bh;
    size_t threadLane = item_ct1.get_local_id(2);
    size_t wIdx = item_ct1.get_local_id(1);
    size_t gIdx = item_ct1.get_group(0) * WARP_SIZE;

    int hash_dim = sketch_dim/8;
    int outlier_hash_dim = outlier_sketch_dim/8;

    int base_index_outlier_indices = (k_bh * n_size * outlier_counts) + (n * outlier_counts);
    const uint8_t* outlier_ind = outlier_indices + base_index_outlier_indices;

    int base_index_query_sketch = (q_bh * sketch_dim);
    const float* q_sketch = query_sketch + base_index_query_sketch;

    int base_index_key_quant = (k_bh * n_size * group_size * hash_dim) + (n * group_size * hash_dim) + (gIdx * hash_dim);
    const uint8_t* k_quant = key_quant + base_index_key_quant;

    int base_index_outlier_quant = (k_bh * n_size * group_size * outlier_hash_dim) + (n * group_size * outlier_hash_dim) + (gIdx * outlier_hash_dim);
    const uint8_t* outlier_quant = key_outlier_quant + base_index_outlier_quant;

    int base_index_key_norm = (k_bh * n_size * group_size) + (n * group_size) + gIdx;
    const T* k_norm = key_norm + base_index_key_norm;
    const T* outlier_norm = key_outlier_norm + base_index_key_norm;

    int base_index_query_states = (q_bh * emb_dim);
    const T* query = query_states + base_index_query_states;

    // load query states into shared memory

    size_t tIdx = wIdx * WARP_SIZE + threadLane;
    for (size_t tile_idx{tIdx}; tile_idx < (gqa_group_size * emb_dim); tile_idx += (WARP_SIZE * WARPS_PER_BLOCK)) {
        size_t h_idx = tile_idx / emb_dim;
        size_t chnl_idx = tile_idx % emb_dim;
        shared_query[h_idx][chnl_idx] = convert_to_float<T>(query[h_idx*emb_dim + chnl_idx]);
    }
    // load outlier indices into shared buffer

    for (size_t tile_idx{tIdx}; tile_idx < outlier_counts; tile_idx += (WARP_SIZE * WARPS_PER_BLOCK)) {
        shared_outlier_ind[tile_idx] = outlier_ind[tile_idx];
    }
    // allocate shared memory to inner products of quantized keys or outliers with query_sketch

    if (wIdx < gqa_group_size) {
        shared_innprod[wIdx][threadLane] = 0.0;
        shared_outlier_innprod[wIdx][threadLane] = 0.0;
    }
    /*
    DPCT1065:42: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // reserve shared memory for a block of query sketch and query outlier sketch

    for (size_t chnl_tile{0}; chnl_tile < sketch_dim; chnl_tile += (8*WARP_SIZE)){
        // load a block of query sketch and compute query outlier sketch
        for (size_t gqa_idx{0}; gqa_idx < gqa_group_size; gqa_idx++){
            for (size_t q_idx{tIdx}; q_idx < (8*WARP_SIZE); q_idx += (WARP_SIZE * WARPS_PER_BLOCK)) {
                if (chnl_tile+q_idx < sketch_dim){
                    shared_q_sketch[gqa_idx][q_idx/8][q_idx%8] = q_sketch[(gqa_idx*sketch_dim) + chnl_tile+q_idx];
                }
                else {
                    shared_q_sketch[gqa_idx][q_idx/8][q_idx%8] = 0.0;
                }
            }
        }
        for (size_t q_idx{tIdx}; q_idx < (8*WARP_SIZE); q_idx += (WARP_SIZE * WARPS_PER_BLOCK)) {
            for (size_t gqa_idx{0}; gqa_idx < gqa_group_size; gqa_idx++) {
                shared_q_outliers_sketch[gqa_idx][q_idx/8][q_idx%8] = 0.0;
            }
            if (chnl_tile+q_idx < sketch_dim){
                for (size_t i{0}; i < outlier_counts; i++){
                    int otlr_idx = shared_outlier_ind[i];
                    float rand_prj_buffer = convert_to_float<Tproj>(rand_prj[(otlr_idx * sketch_dim) + chnl_tile+q_idx]);
                    for (size_t gqa_idx{0}; gqa_idx < gqa_group_size; gqa_idx++) {
                        shared_q_outliers_sketch[gqa_idx][q_idx/8][q_idx%8] += shared_query[gqa_idx][otlr_idx] * rand_prj_buffer; // convert_to_float(const_query[q_bh][otlr_idx])
                    }
                }
            }
        }
        /*
        DPCT1118:17: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:44: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        for (size_t grp_tile{wIdx}; grp_tile < WARP_SIZE; grp_tile += WARPS_PER_BLOCK) {
            // load key quant and outlier quant
            uint8_t key_quant_buffer = k_quant[grp_tile*hash_dim + chnl_tile/8 + threadLane];
            uint8_t outlier_quant_buffer = 0;
            if (chnl_tile + 8*threadLane < outlier_sketch_dim){
                outlier_quant_buffer = outlier_quant[grp_tile*outlier_hash_dim + chnl_tile/8 + threadLane];
            }
            /*
            DPCT1118:19: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            /*
            DPCT1065:46: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            for (size_t gqa_idx{0}; gqa_idx < gqa_group_size; gqa_idx++) {
                float k_inner_prod = 0.0;
                float outlier_inner_prod = 0.0;
                for (int shift = 0; shift < 8; shift++) {
                    float q_sketch_val = shared_q_sketch[gqa_idx][threadLane][shift] - shared_q_outliers_sketch[gqa_idx][threadLane][shift];
                    k_inner_prod += (((key_quant_buffer >> shift)&1) ? q_sketch_val :-q_sketch_val);
                    if (chnl_tile + 8*threadLane < outlier_sketch_dim) {
                        float q_otlr_sketch_val = shared_q_outliers_sketch[gqa_idx][threadLane][shift];
                        outlier_inner_prod += (((outlier_quant_buffer >> shift)&1) ? q_otlr_sketch_val :-q_otlr_sketch_val);
                    }
                }
                /*
                DPCT1118:20: SYCL group functions and algorithms must be
                encountered in converged control flow. You may need to adjust
                the code.
                */
                /*
                DPCT1065:47: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();

                k_inner_prod = warpReduceSum(k_inner_prod);
                outlier_inner_prod = warpReduceSum(outlier_inner_prod);

                if (threadLane == 0) {
                    shared_innprod[gqa_idx][grp_tile] += k_inner_prod;
                    shared_outlier_innprod[gqa_idx][grp_tile] += outlier_inner_prod;
                }
            }
        }
        /*
        DPCT1118:18: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:45: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
    /*
    DPCT1065:43: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (gIdx+threadLane >= group_size) return;
    if (wIdx < gqa_group_size) {
        float scl = sqrtf(M_PI_2) / static_cast<float>(sketch_dim);
        float scl_otlr = sqrtf(M_PI_2) / static_cast<float>(outlier_sketch_dim);
        float norm_otlr = convert_to_float<T>(outlier_norm[threadLane]);
        float norm_k = sqrtf(pow(convert_to_float<T>(k_norm[threadLane]), 2) -
                             norm_otlr * norm_otlr);
        float score = scl * norm_k * shared_innprod[wIdx][threadLane] + scl_otlr * norm_otlr * shared_outlier_innprod[wIdx][threadLane];
        scores[((q_bh + wIdx) * n_size * group_size) + (n * group_size) + gIdx + threadLane] = score;
    }
}


template <typename T, typename Tproj>
torch::Tensor QJLGQAScoreCudaTemplate(
    torch::Tensor key_quant,
    torch::Tensor key_outlier_quant,
    torch::Tensor key_norm,
    torch::Tensor key_outlier_norm,
    torch::Tensor outlier_indices,
    torch::Tensor query_sketch,
    torch::Tensor query_states,
    torch::Tensor rand_prj) {

    auto options = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat);

    int batch = key_quant.size(0);
    int k_head = key_quant.size(1);
    int n = key_quant.size(2);
    int group_size = key_quant.size(3);
    int q_head = query_states.size(1);
    int emb_dim = query_states.size(3);
    int sketch_dim = rand_prj.size(1);
    int outlier_sketch_dim = 8*key_outlier_quant.size(4);
    int outlier_counts = outlier_indices.size(3);
    int gqa_group_size = q_head / k_head;

    auto scores = torch::zeros({batch, q_head, n * group_size, 1}, options).contiguous();
    
    auto query_states_ptr = query_states.data_ptr<T>();
    auto key_norm_ptr = key_norm.data_ptr<T>();
    auto key_outlier_norm_ptr = key_outlier_norm.data_ptr<T>();
    auto rand_prj_ptr = rand_prj.data_ptr<Tproj>();

    int blocksPerGroup = (group_size + WARP_SIZE - 1) / WARP_SIZE;
    dpct::dim3 numBlocks(batch * k_head, n, blocksPerGroup);
    dpct::dim3 threadsPerBlockDim(WARP_SIZE, WARPS_PER_BLOCK, 1);

    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:69: 'GQA_GROUP_SIZE' expression was replaced with a value.
        Modify the code to use the original expression, provided in comments, if
        it is correct.
        */
        /*
        DPCT1101:70: 'EMB_DIM' expression was replaced with a value. Modify the
        code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float[4 /*GQA_GROUP_SIZE*/][128 /*EMB_DIM*/], 0>
            shared_query_acc_ct1(cgh);
        /*
        DPCT1101:71: 'WARP_SIZE' expression was replaced with a value. Modify
        the code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<uint8_t, 1> shared_outlier_ind_acc_ct1(
            sycl::range<1>(32 /*WARP_SIZE*/), cgh);
        /*
        DPCT1101:72: 'GQA_GROUP_SIZE' expression was replaced with a value.
        Modify the code to use the original expression, provided in comments, if
        it is correct.
        */
        /*
        DPCT1101:73: 'WARP_SIZE' expression was replaced with a value. Modify
        the code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float[4 /*GQA_GROUP_SIZE*/][32 /*WARP_SIZE*/], 0>
            shared_innprod_acc_ct1(cgh);
        /*
        DPCT1101:74: 'GQA_GROUP_SIZE' expression was replaced with a value.
        Modify the code to use the original expression, provided in comments, if
        it is correct.
        */
        /*
        DPCT1101:75: 'WARP_SIZE' expression was replaced with a value. Modify
        the code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float[4 /*GQA_GROUP_SIZE*/][32 /*WARP_SIZE*/], 0>
            shared_outlier_innprod_acc_ct1(cgh);
        /*
        DPCT1101:76: 'GQA_GROUP_SIZE' expression was replaced with a value.
        Modify the code to use the original expression, provided in comments, if
        it is correct.
        */
        /*
        DPCT1101:77: 'WARP_SIZE' expression was replaced with a value. Modify
        the code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float[4 /*GQA_GROUP_SIZE*/][32 /*WARP_SIZE*/][8],
                             0>
            shared_q_sketch_acc_ct1(cgh);
        /*
        DPCT1101:78: 'GQA_GROUP_SIZE' expression was replaced with a value.
        Modify the code to use the original expression, provided in comments, if
        it is correct.
        */
        /*
        DPCT1101:79: 'WARP_SIZE' expression was replaced with a value. Modify
        the code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float[4 /*GQA_GROUP_SIZE*/][32 /*WARP_SIZE*/][8],
                             0>
            shared_q_outliers_sketch_acc_ct1(cgh);

        auto key_quant_data_ptr_uint8_t_ct1 = key_quant.data_ptr<uint8_t>();
        auto key_outlier_quant_data_ptr_uint8_t_ct2 =
            key_outlier_quant.data_ptr<uint8_t>();
        auto outlier_indices_data_ptr_uint8_t_ct5 =
            outlier_indices.data_ptr<uint8_t>();
        auto query_sketch_data_ptr_float_ct6 = query_sketch.data_ptr<float>();
        auto scores_data_ptr_float_ct8 = scores.data_ptr<float>();

        cgh.parallel_for(
            sycl::nd_range<3>(numBlocks * threadsPerBlockDim,
                              threadsPerBlockDim),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                calc_gqa_score_kernel(
                    query_states_ptr, key_quant_data_ptr_uint8_t_ct1,
                    key_outlier_quant_data_ptr_uint8_t_ct2, key_norm_ptr,
                    key_outlier_norm_ptr, outlier_indices_data_ptr_uint8_t_ct5,
                    query_sketch_data_ptr_float_ct6, rand_prj_ptr,
                    scores_data_ptr_float_ct8, batch, k_head, n, group_size,
                    gqa_group_size, sketch_dim, outlier_sketch_dim, emb_dim,
                    outlier_counts, shared_query_acc_ct1,
                    shared_outlier_ind_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    shared_innprod_acc_ct1, shared_outlier_innprod_acc_ct1,
                    shared_q_sketch_acc_ct1, shared_q_outliers_sketch_acc_ct1);
            });
    });

    return scores;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qjl_gqa_score_cuda_half_half", &QJLGQAScoreCudaTemplate<c10::Half, c10::Half>, "Cuda kernel to calculate scores fully parallel using Half precision",
          py::arg("key_quant"),
          py::arg("key_outlier_quant"),
          py::arg("key_norm"),
          py::arg("key_outlier_norm"),
          py::arg("outlier_indices"),
          py::arg("query_sketch"),
          py::arg("query_states"),
          py::arg("rand_prj"));

    m.def("qjl_gqa_score_cuda_half_float", &QJLGQAScoreCudaTemplate<c10::Half, float>, "Cuda kernel to calculate scores fully parallel using Half to Float precision",
          py::arg("key_quant"),
          py::arg("key_outlier_quant"),
          py::arg("key_norm"),
          py::arg("key_outlier_norm"),
          py::arg("outlier_indices"),
          py::arg("query_sketch"),
          py::arg("query_states"),
          py::arg("rand_prj"));

    m.def("qjl_gqa_score_cuda_float_float", &QJLGQAScoreCudaTemplate<float, float>, "Cuda kernel to calculate scores fully parallel using Float precision",
          py::arg("key_quant"),
          py::arg("key_outlier_quant"),
          py::arg("key_norm"),
          py::arg("key_outlier_norm"),
          py::arg("outlier_indices"),
          py::arg("query_sketch"),
          py::arg("query_states"),
          py::arg("rand_prj"));

    m.def("qjl_gqa_score_cuda_bf16_bf16", &QJLGQAScoreCudaTemplate<at::BFloat16, at::BFloat16>, "Cuda kernel to calculate scores fully parallel using BF16 precision",
          py::arg("key_quant"),
          py::arg("key_outlier_quant"),
          py::arg("key_norm"),
          py::arg("key_outlier_norm"),
          py::arg("outlier_indices"),
          py::arg("query_sketch"),
          py::arg("query_states"),
          py::arg("rand_prj"));

    m.def("qjl_gqa_score_cuda_bf16_float", &QJLGQAScoreCudaTemplate<at::BFloat16, float>, "Cuda kernel to calculate scores fully parallel using BF16 to Float precision",
          py::arg("key_quant"),
          py::arg("key_outlier_quant"),
          py::arg("key_norm"),
          py::arg("key_outlier_norm"),
          py::arg("outlier_indices"),
          py::arg("query_sketch"),
          py::arg("query_states"),
          py::arg("rand_prj"));
}
