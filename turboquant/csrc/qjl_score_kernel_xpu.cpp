#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <cmath>

template <typename T> inline float convert_to_float(T value) { return 0.0f; }
template <> inline float convert_to_float<c10::Half>(c10::Half value) { return (float)value; }
template <> inline float convert_to_float<float>(float value) { return value; }
template <> inline float convert_to_float<at::BFloat16>(at::BFloat16 value) { return (float)value; }

template <typename T> inline T convert_from_float(float value) { return (T)0; }
template <> inline c10::Half convert_from_float<c10::Half>(float value) { return (c10::Half)value; }
template <> inline float convert_from_float<float>(float value) { return value; }
template <> inline at::BFloat16 convert_from_float<at::BFloat16>(float value) { return (at::BFloat16)value; }


#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define EMB_DIM 128
#define FULL_MASK 0xffffffff









inline float warpReduceSum(sycl::sub_group sg, float val) {
    return sycl::reduce_over_group(sg, val, sycl::plus<>());
}

template <typename T, typename Tproj>
[[intel::reqd_sub_group_size(32)]]
void calc_score_kernel(T *query_states, const uint8_t *key_quant,
                       const uint8_t *key_outlier_quant, T *key_norm,
                       T *key_outlier_norm, const uint8_t *outlier_indices,
                       const float *query_sketch, const Tproj *rand_prj,
                       float *scores, int batch_size, int head_size, int n_size,
                       int group_size, int sketch_dim, int outlier_sketch_dim,
                       int emb_dim, int outlier_counts, float *shared_query,
                       uint8_t *shared_outlier_ind, float *shared_innprod,
                       float *shared_outlier_innprod,
                       float shared_q_sketch[32 /*WARP_SIZE*/][8],
                       float shared_q_outliers_sketch[32 /*WARP_SIZE*/][8]) {

    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sg = item_ct1.get_sub_group();
    size_t bh = item_ct1.get_group(2);
    size_t n = item_ct1.get_group(1);
    size_t threadLane = sg.get_local_id()[0];
    size_t wIdx = item_ct1.get_local_id(1);
    size_t gIdx = item_ct1.get_group(0) * 32;

    int hash_dim = sketch_dim/8;
    int outlier_hash_dim = outlier_sketch_dim/8;

    int base_index_outlier_indices = (bh * n_size * outlier_counts) + (n * outlier_counts);
    const uint8_t* outlier_ind = outlier_indices + base_index_outlier_indices;

    int base_index_query_sketch = (bh * sketch_dim);
    const float* q_sketch = query_sketch + base_index_query_sketch;

    int base_index_key_quant = (bh * n_size * group_size * hash_dim) + (n * group_size * hash_dim) + (gIdx * hash_dim);
    const uint8_t* k_quant = key_quant + base_index_key_quant;

    int base_index_outlier_quant = (bh * n_size * group_size * outlier_hash_dim) + (n * group_size * outlier_hash_dim) + (gIdx * outlier_hash_dim);
    const uint8_t* outlier_quant = key_outlier_quant + base_index_outlier_quant;

    int base_index_key_norm = (bh * n_size * group_size) + (n * group_size) + gIdx;
    const T* k_norm = key_norm + base_index_key_norm;
    const T* outlier_norm = key_outlier_norm + base_index_key_norm;

    int base_index_query_states = (bh * emb_dim);
    const T* query = query_states + base_index_query_states;

    size_t tIdx = wIdx * 32 + threadLane;
    for (size_t tile_idx{tIdx}; tile_idx < emb_dim; tile_idx += (32 * 8)) {
        shared_query[tile_idx] = convert_to_float<T>(query[tile_idx]);
    }

    for (size_t tile_idx{tIdx}; tile_idx < outlier_counts; tile_idx += (32 * 8)) {
        shared_outlier_ind[tile_idx] = outlier_ind[tile_idx];
    }

    if (wIdx == 0) {
        shared_innprod[threadLane] = 0.0;
        shared_outlier_innprod[threadLane] = 0.0;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    for (size_t chnl_tile{0}; chnl_tile < sketch_dim; chnl_tile += (8*32)){
        for (size_t q_idx{tIdx}; q_idx < (8*32); q_idx += (32 * 8)) {
            shared_q_sketch[q_idx/8][q_idx%8] = 0.0;
            shared_q_outliers_sketch[q_idx/8][q_idx%8] = 0.0;
            if (chnl_tile+q_idx < sketch_dim){
                shared_q_sketch[q_idx/8][q_idx%8] = q_sketch[chnl_tile+q_idx];
                for (size_t i{0}; i < outlier_counts; i++){
                    int otlr_idx = shared_outlier_ind[i];
                    shared_q_outliers_sketch[q_idx/8][q_idx%8] += shared_query[otlr_idx] * convert_to_float<Tproj>(rand_prj[(otlr_idx * sketch_dim) + chnl_tile+q_idx]);
                }
            }
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (size_t grp_tile{wIdx}; grp_tile < 32; grp_tile += 8) {
            uint8_t key_quant_buffer = 0;
            if (chnl_tile / 8 + threadLane < hash_dim) {
                key_quant_buffer = k_quant[grp_tile * hash_dim + chnl_tile / 8 + threadLane];
            }
            uint8_t outlier_quant_buffer = 0;
            if (chnl_tile / 8 + threadLane < outlier_hash_dim) {
                outlier_quant_buffer = outlier_quant[grp_tile * outlier_hash_dim + chnl_tile / 8 + threadLane];
            }

            float k_inner_prod = 0.0;
            float outlier_inner_prod = 0.0;
            for (int shift = 0; shift < 8; shift++) {
                if (chnl_tile + 8 * threadLane + shift < sketch_dim) {
                    float q_sketch_val = shared_q_sketch[threadLane][shift] - shared_q_outliers_sketch[threadLane][shift];
                    k_inner_prod += (((key_quant_buffer >> shift) & 1) ? q_sketch_val : -q_sketch_val);
                }
                if (chnl_tile + 8 * threadLane + shift < outlier_sketch_dim) {
                    float q_otlr_sketch_val = shared_q_outliers_sketch[threadLane][shift];
                    outlier_inner_prod += (((outlier_quant_buffer >> shift) & 1) ? q_otlr_sketch_val : -q_otlr_sketch_val);
                }
            }

            k_inner_prod = warpReduceSum(sg, k_inner_prod);
            outlier_inner_prod = warpReduceSum(sg, outlier_inner_prod);

            if (threadLane == 0) {
                shared_innprod[grp_tile] += k_inner_prod;
                shared_outlier_innprod[grp_tile] += outlier_inner_prod;
            }
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    if (gIdx+threadLane >= group_size) return;
    if (wIdx == 0) {
        float scl = 1.25331413732f / static_cast<float>(sketch_dim); // sqrt(pi/2)
        float scl_otlr = 1.25331413732f / static_cast<float>(outlier_sketch_dim);
        float norm_otlr = convert_to_float<T>(outlier_norm[threadLane]);
        float nk_val = convert_to_float<T>(k_norm[threadLane]);
        float norm_k = sycl::sqrt(sycl::max(0.0f, nk_val * nk_val - norm_otlr * norm_otlr));
        float score = scl * norm_k * shared_innprod[threadLane] + scl_otlr * norm_otlr * shared_outlier_innprod[threadLane];
        scores[(bh * n_size * group_size) + (n * group_size) + gIdx + threadLane] = score;
    }
}


template <typename T, typename Tproj>
torch::Tensor QJLScoreCudaTemplate(
    torch::Tensor key_quant,
    torch::Tensor key_outlier_quant,
    torch::Tensor key_norm,
    torch::Tensor key_outlier_norm,
    torch::Tensor outlier_indices,
    torch::Tensor query_sketch,
    torch::Tensor query_states,
    torch::Tensor rand_prj) {

    auto options = torch::TensorOptions().device(torch::kXPU, 0).dtype(torch::kFloat);

    int emb_dim = query_states.size(-1);
    int group_size = 32;
    int sketch_dim = rand_prj.size(1); // Score kernel uses (head_dim, sketch_dim) layout?
    int outlier_sketch_dim = 8 * key_outlier_quant.size(-1);
    int outlier_counts = outlier_indices.size(-1);
    
    // Flatten batch/head/n dimensions
    int num_elements = key_quant.numel();
    int batch_head_n = num_elements / (group_size * (sketch_dim / 8));
    int batch = 1, head = 1, n = batch_head_n;

    auto scores = torch::zeros({batch, head, n * group_size, 1}, options).contiguous();
    
    auto query_states_ptr = query_states.data_ptr<T>();
    auto key_norm_ptr = key_norm.data_ptr<T>();
    auto key_outlier_norm_ptr = key_outlier_norm.data_ptr<T>();
    auto rand_prj_ptr = rand_prj.data_ptr<Tproj>();

    sycl::range<3> numBlocks(batch * head, n, (group_size + 31) / 32);
    sycl::range<3> threadsPerBlockDim(32, 8, 1);

    c10::xpu::getCurrentXPUStream().queue().submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:55: 'EMB_DIM' expression was replaced with a value. Modify the
        code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float, 1> shared_query_acc_ct1(
            sycl::range<1>(128 /*EMB_DIM*/), cgh);
        /*
        DPCT1101:56: 'WARP_SIZE' expression was replaced with a value. Modify
        the code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<uint8_t, 1> shared_outlier_ind_acc_ct1(
            sycl::range<1>(32 /*WARP_SIZE*/), cgh);
        /*
        DPCT1101:57: 'WARP_SIZE' expression was replaced with a value. Modify
        the code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float, 1> shared_innprod_acc_ct1(
            sycl::range<1>(32 /*WARP_SIZE*/), cgh);
        /*
        DPCT1101:58: 'WARP_SIZE' expression was replaced with a value. Modify
        the code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float, 1> shared_outlier_innprod_acc_ct1(
            sycl::range<1>(32 /*WARP_SIZE*/), cgh);
        /*
        DPCT1101:59: 'WARP_SIZE' expression was replaced with a value. Modify
        the code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float[32 /*WARP_SIZE*/][8], 0>
            shared_q_sketch_acc_ct1(cgh);
        /*
        DPCT1101:60: 'WARP_SIZE' expression was replaced with a value. Modify
        the code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float[32 /*WARP_SIZE*/][8], 0>
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
                calc_score_kernel(
                    query_states_ptr, key_quant_data_ptr_uint8_t_ct1,
                    key_outlier_quant_data_ptr_uint8_t_ct2, key_norm_ptr,
                    key_outlier_norm_ptr, outlier_indices_data_ptr_uint8_t_ct5,
                    query_sketch_data_ptr_float_ct6, rand_prj_ptr,
                    scores_data_ptr_float_ct8, batch, head, n, group_size,
                    sketch_dim, outlier_sketch_dim, emb_dim, outlier_counts,
                    shared_query_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    shared_outlier_ind_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    shared_innprod_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    shared_outlier_innprod_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    shared_q_sketch_acc_ct1, shared_q_outliers_sketch_acc_ct1);
            });
    });

    return scores;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qjl_score_xpu_half_half", &QJLScoreCudaTemplate<c10::Half, c10::Half>);
    m.def("qjl_score_xpu_half_float", &QJLScoreCudaTemplate<c10::Half, float>);
    m.def("qjl_score_xpu_float_float", &QJLScoreCudaTemplate<float, float>);
    m.def("qjl_score_xpu_bf16_bf16", &QJLScoreCudaTemplate<at::BFloat16, at::BFloat16>);
    m.def("qjl_score_xpu_bf16_float", &QJLScoreCudaTemplate<at::BFloat16, float>);
}
