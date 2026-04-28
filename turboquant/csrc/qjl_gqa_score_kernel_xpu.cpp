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
#define WARPS_PER_BLOCK 8
#define EMB_DIM 128
#define GQA_GROUP_SIZE 4

template <typename T, typename Tproj>
[[intel::reqd_sub_group_size(32)]]
void calc_gqa_score_kernel(
    T *query_states, const uint8_t *key_quant, const uint8_t *key_outlier_quant,
    T *key_norm, T *key_outlier_norm, const uint8_t *outlier_indices,
    const float *query_sketch, const Tproj *rand_prj, float *scores,
    int batch_size, int key_head_size, int n_size, int group_size,
    int gqa_group_size, int sketch_dim, int outlier_sketch_dim, int emb_dim,
    int outlier_counts,
    float shared_query[4][128],
    uint8_t *shared_outlier_ind,
    float shared_innprod[4][32],
    float shared_outlier_innprod[4][32],
    float shared_q_sketch[4][32][8],
    float shared_q_outliers_sketch[4][32][8]) {

    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sg = item_ct1.get_sub_group();
    size_t k_bh = item_ct1.get_group(2);
    size_t n = item_ct1.get_group(1);
    size_t q_bh = gqa_group_size * k_bh;
    size_t threadLane = sg.get_local_id()[0];
    size_t wIdx = item_ct1.get_local_id(1);
    size_t gIdx = item_ct1.get_group(0) * WARP_SIZE;

    int hash_dim = sketch_dim/8;
    int outlier_hash_dim = outlier_sketch_dim/8;

    int base_index_outlier_indices = (k_bh * n_size * outlier_counts) + (n * outlier_counts);
    const uint8_t* outlier_ind = outlier_indices + base_index_outlier_indices;

    int base_index_query_sketch = (q_bh * sketch_dim);
    const float* q_sketch_base = query_sketch + base_index_query_sketch;

    int base_index_key_quant = (k_bh * n_size * group_size * hash_dim) + (n * group_size * hash_dim) + (gIdx * hash_dim);
    const uint8_t* k_quant = key_quant + base_index_key_quant;

    int base_index_outlier_quant = (k_bh * n_size * group_size * outlier_hash_dim) + (n * group_size * outlier_hash_dim) + (gIdx * outlier_hash_dim);
    const uint8_t* outlier_quant = key_outlier_quant + base_index_outlier_quant;

    int base_index_key_norm = (k_bh * n_size * group_size) + (n * group_size) + gIdx;
    const T* k_norm = key_norm + base_index_key_norm;
    const T* outlier_norm = key_outlier_norm + base_index_key_norm;

    int base_index_query_states = (q_bh * emb_dim);
    const T* query_base = query_states + base_index_query_states;

    size_t tIdx = wIdx * 32 + threadLane;
    for (size_t tile_idx{tIdx}; tile_idx < (gqa_group_size * emb_dim); tile_idx += (32 * WARPS_PER_BLOCK)) {
        int h_idx = tile_idx / emb_dim;
        int chnl_idx = tile_idx % emb_dim;
        shared_query[h_idx][chnl_idx] = convert_to_float<T>(query_base[h_idx*emb_dim + chnl_idx]);
    }
    for (size_t tile_idx{tIdx}; tile_idx < outlier_counts; tile_idx += (32 * WARPS_PER_BLOCK)) {
        shared_outlier_ind[tile_idx] = outlier_ind[tile_idx];
    }
    if (wIdx < gqa_group_size) {
        shared_innprod[wIdx][threadLane] = 0.0;
        shared_outlier_innprod[wIdx][threadLane] = 0.0;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    for (size_t chnl_tile{0}; chnl_tile < sketch_dim; chnl_tile += (8*WARP_SIZE)){
        for (size_t gqa_idx{0}; gqa_idx < gqa_group_size; gqa_idx++){
            for (size_t q_idx{tIdx}; q_idx < (8*WARP_SIZE); q_idx += (32 * WARPS_PER_BLOCK)) {
                if (chnl_tile+q_idx < sketch_dim){
                    shared_q_sketch[gqa_idx][q_idx/8][q_idx%8] = q_sketch_base[(gqa_idx*sketch_dim) + chnl_tile+q_idx];
                } else {
                    shared_q_sketch[gqa_idx][q_idx/8][q_idx%8] = 0.0;
                }
            }
        }
        for (size_t q_idx{tIdx}; q_idx < (8*WARP_SIZE); q_idx += (32 * WARPS_PER_BLOCK)) {
            for (size_t gqa_idx{0}; gqa_idx < gqa_group_size; gqa_idx++) shared_q_outliers_sketch[gqa_idx][q_idx/8][q_idx%8] = 0.0;
            if (chnl_tile+q_idx < sketch_dim){
                for (size_t i{0}; i < outlier_counts; i++){
                    int otlr_idx = shared_outlier_ind[i];
                    float rand_prj_buffer = convert_to_float<Tproj>(rand_prj[(otlr_idx * sketch_dim) + chnl_tile+q_idx]);
                    for (size_t gqa_idx{0}; gqa_idx < gqa_group_size; gqa_idx++) {
                        shared_q_outliers_sketch[gqa_idx][q_idx/8][q_idx%8] += shared_query[gqa_idx][otlr_idx] * rand_prj_buffer;
                    }
                }
            }
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (size_t grp_tile{wIdx}; grp_tile < WARP_SIZE; grp_tile += WARPS_PER_BLOCK) {
            uint8_t key_quant_buffer = k_quant[grp_tile*hash_dim + chnl_tile/8 + threadLane];
            uint8_t outlier_quant_buffer = 0;
            if (chnl_tile + 8*threadLane < outlier_sketch_dim){
                outlier_quant_buffer = outlier_quant[grp_tile*outlier_hash_dim + chnl_tile/8 + threadLane];
            }

            for (size_t gqa_idx{0}; gqa_idx < gqa_group_size; gqa_idx++) {
                float k_inner_prod = 0.0, outlier_inner_prod = 0.0;
                for (int shift = 0; shift < 8; shift++) {
                    float q_sketch_val = shared_q_sketch[gqa_idx][threadLane][shift] - shared_q_outliers_sketch[gqa_idx][threadLane][shift];
                    k_inner_prod += (((key_quant_buffer >> shift)&1) ? q_sketch_val :-q_sketch_val);
                    if (chnl_tile + 8*threadLane < outlier_sketch_dim) {
                        float q_otlr_sketch_val = shared_q_outliers_sketch[gqa_idx][threadLane][shift];
                        outlier_inner_prod += (((outlier_quant_buffer >> shift)&1) ? q_otlr_sketch_val :-q_otlr_sketch_val);
                    }
                }
                k_inner_prod = sycl::reduce_over_group(sg, k_inner_prod, sycl::plus<>());
                outlier_inner_prod = sycl::reduce_over_group(sg, outlier_inner_prod, sycl::plus<>());
                if (threadLane == 0) {
                    shared_innprod[gqa_idx][grp_tile] += k_inner_prod;
                    shared_outlier_innprod[gqa_idx][grp_tile] += outlier_inner_prod;
                }
            }
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    if (gIdx+threadLane >= group_size) return;
    if (wIdx < gqa_group_size) {
        float scl = 1.25331413731550025121f / static_cast<float>(sketch_dim); // sqrt(pi/2)
        float scl_otlr = 1.25331413731550025121f / static_cast<float>(outlier_sketch_dim);
        float norm_otlr = convert_to_float<T>(outlier_norm[threadLane]);
        float norm_k_sq = powf(convert_to_float<T>(k_norm[threadLane]), 2.0f) - norm_otlr * norm_otlr;
        float norm_k = sqrtf(fmaxf(0.0f, norm_k_sq));
        float score = scl * norm_k * shared_innprod[wIdx][threadLane] + scl_otlr * norm_otlr * shared_outlier_innprod[wIdx][threadLane];
        scores[((q_bh + wIdx) * n_size * group_size) + (n * group_size) + gIdx + threadLane] = score;
    }
}

template <typename T, typename Tproj>
torch::Tensor QJLGQAScoreCudaTemplate(
    torch::Tensor key_quant, torch::Tensor key_outlier_quant, torch::Tensor key_norm,
    torch::Tensor key_outlier_norm, torch::Tensor outlier_indices, torch::Tensor query_sketch,
    torch::Tensor query_states, torch::Tensor rand_prj) {
    int batch = key_quant.size(0), k_head = key_quant.size(1), n = key_quant.size(2), group_size = key_quant.size(3);
    int q_head = query_states.size(1), emb_dim = query_states.size(3), sketch_dim = rand_prj.size(1);
    int outlier_sketch_dim = 8*key_outlier_quant.size(4), outlier_counts = outlier_indices.size(3), gqa_group_size = q_head / k_head;
    auto scores = torch::zeros({batch, q_head, n * group_size, 1}, torch::TensorOptions().device(torch::kXPU, 0).dtype(torch::kFloat)).contiguous();
    sycl::range<3> numBlocks(batch * k_head, n, (group_size + 31) / 32);
    sycl::range<3> threadsPerBlockDim(32, 8, 1);
    c10::xpu::getCurrentXPUStream().queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float[4][128], 0> shared_query(cgh);
        sycl::local_accessor<uint8_t, 1> shared_outlier_ind(sycl::range<1>(32), cgh);
        sycl::local_accessor<float[4][32], 0> shared_innprod(cgh);
        sycl::local_accessor<float[4][32], 0> shared_outlier_innprod(cgh);
        sycl::local_accessor<float[4][32][8], 0> shared_q_sketch(cgh);
        sycl::local_accessor<float[4][32][8], 0> shared_q_outliers_sketch(cgh);
        auto qs_ptr = query_states.data_ptr<T>();
        auto k_q_ptr = key_quant.data_ptr<uint8_t>();
        auto ko_q_ptr = key_outlier_quant.data_ptr<uint8_t>();
        auto o_i_ptr = outlier_indices.data_ptr<uint8_t>();
        auto kn_ptr = key_norm.data_ptr<T>();
        auto kon_ptr = key_outlier_norm.data_ptr<T>();
        auto rp_ptr = rand_prj.data_ptr<Tproj>();
        auto qsk_ptr = query_sketch.data_ptr<float>();
        auto s_ptr = scores.data_ptr<float>();
        cgh.parallel_for(sycl::nd_range<3>(numBlocks * threadsPerBlockDim, threadsPerBlockDim), [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
            calc_gqa_score_kernel(qs_ptr, k_q_ptr, ko_q_ptr, kn_ptr, kon_ptr, o_i_ptr, qsk_ptr, rp_ptr, s_ptr, batch, k_head, n, group_size, gqa_group_size, sketch_dim, outlier_sketch_dim, emb_dim, outlier_counts, shared_query, shared_outlier_ind.get_pointer(), shared_innprod, shared_outlier_innprod, shared_q_sketch, shared_q_outliers_sketch);
        });
    });
    return scores;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qjl_gqa_score_xpu_half_half", &QJLGQAScoreCudaTemplate<c10::Half, c10::Half>);
    m.def("qjl_gqa_score_xpu_half_float", &QJLGQAScoreCudaTemplate<c10::Half, float>);
    m.def("qjl_gqa_score_xpu_float_float", &QJLGQAScoreCudaTemplate<float, float>);
    m.def("qjl_gqa_score_xpu_bf16_bf16", &QJLGQAScoreCudaTemplate<at::BFloat16, at::BFloat16>);
    m.def("qjl_gqa_score_xpu_bf16_float", &QJLGQAScoreCudaTemplate<at::BFloat16, float>);
}
