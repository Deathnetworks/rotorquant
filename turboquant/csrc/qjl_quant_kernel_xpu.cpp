#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <c10/xpu/XPUStream.h>

// Amortized Fused Sparse Outlier and Packing Kernel V5 (64-bit mask checks)
void compute_qjl_fused_multi_row_v5_kernel(
    const sycl::half* __restrict__ key_states,
    const uint8_t* __restrict__ outlier_indices,
    const sycl::half* __restrict__ rand_prj,
    const sycl::half* __restrict__ total_results,
    uint8_t* __restrict__ key_quant,
    uint8_t* __restrict__ key_outlier_quant,
    sycl::half* __restrict__ outlier_norms,
    int total_rows,
    int sketch_dim,
    int emb_dim,
    int outlier_counts,
    sycl::half* shared_proj // [128][128]
) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    auto sg = item_ct1.get_sub_group();
    int lid = item_ct1.get_local_id(0);
    int gid = item_ct1.get_group(0);
    int sg_lid = sg.get_local_id()[0];
    int sg_id = lid / 32;
    
    int start_row = gid * 128;
    int num_rows_in_block = sycl::min(128, total_rows - start_row);

    // 1. Load proj into SLM
    for (int i = 0; i < 128; i++) {
        shared_proj[i * 128 + lid] = rand_prj[lid * 128 + i];
    }
    
    // 2. Load 4 masks using 64-bit loads
    uint64_t mask_u64[4][2];
    bool block_has_outliers = false;
    for (int m_idx = 0; m_idx < 4; m_idx++) {
        int bhn = (start_row / 32) + m_idx;
        if (bhn * 32 < total_rows) {
            const uint64_t* m_ptr = (const uint64_t*)(outlier_indices + (bhn * outlier_counts));
            mask_u64[m_idx][0] = m_ptr[0];
            mask_u64[m_idx][1] = m_ptr[1];
            if (mask_u64[m_idx][0] != 0 || mask_u64[m_idx][1] != 0) block_has_outliers = true;
        } else {
            mask_u64[m_idx][0] = 0;
            mask_u64[m_idx][1] = 0;
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    // 3. Main Loop
    for (int r = 0; r < num_rows_in_block; r++) {
        int row_idx = start_row + r;
        float out_sum = 0.0f;
        
        if (block_has_outliers) {
            int m_idx = r / 32;
            if (mask_u64[m_idx][0] != 0 || mask_u64[m_idx][1] != 0) {
                const uint8_t* local_mask = (const uint8_t*)&mask_u64[m_idx][0];
                
                // Norm
                if (lid == 0) {
                    float norm_sum = 0.0f;
                    for (int c_byte = 0; c_byte < 16; c_byte++) {
                        uint8_t m = local_mask[c_byte];
                        if (m == 0) continue;
                        for (int i = 0; i < 8; i++) {
                            if ((m >> i) & 1) {
                                float val = (float)key_states[row_idx * emb_dim + c_byte * 8 + i];
                                norm_sum += val * val;
                            }
                        }
                    }
                    outlier_norms[row_idx] = (sycl::half)std::sqrt(norm_sum);
                }

                // Sparse Projection
                for (int c_byte = 0; c_byte < 16; c_byte++) {
                    uint8_t m = local_mask[c_byte];
                    if (m == 0) continue;
                    uint8_t temp_m = m;
                    while (temp_m > 0) {
                        int i = sycl::ctz((uint16_t)temp_m);
                        int c = c_byte * 8 + i;
                        out_sum += (float)key_states[row_idx * emb_dim + c] * (float)shared_proj[c * 128 + lid];
                        temp_m &= ~(1 << i);
                    }
                }
            } else {
                if (lid == 0) outlier_norms[row_idx] = 0.0f;
            }
        } else {
            if (lid == 0) outlier_norms[row_idx] = 0.0f;
        }

        // Packing
        float total = (float)total_results[row_idx * sketch_dim + lid];
        bool in_bit = (total - out_sum) > 0.0f;
        bool out_bit = out_sum > 0.0f;

        auto in_mask = sycl::ext::oneapi::group_ballot(sg, in_bit);
        auto out_mask = sycl::ext::oneapi::group_ballot(sg, out_bit);

        if (sg_lid == 0) {
            uint32_t in_val;
            in_mask.extract_bits(in_val);
            uint32_t out_val;
            out_mask.extract_bits(out_val);
            int out_offset = row_idx * (sketch_dim / 8) + (sg_id * 4);
            *(uint32_t*)(key_quant + out_offset) = in_val;
            *(uint32_t*)(key_outlier_quant + out_offset) = out_val;
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qjl_quant", [](
        torch::Tensor key_states,
        torch::Tensor outlier_indices,
        torch::Tensor rand_prj,
        int outlier_sketch_dim) {
        
        int batch = (int)key_states.size(0);
        int head = (int)key_states.size(1);
        int num_groups = (int)key_states.size(2);
        int group_size = (int)key_states.size(3);
        int emb_dim = (int)key_states.size(4);
        int sketch_dim = (int)rand_prj.size(0);
        int total_rows = batch * head * num_groups * group_size;

        auto flat_keys = key_states.view({-1, emb_dim});
        auto total_results = torch::matmul(flat_keys, rand_prj.t());
        
        auto options_u8 = torch::TensorOptions().device(torch::kXPU, 0).dtype(torch::kUInt8);
        auto key_quant = torch::empty({batch, head, num_groups, group_size, sketch_dim / 8}, options_u8);
        auto key_outlier_quant = torch::empty({batch, head, num_groups, group_size, outlier_sketch_dim / 8}, options_u8);
        auto outlier_norms = torch::empty({batch, head, num_groups, group_size}, key_states.options());
        
        auto &q = c10::xpu::getCurrentXPUStream().queue();
        q.submit([&](sycl::handler &cgh) {
            auto keys_ptr = reinterpret_cast<const sycl::half*>(key_states.data_ptr<c10::Half>());
            auto mask_ptr = outlier_indices.data_ptr<uint8_t>();
            auto proj_ptr = reinterpret_cast<const sycl::half*>(rand_prj.data_ptr<c10::Half>());
            auto total_ptr = reinterpret_cast<const sycl::half*>(total_results.data_ptr<c10::Half>());
            auto quant_ptr = key_quant.data_ptr<uint8_t>();
            auto o_quant_ptr = key_outlier_quant.data_ptr<uint8_t>();
            auto out_norms_ptr = reinterpret_cast<sycl::half*>(outlier_norms.data_ptr<c10::Half>());
            int outlier_counts = (int)outlier_indices.size(-1);
            
            sycl::local_accessor<sycl::half, 1> shared_proj(sycl::range<1>(128 * 128), cgh);

            int num_blocks = (total_rows + 127) / 128;
            cgh.parallel_for(sycl::nd_range<1>(num_blocks * 128, 128), [=](sycl::nd_item<1> item) {
                compute_qjl_fused_multi_row_v5_kernel(keys_ptr, mask_ptr, proj_ptr, total_ptr, quant_ptr, o_quant_ptr, out_norms_ptr, total_rows, sketch_dim, emb_dim, outlier_counts,
                    shared_proj.get_multi_ptr<sycl::access::decorated::no>().get());
            });
        });

        return std::make_tuple(key_quant, key_outlier_quant, outlier_norms);
    });
}
