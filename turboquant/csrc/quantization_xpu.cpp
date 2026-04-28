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
#define PACK_SIZE 16

template <typename T>
void batchedQuantizedMultiplyAccumulate(T *_inputs, const uint32_t *_weight,
                                        T *_zeros, T *_scale, T *_outputs,
                                        const int IC, const int OC,
                                        const int group_size, const int nh,
                                        const bool mqa, const int bit) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sg = item_ct1.get_sub_group();
    const int pack_factor = 32 / bit;
    const int batch_idx = item_ct1.get_group(2);
    const int packed_oc_idx = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    const int oc_start_idx = packed_oc_idx * pack_factor;
    const int group_idx = oc_start_idx / group_size; 
    
    T* inputs = _inputs + batch_idx * IC;
    T* outputs = _outputs + batch_idx * OC;
    int _batch_idx = mqa ? (batch_idx / nh) : batch_idx;
    
    const uint32_t* weight = _weight + _batch_idx * OC * IC / pack_factor;
    T* scaling_factors = _scale + _batch_idx * OC * IC / group_size;
    T* zeros = _zeros + _batch_idx * OC * IC / group_size;
    
    const int TILE_DIM = 128;
    const int num = 0xFF >> (8-bit);
    float psum[PACK_SIZE]{};

    for (int k=0; k < (IC + TILE_DIM - 1) / TILE_DIM; k++){
      uint32_t qw[4]{};
      T cscale[4]{}, czero[4]{}, inp[4]{};
      int lid = item_ct1.get_local_id(2);
      int weight_offset = packed_oc_idx * IC + k * TILE_DIM + lid * 4;
      int scale_mn_offset = group_idx * IC + k * TILE_DIM + lid * 4;
      int inputs_ptr_delta = k * TILE_DIM + lid * 4;

      for (int i=0; i<4; i++){
        if (weight_offset + i < OC * IC / pack_factor) qw[i] = *(weight + weight_offset + i);
        if (scale_mn_offset + i < OC * IC / group_size) {
          cscale[i] = *(scaling_factors + scale_mn_offset + i);
          czero[i] = *(zeros + scale_mn_offset + i);
        }
        if (inputs_ptr_delta + i < IC) inp[i] = *(inputs + inputs_ptr_delta + i);
      }

      #pragma unroll
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        uint32_t cur_packed_weight = qw[ic_0];
        float cur_inp = convert_to_float<T>(inp[ic_0]);
        float cur_scale = convert_to_float<T>(cscale[ic_0]);
        float cur_zero = convert_to_float<T>(czero[ic_0]);
        for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
          int oc_idx = oc_start_idx + ic_1;
          if (oc_idx < OC){
            float cur_single_weight_fp = (float)(cur_packed_weight & num);
            float dequantized_weight = cur_scale * cur_single_weight_fp + cur_zero;
            cur_packed_weight = cur_packed_weight >> bit;
            psum[ic_1] += dequantized_weight * cur_inp;
          }
        }
      }
    }

    for (int i=0; i < pack_factor; i++){
      int oc_idx = oc_start_idx + i;
      if (oc_idx < OC){
        float sum = sycl::reduce_over_group(sg, psum[i], sycl::plus<>());
        if (item_ct1.get_local_id(2) == 0) outputs[oc_idx] = convert_from_float<T>(sum);
      }
    }
}

template <typename T>
torch::Tensor batchedQuantizedMultiplyAccumulateTemplate(torch::Tensor _in_feats, torch::Tensor _kernel, torch::Tensor _scaling_factors, torch::Tensor _zeros, const int bit, const int group_size, const int nh, const bool mqa) {
    int BS = _in_feats.size(0), num_in_feats = _in_feats.size(1), num_in_channels = _in_feats.size(2);
    int num_out_channels = _zeros.size(1) * group_size;
    auto in_feats = _in_feats.data_ptr<T>();
    auto zeros = _zeros.data_ptr<T>();
    auto scaling_factors = _scaling_factors.data_ptr<T>();
    auto out_feats_tensor = torch::empty({BS, num_in_feats, num_out_channels}, _in_feats.options());
    auto out_p = out_feats_tensor.data_ptr<T>();
    auto kernel = reinterpret_cast<uint32_t*>(_kernel.data_ptr<int>());
    int pack_factor = 32 / bit;
    sycl::range<3> num_blocks(BS, (num_out_channels / pack_factor + 3) / 4, num_in_feats);
    sycl::range<3> num_threads(32, 4, 1);
    c10::xpu::getCurrentXPUStream().queue().parallel_for(sycl::nd_range<3>(num_blocks * num_threads, num_threads), [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
        batchedQuantizedMultiplyAccumulate(in_feats, kernel, zeros, scaling_factors, out_p, num_in_channels, num_out_channels, group_size, nh, mqa, bit);
    });
    return out_feats_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batchedQuantizedMultiplyAccumulate_half", &batchedQuantizedMultiplyAccumulateTemplate<c10::Half>);
    m.def("batchedQuantizedMultiplyAccumulate_float", &batchedQuantizedMultiplyAccumulateTemplate<float>);
    m.def("batchedQuantizedMultiplyAccumulate_bf16", &batchedQuantizedMultiplyAccumulateTemplate<at::BFloat16>);
}