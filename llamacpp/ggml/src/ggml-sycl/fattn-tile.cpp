#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/work_group_static.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "fattn-common.hpp"
#include "fattn-tile.hpp"
#include <cmath>
#include <float.h>
namespace syclex = sycl::ext::oneapi::experimental;

#define FATTN_TILE_CASE(DKQ, DV) \
    if (Q->ne[0] == (DKQ) && V->ne[0] == (DV)) { \
        ggml_sycl_flash_attn_ext_tile_case<DKQ, DV>(ctx, dst); \
        return; \
    }

void ggml_sycl_flash_attn_ext_tile(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * V = dst->src[2];

    FATTN_TILE_CASE( 40,  40);
    FATTN_TILE_CASE( 64,  64);
    FATTN_TILE_CASE( 72,  72);
    FATTN_TILE_CASE( 80,  80);
    FATTN_TILE_CASE( 96,  96);
    FATTN_TILE_CASE(112, 112);
    FATTN_TILE_CASE(128, 128);
    FATTN_TILE_CASE(256, 256);
    FATTN_TILE_CASE(512, 512);
    FATTN_TILE_CASE(576, 512);

    GGML_ABORT("No Flash-Attention tile kernel found for head size Q=%lld, V=%lld", (long long)Q->ne[0], (long long)V->ne[0]);
}
