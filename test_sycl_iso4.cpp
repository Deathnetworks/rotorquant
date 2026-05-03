
#include <iostream>
#include <vector>
#include <cmath>
#include <sycl/sycl.hpp>
#include "ggml-sycl.h"
#include "ggml-sycl/convert.cpp" // Include the implementation for testing
#include "ggml-sycl/dequantize.hpp"

// We need to mock some things or include headers
#include "ggml.h"

// Define block_iso4 for the test if not available
typedef struct {
    uint16_t d; // ggml_half
    uint8_t qs[16];
} block_iso4_test;

int main() {
    sycl::queue q;
    std::cout << "Using device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    const int ne = 32;
    block_iso4_test h_x;
    h_x.d = 0x3C00; // 1.0 in FP16
    for(int i=0; i<16; i++) h_x.qs[i] = 0x88; // 8, 8 -> centroids[8], centroids[8]

    block_iso4_test *d_x = sycl::malloc_device<block_iso4_test>(1, q);
    float *d_y = sycl::malloc_device<float>(ne, q);

    q.memcpy(d_x, &h_x, sizeof(block_iso4_test)).wait();

    // Call dequantize_row_iso4_sycl
    // We need to match the signature in convert.cpp
    // template <typename dst_t>
    // static void dequantize_row_iso4_sycl(const void *vx, dst_t *y, const int64_t k, const int64_t ne00, dpct::queue_ptr stream)

    dequantize_row_iso4_sycl<float>(d_x, d_y, ne, ne, &q);
    q.wait();

    std::vector<float> h_y(ne);
    q.memcpy(h_y.data(), d_y, ne * sizeof(float)).wait();

    std::cout << "Dequantized values:" << std::endl;
    for(int i=0; i<ne; i++) {
        std::cout << h_y[i] << " ";
        if((i+1)%8 == 0) std::cout << std::endl;
    }

    sycl::free(d_x, q);
    sycl::free(d_y, q);

    return 0;
}
