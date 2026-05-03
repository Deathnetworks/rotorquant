#ifndef GGML_SYCL_CPY_HPP
#define GGML_SYCL_CPY_HPP

#include "common.hpp"
#include <float.h>

typedef void (*cpy_kernel_t)(const char * cx, char * cdst, int ib);

__dpct_inline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) {
        return 0;
    }
    if (x >= val[n - 1]) {
        return n - 1;
    }
    int ml = 0, mu = n - 1;
    while (mu - ml > 1) {
        int mav = (ml + mu) / 2;
        if (x < val[mav]) {
            mu = mav;
        } else {
            ml = mav;
        }
    }
    return x - val[mu - 1] < val[mu] - x ? mu - 1 : mu;
}

inline void cpy_blck_f32_q8_0(const char * cxi, char * cdsti, int ib) {
    GGML_UNUSED(ib);
    const float * xi   = (const float *) cxi;
    block_q8_0 *  dsti = (block_q8_0 *) cdsti;

    float amax = 0.0f;  // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = xi[j];
        amax          = sycl::fmax(amax, sycl::fabs((float) v));
    }

    const float d  = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = xi[j] * id;

        dsti->qs[j] = sycl::round((float) x0);
    }
}

inline void cpy_blck_f32_q4_0(const char * cxi, char * cdsti, int ib) {
    GGML_UNUSED(ib);
    const float * xi   = (const float *) cxi;
    block_q4_0 *  dsti = (block_q4_0 *) cdsti;

    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = xi[j];
        if (amax < sycl::fabs((float) v)) {
            amax = sycl::fabs((float) v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    for (int j = 0; j < QK4_0 / 2; ++j) {
        const float x0 = xi[0 + j] * id;
        const float x1 = xi[QK4_0 / 2 + j] * id;

        const uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 8.5f));
        const uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 8.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

inline void cpy_blck_f32_q4_1(const char * cxi, char * cdsti, int ib) {
    GGML_UNUSED(ib);
    const float * xi   = (const float *) cxi;
    block_q4_1 *  dsti = (block_q4_1 *) cdsti;

    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = xi[j];

        vmin = sycl::min(v, vmin);
        vmax = sycl::max(v, vmax);
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = vmin;

    for (int j = 0; j < QK4_1 / 2; ++j) {
        const float x0 = (xi[0 + j] - vmin) * id;
        const float x1 = (xi[QK4_1 / 2 + j] - vmin) * id;

        const uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 0.5f));
        const uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 0.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

inline void cpy_blck_f32_q5_0(const char * cxi, char * cdsti, int ib) {
    GGML_UNUSED(ib);
    const float * xi   = (const float *) cxi;
    block_q5_0 *  dsti = (block_q5_0 *) cdsti;

    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = xi[j];
        if (amax < sycl::fabs((float) v)) {
            amax = sycl::fabs((float) v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0 / 2; ++j) {
        const float x0 = xi[0 + j] * id;
        const float x1 = xi[QK5_0 / 2 + j] * id;

        const uint8_t xi0 = dpct::min(31, (int8_t) (x0 + 16.5f));
        const uint8_t xi1 = dpct::min(31, (int8_t) (x1 + 16.5f));

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0 / 2);
    }
    memcpy(dsti->qh, &qh, sizeof(qh));
}

inline void cpy_blck_f32_q5_1(const char * cxi, char * cdsti, int ib) {
    GGML_UNUSED(ib);
    const float * xi   = (const float *) cxi;
    block_q5_1 *  dsti = (block_q5_1 *) cdsti;

    float min = xi[0];
    float max = xi[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = xi[j];
        min           = v < min ? v : min;
        max           = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1 / 2; ++j) {
        const float x0 = (xi[0 + j] - min) * id;
        const float x1 = (xi[QK5_1 / 2 + j] - min) * id;

        const uint8_t xi0 = (uint8_t) (x0 + 0.5f);
        const uint8_t xi1 = (uint8_t) (x1 + 0.5f);

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1 / 2);
    }
    memcpy(dsti->qh, &qh, sizeof(qh));
}

inline void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti, int ib) {
    GGML_UNUSED(ib);
    const float *  xi   = (const float *) cxi;
    block_iq4_nl * dsti = (block_iq4_nl *) cdsti;

    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = xi[j];
        if (amax < sycl::fabs((float) v)) {
            amax = sycl::fabs((float) v);
            vmax = v;
        }
    }

    float       d  = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f / d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL / 2; ++j) {
        const float   x0  = xi[0 + j] * id;
        const float   x1  = xi[QK4_NL / 2 + j] * id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        dsti->qs[j]       = xi0 | (xi1 << 4);
        const float v0    = kvalues_iq4nl[xi0];
        const float v1    = kvalues_iq4nl[xi1];
        const float w0    = xi[0 + j] * xi[0 + j];
        const float w1    = xi[QK4_NL / 2 + j] * xi[QK4_NL / 2 + j];
        sumqx += w0 * v0 * xi[j] + w1 * v1 * xi[QK4_NL / 2 + j];
        sumq2 += w0 * v0 * v0 + w1 * v1 * v1;
    }

    dsti->d = sumq2 > 0 ? sumqx / sumq2 : d;
}

#include "rotor-math.hpp"

inline void cpy_blck_f32_iso4(const char * cxi, char * cdsti, int ib) {
    const float * xi   = (const float *) cxi;
    block_iso4 *  dsti = (block_iso4 *) cdsti;

    float rotated[32];
    // Apply 4D quaternions to 8 groups of 4
    for (int g = 0; g < 8; g++) {
        float s, qx, qy, qz;
        get_deterministic_quaternion(ib * 8 + g, &s, &qx, &qy, &qz);
        float quad[4] = {xi[g*4], xi[g*4 + 1], xi[g*4 + 2], xi[g*4 + 3]};
        float res[4];
        quat_multiply_left(s, qx, qy, qz, quad, res);
        for (int l = 0; l < 4; l++) rotated[g*4 + l] = res[l];
    }

    float amax = 0.0f;
    for (int j = 0; j < 32; j++) amax = sycl::fmax(amax, sycl::fabs(rotated[j]));

    const float d = amax / 2.7330780898512423f;
    const float id = d ? 1.0f/d : 0.0f;
    dsti->d = GGML_FP32_TO_FP16(d);

    for (int j = 0; j < 32/2; ++j) {
        float x0 = rotated[2*j] * id;
        float x1 = rotated[2*j + 1] * id;

        uint8_t xi0 = 0, xi1 = 0;
        float mindist0 = FLT_MAX, mindist1 = FLT_MAX;

        for (int l = 0; l < 16; ++l) {
            float dist0 = sycl::fabs(x0 - iso4_normalized_centroids[l]);
            if (dist0 < mindist0) { mindist0 = dist0; xi0 = l; }
            float dist1 = sycl::fabs(x1 - iso4_normalized_centroids[l]);
            if (dist1 < mindist1) { mindist1 = dist1; xi1 = l; }
        }
        dsti->qs[j] = xi0 | (xi1 << 4);
    }
}

inline void cpy_blck_f32_rotor4(const char * cxi, char * cdsti, int ib) {
    const float * xi   = (const float *) cxi;
    block_rotor4 * dsti = (block_rotor4 *) cdsti;

    float rotated[32];
    // Apply rotors to 10 groups of 3
    for (int g = 0; g < 10; g++) {
        float s, p12, p13, p23;
        get_deterministic_rotor(ib * 10 + g, s, p12, p13, p23);
        float triple[3] = {xi[g*3], xi[g*3+1], xi[g*3+2]};
        float res[3];
        rotor_sandwich_vec(s, p12, p13, p23, triple, res);
        rotated[g*3]   = res[0];
        rotated[g*3+1] = res[1];
        rotated[g*3+2] = res[2];
    }
    // Leftovers
    rotated[30] = xi[30];
    rotated[31] = xi[31];

    float amax = 0.0f;
    for (int j = 0; j < 32; j++) {
        amax = sycl::fmax(amax, sycl::fabs(rotated[j]));
    }

    const float d = amax / 2.7330780898512423f; // Max centroid
    const float id = d ? 1.0f/d : 0.0f;

    dsti->d = GGML_FP32_TO_FP16(d);

    for (int j = 0; j < 32/2; ++j) {
        const float x0 = rotated[2*j] * id;
        const float x1 = rotated[2*j + 1] * id;

        uint8_t xi0 = 0;
        uint8_t xi1 = 0;
        float mindist0 = sycl::fabs(x0 - iso4_normalized_centroids[0]);
        float mindist1 = sycl::fabs(x1 - iso4_normalized_centroids[0]);

        for (int l = 1; l < 16; l++) {
            float dist0 = sycl::fabs(x0 - iso4_normalized_centroids[l]);
            float dist1 = sycl::fabs(x1 - iso4_normalized_centroids[l]);
            if (dist0 < mindist0) { mindist0 = dist0; xi0 = l; }
            if (dist1 < mindist1) { mindist1 = dist1; xi1 = l; }
        }
        dsti->qs[j] = xi0 | (xi1 << 4);
    }
}

void ggml_sycl_cpy(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1);
void ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif  // GGML_SYCL_CPY_HPP
