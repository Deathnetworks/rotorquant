// rotor-math.hpp
#ifndef GGML_SYCL_ROTOR_MATH_HPP
#define GGML_SYCL_ROTOR_MATH_HPP

#include "common.hpp"

// Centroids for ISO4 and ROTOR4
static const float iso4_normalized_centroids[16] = {
    -2.7330780898512423f, -2.069569107137198f, -1.6186094303625083f, -1.2567617768048578f,
    -0.9631621213322137f, -0.7107775586617066f, -0.47671758963283256f, -0.24436574347710214f,
     0.24436574347710214f,  0.47671758963283256f,  0.7107775586617066f,  0.9631621213322137f,
     1.2567617768048578f,  1.6186094303625083f,  2.069569107137198f,  2.7330780898512423f
};

#define rotor4_normalized_centroids iso4_normalized_centroids

static __dpct_inline__ float get_next_lcg(uint32_t * x) {
    *x = *x * 1664525u + 1013904223u;
    return (float)*x / 4294967296.0f;
}

static __dpct_inline__ void get_deterministic_quaternion(int group_idx, float * s, float * x, float * y, float * z) {
    uint32_t seed = 42 + (uint32_t)group_idx;
    float q0 = get_next_lcg(&seed) * 2.0f - 1.0f;
    float q1 = get_next_lcg(&seed) * 2.0f - 1.0f;
    float q2 = get_next_lcg(&seed) * 2.0f - 1.0f;
    float q3 = get_next_lcg(&seed) * 2.0f - 1.0f;
    float norm = sycl::sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3 + 1e-8f);
    *s = q0 / norm; *x = q1 / norm; *y = q2 / norm; *z = q3 / norm;
}

static __dpct_inline__ void get_deterministic_rotor(int group_idx, float & s, float & p12, float & p13, float & p23) {
    uint32_t x = 42 + (uint32_t)group_idx;
    float v1 = get_next_lcg(&x) * 2.0f - 1.0f;
    float v2 = get_next_lcg(&x) * 2.0f - 1.0f;
    float v3 = get_next_lcg(&x) * 2.0f - 1.0f;
    float angle = get_next_lcg(&x) * 2.0f * 3.14159265f;
    float norm = sycl::sqrt(v1*v1 + v2*v2 + v3*v3 + 1e-8f);
    v1 /= norm; v2 /= norm; v3 /= norm;
    s = sycl::cos(angle/2.0f);
    float sin_ha = sycl::sin(angle/2.0f);
    p12 = sin_ha * v1; p13 = sin_ha * v2; p23 = sin_ha * v3;
}

static __dpct_inline__ void quat_multiply_left(float s, float qx, float qy, float qz, const float *v, float *res) {
    res[0] = s*v[0] - qx*v[1] - qy*v[2] - qz*v[3];
    res[1] = s*v[1] + qx*v[0] + qy*v[3] - qz*v[2];
    res[2] = s*v[2] - qx*v[3] + qy*v[0] + qz*v[1];
    res[3] = s*v[3] + qx*v[2] - qy*v[1] + qz*v[0];
}

static __dpct_inline__ void gp_rotor_mv(float s, float p12, float p13, float p23, const float *x, float *r) {
    r[0] = s*x[0] - p12*x[4] - p13*x[5] - p23*x[6];
    r[1] = s*x[1] + p12*x[2] + p13*x[3] + p23*x[7];
    r[2] = s*x[2] - p12*x[1] + p23*x[3] - p13*x[7];
    r[3] = s*x[3] - p13*x[1] - p23*x[2] + p12*x[7];
    r[4] = s*x[4] + p12*x[0] + p13*x[6] - p23*x[5];
    r[5] = s*x[5] + p13*x[0] - p12*x[6] + p23*x[4];
    r[6] = s*x[6] + p23*x[0] + p12*x[5] - p13*x[4];
    r[7] = s*x[7] - p23*x[1] + p13*x[2] - p12*x[3];
}

static __dpct_inline__ void gp_mv_rotor(const float *x, float s, float p12, float p13, float p23, float *r) {
    r[0] = s*x[0] - p12*x[4] - p13*x[5] - p23*x[6];
    r[1] = s*x[1] - p12*x[2] - p13*x[3] + p23*x[7];
    r[2] = s*x[2] + p12*x[1] - p23*x[3] - p13*x[7];
    r[3] = s*x[3] + p13*x[1] + p23*x[2] + p12*x[7];
    r[4] = s*x[4] + p12*x[0] + p23*x[5] - p13*x[6];
    r[5] = s*x[5] + p13*x[0] - p23*x[4] + p12*x[6];
    r[6] = s*x[6] + p23*x[0] + p13*x[4] - p12*x[5];
    r[7] = s*x[7] + p23*x[1] - p13*x[2] + p12*x[3];
}

static __dpct_inline__ void rotor_sandwich_vec(float s, float p12, float p13, float p23, const float *x_in, float *y_out) {
    float x_mv[8] = {0, x_in[0], x_in[1], x_in[2], 0, 0, 0, 0};
    float tmp[8];
    gp_rotor_mv(s, p12, p13, p23, x_mv, tmp);
    float res[8];
    gp_mv_rotor(tmp, s, -p12, -p13, -p23, res);
    y_out[0] = res[1];
    y_out[1] = res[2];
    y_out[2] = res[3];
}

#endif // GGML_SYCL_ROTOR_MATH_HPP
