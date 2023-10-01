#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define TILE_SIZE 32

__kernel void matrix_multiplication_naive(__global const float* as, __global const float* bs, __global float* cs,
                                    unsigned int M, unsigned int K, unsigned int N) // Ha-ha, MKN.
{
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    float sum = 0.f;
    for (int k = 0; k < K; ++k) {
        sum += as[j * K + k] * bs[k * N + i];
    }
    cs[j * N + i] = sum;
}

__kernel void matrix_multiplication_local_memory(__global const float* as, __global const float* bs, __global float* cs,
                                          unsigned int M, unsigned int K, unsigned int N) // Ha-ha, MKN.
{
    const unsigned int lid_i = get_local_id(0);
    const unsigned int lid_j = get_local_id(1);
    const unsigned int gid_i = get_global_id(0);
    const unsigned int gid_j = get_global_id(1);

    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    float sum = 0.f;
    for (int k_tile = 0; k_tile * TILE_SIZE < K; ++k_tile) {
        tile_a[lid_j][lid_i] = as[gid_j * K + k_tile * TILE_SIZE + lid_i];
        tile_b[lid_j][lid_i] = bs[(k_tile * TILE_SIZE + lid_j) * N + gid_i];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += tile_a[lid_j][k] * tile_b[k][lid_i];

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    cs[gid_j * N + gid_i] = sum;
}

#define THREAD_WORK 8
#define GAP (TILE_SIZE / THREAD_WORK)

__kernel void matrix_multiplication_MWPT(__global const float* as, __global const float* bs, __global float* cs,
                                                 unsigned int M, unsigned int K, unsigned int N) // Ha-ha, MKN.
{
    const unsigned int lid_i = get_local_id(0);
    const unsigned int lid_j = get_local_id(1);
    const unsigned int gid_i = get_global_id(0);
    const unsigned int gid_j = get_global_id(1);

    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    float sum[THREAD_WORK] = {};

    for (int k_tile = 0; k_tile * TILE_SIZE < K; ++k_tile) {
        for (int t = 0; t < THREAD_WORK; ++t) {
            tile_a[lid_j + t * GAP][lid_i] = as[(gid_j + t * GAP) * K + k_tile * TILE_SIZE + lid_i];
            tile_b[lid_j + t * GAP][lid_i] = bs[(k_tile * TILE_SIZE + lid_j + t * GAP) * N + gid_i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k)
            for (int t = 0; t < THREAD_WORK; ++t)
                sum[t] += tile_a[lid_j + t * GAP][k] * tile_b[k][lid_i];

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int t = 0; t < THREAD_WORK; ++t)
        cs[(gid_j + t * GAP) * N + gid_i] = sum[t];
}