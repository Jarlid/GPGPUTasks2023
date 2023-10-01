#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define TILE_SIZE 32

__kernel void matrix_transpose(__global const float* as, __global float* as_t, unsigned int M, unsigned int K)
{
    const unsigned int lid_i = get_local_id(0);
    const unsigned int lid_j = get_local_id(1);
    const unsigned int gid_i = get_global_id(0);
    const unsigned int gid_j = get_global_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE];
    tile[lid_i][lid_j] = as[gid_j * K + gid_i];

    barrier(CLK_LOCAL_MEM_FENCE);

    as_t[gid_i * M + gid_j] = tile[lid_i][lid_j];
}