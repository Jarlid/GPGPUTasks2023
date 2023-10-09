#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

bool comes_before(const unsigned int i, const unsigned int j, __global const float* as, const unsigned int n, const unsigned int which_half) {
    if (j >= n)
        return true;
    if (which_half == 0)
        return as[i] <= as[j];
    return as[i] < as[j];
}

__kernel void merge(__global float* as, __global float* bs, const unsigned int n, const unsigned int merge_chunk_size)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) return;

    const unsigned int chunk_num = i / (2 * merge_chunk_size);
    const unsigned int which_half = i / merge_chunk_size % 2; // 0 - первая половина, 1 - вторая.
    const unsigned int other_half = (which_half + 1) % 2;

    unsigned int l = -1, r = merge_chunk_size;
    while (r - l > 1) {
        unsigned int m = (l + r) / 2;
        if (comes_before(i, (chunk_num * 2 + other_half) * merge_chunk_size + m, as, n, which_half))
            r = m;
        else
            l = m;
    }

    bs[i + r - which_half * merge_chunk_size] = as[i];
}
