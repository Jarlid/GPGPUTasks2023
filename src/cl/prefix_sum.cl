#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void prefix_sum(__global unsigned int *as, const unsigned int n, const unsigned int chunk_size) {
    const unsigned int i = get_global_id(0);

    const unsigned int index = i + (i / chunk_size + 1) * chunk_size;
    if (index > n)
        return;

    const unsigned int addon_index = (2 * (i / chunk_size) + 1) * chunk_size - 1;

    as[index] += as[addon_index];
}