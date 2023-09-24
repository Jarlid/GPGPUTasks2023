#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void sum(__global unsigned int* sum, __global const unsigned int* as, const unsigned int n) {
    return;
}

__kernel void atomic_sum(__global unsigned int* sum, __global const unsigned int* as, const unsigned int n) {
    const unsigned int index = get_global_id(0);
    if (index >= n)
        return;
    atomic_add(sum, as[index]);
}
