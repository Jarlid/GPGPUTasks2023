#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void bitonic(__global float *as, const unsigned int n, const unsigned int chunk_size, const unsigned int swap_size) {
    const unsigned int i = get_global_id(0);
    const unsigned int swap_group_num = i / swap_size;

    const unsigned int index = i + swap_group_num * swap_size;\
    const unsigned int chunk_num = index / (2 * chunk_size);

    // if (index + swap_size < n) // добавление этой строки почему-то крашит всю программу, и я не понимаю, почему.
        if ((unsigned int) (as[index] < as[index + swap_size]) == (chunk_num % 2)) {
            float amogus = as[index];
            as[index] = as[index + swap_size];
            as[index + swap_size] = amogus;
        }
}
