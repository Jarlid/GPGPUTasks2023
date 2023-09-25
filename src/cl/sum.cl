#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define VALUES_PER_WORKITEM 64
#define WORKGROUP_SIZE 128

__kernel void sum(__global unsigned int* sum, __global const unsigned int* as, const unsigned int n) {
    return;
}

__kernel void atomic_sum(__global unsigned int* sum, __global const unsigned int* as, const unsigned int n) {
    const unsigned int index = get_global_id(0);
    if (index >= n)
        return;
    atomic_add(sum, as[index]);
}

__kernel void loop_sum(__global unsigned int* sum, __global const unsigned int* as, const unsigned int n) {
    const unsigned int gid = get_global_id(0);

    int part_sum = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int index = gid * VALUES_PER_WORKITEM + i;
        if (index < n)
            part_sum += as[index];
    }

    atomic_add(sum, part_sum);
}

__kernel void coalesced_loop_sum(__global unsigned int* sum, __global const unsigned int* as, const unsigned int n) {
    const unsigned int lid = get_local_id(0);
    const unsigned int wid = get_group_id(0);
    const unsigned int grs = get_local_size(0);

    unsigned int part_sum = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        unsigned int index = wid * grs * VALUES_PER_WORKITEM + i * grs + lid;
        if (index < n)
            part_sum += as[index];
    }

    atomic_add(sum, part_sum);
}

__kernel void local_memory_sum(__global unsigned int* sum, __global const unsigned int* as, const unsigned int n) {
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);

    __local unsigned int local_as[WORKGROUP_SIZE];
    local_as[lid] = as[gid];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        unsigned int part_sum = 0;
        for (unsigned int index = 0; index < WORKGROUP_SIZE; ++index)
            part_sum += local_as[index];
        atomic_add(sum, part_sum);
    }
}

__kernel void tree_sum(__global unsigned int* sum, __global const unsigned int* as, const unsigned int n) {
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);

    __local unsigned int local_as[WORKGROUP_SIZE];
    local_as[lid] = gid < n ? as[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int n_values = WORKGROUP_SIZE; n_values > 1; n_values /= 2) {
        if (2 * lid < n_values)
            local_as[lid] += local_as[lid + n_values / 2];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
        atomic_add(sum, local_as[0]);
}
