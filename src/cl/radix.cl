#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

// База:

// Эти штуки зависят друг от друга, мне было лень делать цикл.
#define OFFSET_SIZE 4
#define OFFSET_VARIANTS 16

#define WORKGROUP_SIZE 128

unsigned int trimmed_value(const unsigned int value, const unsigned int offset) {
    return (value >> offset) & (OFFSET_VARIANTS - 1); // Могут получиться значения от 0 до 15.
}

__kernel void counter(__global const unsigned int *as, __global unsigned int *counter_table, const unsigned int offset) {
    const unsigned global_id = get_global_id(0);
    const unsigned local_id = get_local_id(0);
    const unsigned group_id = get_group_id(0);

    __local unsigned int ct_row[OFFSET_VARIANTS];
    if (local_id < OFFSET_VARIANTS)
        ct_row[local_id] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int trimmed_a = trimmed_value(as[global_id], offset);
    atomic_add(ct_row + trimmed_a, 1);

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < OFFSET_VARIANTS)
        counter_table[group_id * OFFSET_VARIANTS + local_id] = ct_row[local_id];
}

__kernel void radix_reconstruct(__global unsigned int* as, __global unsigned int* sorted_as,
                                __global unsigned int* t_global_prefix, __global unsigned int* local_prefix,
                                const unsigned int n, const unsigned int offset) {
    const unsigned global_id = get_global_id(0);
    const unsigned group_id = get_group_id(0);

    if (global_id >= n)
        return;

    unsigned int trimmed_a = trimmed_value(as[global_id], offset);
    unsigned int ta_gid = group_id + (n / WORKGROUP_SIZE) * trimmed_a;

    unsigned int final_place = global_id % WORKGROUP_SIZE;

    if (ta_gid > 0) {
        final_place += t_global_prefix[ta_gid - 1];
    }
    if (trimmed_a > 0) {
        final_place -= local_prefix[group_id * OFFSET_VARIANTS + trimmed_a - 1];
    }

    if (final_place < n)
        sorted_as[final_place] = as[global_id];
}

// Вещи, необходимые для mergesort:

bool comes_before(const unsigned int i, const unsigned int j, __global const unsigned int* as, const unsigned int n, const unsigned int which_half, const unsigned int offset) {
    if (j >= n)
        return true;
    if (which_half == 0)
        return trimmed_value(as[i], offset) <= trimmed_value(as[j], offset);
    return trimmed_value(as[i], offset) < trimmed_value(as[j], offset);
}

__kernel void merge(__global unsigned int* as, __global unsigned int* bs, const unsigned int n, const unsigned int merge_chunk_size, const unsigned int offset)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) return;

    const unsigned int chunk_num = i / (2 * merge_chunk_size);
    const unsigned int which_half = i / merge_chunk_size % 2; // 0 - первая половина, 1 - вторая.
    const unsigned int other_half = (which_half + 1) % 2;

    unsigned int l = -1, r = merge_chunk_size;
    while (r - l > 1) {
        unsigned int m = (l + r) / 2;
        if (comes_before(i, (chunk_num * 2 + other_half) * merge_chunk_size + m, as, n, which_half, offset))
            r = m;
        else
            l = m;
    }

    bs[i + r - which_half * merge_chunk_size] = as[i];
}

// Транспонирование матрицы:

#define TILE_SIZE 16

__kernel void matrix_transpose(__global const unsigned int* as, __global unsigned int* as_t, unsigned int M, unsigned int K)
{
    const unsigned int lid_i = get_local_id(0);
    const unsigned int lid_j = get_local_id(1);
    const unsigned int gid_i = get_global_id(0);
    const unsigned int gid_j = get_global_id(1);

    __local unsigned int tile[TILE_SIZE][TILE_SIZE];
    if (gid_j < M && gid_i < K)
        tile[lid_j][lid_i] = as[gid_j * K + gid_i];

    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int wid_i = get_group_id(0);
    const unsigned int wid_j = get_group_id(1);

    const unsigned int index = (wid_i * TILE_SIZE + lid_j) * M + wid_j * TILE_SIZE + lid_i;
    if ((wid_i * TILE_SIZE + lid_j) < K && wid_j * TILE_SIZE + lid_i < M)
        as_t[index] = tile[lid_i][lid_j];
}

// Высчитывание префиксных сумм:

__kernel void prefix_sum(__global unsigned int *as, const unsigned int n, const unsigned int chunk_size) {
    const unsigned int i = get_global_id(0);

    const unsigned int index = i + (i / chunk_size + 1) * chunk_size;
    if (index > n)
        return;

    const unsigned int addon_index = (2 * (i / chunk_size) + 1) * chunk_size - 1;

    as[index] += as[addon_index];
}

__kernel void local_prefix_sum(__global unsigned int* as, __global unsigned int* local_prefix_table, const unsigned int n) {
    const unsigned int id = get_global_id(0);
    if (id % OFFSET_VARIANTS != 0)
        return;

    unsigned sum = 0;
    for (int i = 0; i < OFFSET_VARIANTS; ++i) {
        if (id + i >= n)
            break;
        sum += as[id + i];
        local_prefix_table[id + i] = sum;
    }
}