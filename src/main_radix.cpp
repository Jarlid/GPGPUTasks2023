#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>
#include <bitset>

// Эти штуки зависят друг от друга, мне было лень делать цикл.
#define OFFSET_SIZE 4
#define OFFSET_VARIANTS 16

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    unsigned int workGroupSize = 128;
    unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

    unsigned int M = global_work_size / workGroupSize, K = OFFSET_VARIANTS;

    unsigned int transpose_work_group_size = OFFSET_VARIANTS;
    unsigned int transpose_global_work_size_x = (K + transpose_work_group_size - 1) / transpose_work_group_size * transpose_work_group_size;
    unsigned int transpose_global_work_size_y = (M + transpose_work_group_size - 1) / transpose_work_group_size * transpose_work_group_size;

    unsigned int prefix_global_work_size = (K * M + workGroupSize - 1) / workGroupSize * workGroupSize;

    gpu::gpu_mem_32u as_gpu, bs_gpu, counter_table_gpu, t_counter_table_gpu, local_prefix_table_gpu;
    as_gpu.resizeN(n);
    bs_gpu.resizeN(n); // Дополнительный буфер для различных целей.

    counter_table_gpu.resizeN(M * K);
    t_counter_table_gpu.resizeN(K * M);
    local_prefix_table_gpu.resizeN(M * K);

    {
        ocl::Kernel merge(radix_kernel, radix_kernel_length, "merge");
        merge.compile();
        ocl::Kernel counter(radix_kernel, radix_kernel_length, "counter");
        counter.compile();
        ocl::Kernel transpose(radix_kernel, radix_kernel_length, "matrix_transpose");
        transpose.compile();
        ocl::Kernel global_prefix(radix_kernel, radix_kernel_length, "prefix_sum");
        global_prefix.compile();
        ocl::Kernel local_prefix(radix_kernel, radix_kernel_length, "local_prefix_sum");
        local_prefix.compile();
        ocl::Kernel radix_reconstruct(radix_kernel, radix_kernel_length, "radix_reconstruct");
        radix_reconstruct.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            for (int offset = 0; offset < 32; offset += OFFSET_SIZE) {

                // Сортировка значений в небольших блоках с помощью mergesort.
                for (unsigned int merge_chunk_size = 1; merge_chunk_size < workGroupSize; merge_chunk_size *= 2) {
                    merge.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, bs_gpu, n, merge_chunk_size, offset);
                    std::swap(as_gpu, bs_gpu);
                }

                counter.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, counter_table_gpu, offset);

                transpose.exec(gpu::WorkSize(transpose_work_group_size, transpose_work_group_size,
                                             transpose_global_work_size_x, transpose_global_work_size_y),
                               counter_table_gpu, t_counter_table_gpu, M, K);

                // Делает из счётчика индикатор количества до.
                for (unsigned int chunk_size = 1; chunk_size < K * M; chunk_size *= 2)
                    global_prefix.exec(gpu::WorkSize(workGroupSize, prefix_global_work_size), t_counter_table_gpu, K * M, chunk_size);

                local_prefix.exec(gpu::WorkSize(workGroupSize, prefix_global_work_size), counter_table_gpu, local_prefix_table_gpu, K * M);

                radix_reconstruct.exec(gpu::WorkSize(workGroupSize, global_work_size),
                                       as_gpu, bs_gpu, t_counter_table_gpu, local_prefix_table_gpu,
                                       n, offset);

                std::swap(as_gpu, bs_gpu);
            }

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
