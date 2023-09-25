#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"

#include "cl/sum_cl.h"

#define VALUES_PER_WORKITEM 64


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        // TODO: implement on OpenCL
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        gpu::gpu_mem_32u as_gpu, sum_gpu;
        as_gpu.resizeN(n);
        sum_gpu.resizeN(1);

        as_gpu.writeN(as.data(), n);
        unsigned int zero = 0, sum;

        std::vector<std::tuple<std::string, std::string, unsigned int>> modes = {
                {"GPU atomic", "atomic_sum", n},
                {"GPU loop", "loop_sum", (n + VALUES_PER_WORKITEM - 1) / VALUES_PER_WORKITEM},
                {"GPU coalesced loop", "coalesced_loop_sum", (n + VALUES_PER_WORKITEM - 1) / VALUES_PER_WORKITEM},
                {"GPU local memory", "local_memory_sum", n},
                {"GPU tree", "tree_sum", n}
        };

        for (const auto& mode : modes) {
            std::string print_name, kernel_name;
            unsigned int local_n;
            std::tie(print_name, kernel_name, local_n) = mode;

            ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernel_name);
            kernel.compile();

            unsigned int workGroupSize = 128;
            unsigned int global_work_size = (local_n + workGroupSize - 1) / workGroupSize * workGroupSize;

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                sum_gpu.writeN(&zero, 1);
                kernel.exec(gpu::WorkSize(workGroupSize, global_work_size),
                            sum_gpu, as_gpu, n);
                sum_gpu.readN(&sum, 1);

                EXPECT_THE_SAME(reference_sum, sum, print_name + " result should be consistent!");
                t.nextLap();
            }

            std::cout << print_name << ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << print_name << ": " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
