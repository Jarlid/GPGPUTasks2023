#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float* results,
                         const unsigned int width, const unsigned int height,
                         const float fromX, const float fromY,
                         const float sizeX, const float sizeY,
                         const unsigned int iters, const int smoothing)
{
    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен
    const unsigned int N = smoothing ? 3 : 1;
    float result_sum = 0.f;

    for (unsigned int anti_aliasing_index = 0; anti_aliasing_index < N * N; ++anti_aliasing_index) {
        unsigned int anti_aliasing_i = anti_aliasing_index % N + 1, anti_aliasing_j = anti_aliasing_index / N + 1;

        float x0 = fromX + (i + 1.f * anti_aliasing_i / (N + 1)) * sizeX / width;
        float y0 = fromY + (j + 1.f * anti_aliasing_j / (N + 1)) * sizeY / height;

        float x = x0;
        float y = y0;

        int iter = 0;
        for (; iter < iters; ++iter) {
            float xPrev = x;
            x = x * x - y * y + x0;
            y = 2.0f * xPrev * y + y0;
            if ((x * x + y * y) > threshold2) {
                break;
            }
        }
        result_sum += iter;
    }

    results[j * width + i] = 1.0f * result_sum / (N * N) / iters;
}
