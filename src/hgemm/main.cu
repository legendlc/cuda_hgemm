
#include "tester.h"

#define HGEMM_FUNC(name) void name(half *A, half *B, half *C, size_t M, size_t N, size_t K)

HGEMM_FUNC(simtNaive);

int main()
{
    size_t FLAGS_M = 4096;
    size_t FLAGS_K = 4096;
    size_t FLAGS_N = 4096;
    int FLAGS_warmup_iterations = 1;
    int FLAGS_profiling_iterations = 10;
    int FLAGS_sleep_duration = 100;
    bool FLAGS_enable_check = true;

    Tester tester(
        FLAGS_M, FLAGS_N, FLAGS_K,
        FLAGS_warmup_iterations,
        FLAGS_profiling_iterations,
        FLAGS_sleep_duration,
        FLAGS_enable_check
    );

    tester.evaluate(simtNaive, "Simt-Naive");

    return 0;
}