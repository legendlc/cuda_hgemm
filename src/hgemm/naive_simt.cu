#include <cuda.h>
#include <cuda_fp16.h>

#include "utils.h"

__global__ void simtNaiveKernel(
    const half * __restrict__ A,
    const half * __restrict__ B,
    half * __restrict__ C,
    size_t M, size_t N, size_t K
)
{
    size_t row = threadIdx.y + blockDim.y * blockIdx.y;
    size_t col = threadIdx.x + blockDim.x * blockIdx.x;

    if (row >= M || col >= N) {
        return;
    }

    float tmp = 0.f;
    #pragma unroll
    for (size_t i = 0; i < K; i++) {
        tmp += __half2float(A[row * K + i]) * __half2float(B[i + col * K]);
    }

    C[row * N + col] = __float2half(tmp);
}

void simtNaive(half *A, half *B, half *C,
    size_t M, size_t N, size_t K)
{
    dim3 block(16, 16);
    dim3 grid(div_ceil(N, block.x), div_ceil(M, block.y));

    simtNaiveKernel<<<grid, block>>>(A, B, C, M, N, K);
}