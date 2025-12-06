#include <cstdio>
#include <cuda_runtime.h>
#include "utils.h"

#define THREADS_PRE_BLOCK 128
#define WARP_SIZE 32

__global__ void asp_kernel_v0(
    int M, 
    int N,
    float *A, 
    float *X, 
    float *Y
) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    float sum = 0.0f;
    float *A_ptr = A + blockIdx.x * (M * 32) + (32 * M/4) * warp_id + lane_id;
    float *X_ptr = X + M/4 * warp_id + lane_id;

    for (int bk = 0; bk < M / 4; bk += 32) {
        // load x
        float x = *X_ptr;

        for (int i = 0; i < 32; i++) {
            // shuffle current x
            float cur_x = __shfl_sync(0xffffffff, x, i);
            if (cur_x != 0.0f)
                sum += *A_ptr * cur_x;
            A_ptr += 32;
        }
        X_ptr += 32;
    }

    __shared__ float reduce_sum[3][WARP_SIZE];
    if (warp_id != 0)
        reduce_sum[warp_id - 1][lane_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        for (int i = 0;i < 3; i++){
            sum += reduce_sum[i][lane_id];
        }
        Y[blockIdx.x * 32 + lane_id] = sum;
    }
}

__global__ void dummy_kernel() {
    if (threadIdx.x == 0)
        printf("hello world\n");
}

void sparse_gemv_fp32_launcher(
    float *X, 
    float *W, 
    float *Output, 
    int M, 
    int N)
{
    dim3 blockSize(THREADS_PRE_BLOCK, 1, 1);
    dim3 gridSize(N / WARP_SIZE, 1, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    asp_kernel_v0<<<gridSize, blockSize>>>(M, N, W, X, Output);
    // dummy_kernel<<<gridSize, blockSize>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
}