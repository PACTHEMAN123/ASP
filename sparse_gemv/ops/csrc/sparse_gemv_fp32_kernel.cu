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

// version2: using double pipeline
__global__ void asp_kernel_v2(
    int M, int N,
    float *A, float *X, float *Y
) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    float sum = 0.0f;
    float *A_ptr = A + blockIdx.x * (M * 32) + (32 * M/4) * warp_id + lane_id;
    float *X_ptr = X + M/4 * warp_id + lane_id;
    float X_buf[2][3] = {0};
    float A_buf[2][32] = {0};

    // before pipeline 
    // head stage 0: load x0
    X_buf[0][0] = *X_ptr; X_ptr += 32;
    X_buf[1][0] = *X_ptr; X_ptr += 32;
    // head stage 1: load x1
    X_buf[0][1] = X_buf[0][0]; X_buf[0][0] = *X_ptr; X_ptr += 32;
    X_buf[1][1] = X_buf[1][0]; X_buf[1][0] = *X_ptr; X_ptr += 32;
    // head stage 2: load A
    for (int i = 0; i < 32; i++) {
        float x_load = __shfl_sync(0xffffffff, X_buf[0][1], i);
        if (x_load != 0.0f) A_buf[0][i] = *A_ptr;
        x_load = __shfl_sync(0xffffffff, X_buf[1][1], i);
        if (x_load != 0.0f) A_buf[1][i] = *(A_ptr + 32 * 32);
        A_ptr += 32;
    }
    A_ptr += 32 * 32;

    // main pipeline
    for (int bk = 0; bk < (M/4 - 64*2); bk += 64) {
        // load x
        X_buf[0][2] = X_buf[0][1]; X_buf[0][1] = X_buf[0][0]; X_buf[0][0] = *X_ptr; 
        X_ptr += 32;
        X_buf[1][2] = X_buf[1][1]; X_buf[1][1] = X_buf[1][0]; X_buf[1][0] = *X_ptr; 
        X_ptr += 32;
        // load and compute A
        for (int i = 0; i < 32; i++) {
            // consume buffered A
            float x_calc = __shfl_sync(0xffffffff, X_buf[0][2], i);
            if (x_calc != 0.0f) sum += A_buf[0][i] * x_calc;
            x_calc = __shfl_sync(0xffffffff, X_buf[1][2], i);
            if (x_calc != 0.0f) sum += A_buf[1][i] * x_calc;

            // load new A
            float x_load = __shfl_sync(0xffffffff, X_buf[0][1], i);
            if (x_load != 0.0f) A_buf[0][i] = *A_ptr;
            x_load = __shfl_sync(0xffffffff, X_buf[1][1], i);
            if (x_load != 0.0f) A_buf[1][i] = *(A_ptr + 32 * 32);
            A_ptr += 32;
        }
        A_ptr += 32 * 32;
    }

    // after pipeline 
    // tail stage 0 (no need to load new X)
    X_buf[0][2] = X_buf[0][1]; X_buf[0][1] = X_buf[0][0];
    X_buf[1][2] = X_buf[1][1]; X_buf[1][1] = X_buf[1][0];
    for (int i = 0; i < 32; i++) {
        // consume buffered A
        float x_calc = __shfl_sync(0xffffffff, X_buf[0][2], i);
        if (x_calc != 0.0f) sum += A_buf[0][i] * x_calc;
        x_calc = __shfl_sync(0xffffffff, X_buf[1][2], i);
        if (x_calc != 0.0f) sum += A_buf[1][i] * x_calc;

        // load new A
        float x_load = __shfl_sync(0xffffffff, X_buf[0][1], i);
        if (x_load != 0.0f) A_buf[0][i] = *A_ptr;
        x_load = __shfl_sync(0xffffffff, X_buf[1][1], i);
        if (x_load != 0.0f) A_buf[1][i] = *(A_ptr + 32 * 32);
        A_ptr += 32;
    }
    A_ptr += 32 * 32;

    // tail stage 1 (no need to load new A)
    X_buf[0][2] = X_buf[0][1];
    X_buf[1][2] = X_buf[1][1];
    for (int i = 0; i < 32; i++) {
        float x_calc = __shfl_sync(0xffffffff, X_buf[0][2], i);
        if (x_calc != 0.0f) sum += A_buf[0][i] * x_calc;
        x_calc = __shfl_sync(0xffffffff, X_buf[1][2], i);
        if (x_calc != 0.0f) sum += A_buf[1][i] * x_calc;
    }

    __shared__ float reduce_sum[3][32];
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
    asp_kernel_v2<<<gridSize, blockSize>>>(M, N, W, X, Output);
}