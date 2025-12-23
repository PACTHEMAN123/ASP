#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "utils.h"

#define THREADS_PRE_BLOCK 128
#define WARP_SIZE 32

// write a fp16 sparse gemv kernel



__global__ void asp_kernel_v2(
    int M, int N,
    half *A, half *X, half *Y, half *B
) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    const half ZERO = CUDART_ZERO_FP16;
    half sum = ZERO;
    half *A_ptr = A + blockIdx.x * (M * 32) + (32 * M/4) * warp_id + lane_id;
    half *X_ptr = X + M/4 * warp_id + lane_id;
    half X_buf[2][3] = {ZERO};
    half A_buf[2][32] = {ZERO};

    // before pipeline 
    // head stage 0: load x0
    X_buf[0][0] = *X_ptr; X_ptr += 32;
    X_buf[1][0] = *X_ptr; X_ptr += 32;
    // head stage 1: load x1
    X_buf[0][1] = X_buf[0][0]; X_buf[0][0] = *X_ptr; X_ptr += 32;
    X_buf[1][1] = X_buf[1][0]; X_buf[1][0] = *X_ptr; X_ptr += 32;
    // head stage 2: load A
    for (int i = 0; i < 32; i++) {
        half x_load = __shfl_sync(0xffffffff, X_buf[0][1], i);
        if (!__heq(x_load, ZERO)) A_buf[0][i] = *A_ptr;
        x_load = __shfl_sync(0xffffffff, X_buf[1][1], i);
        if (!__heq(x_load, ZERO)) A_buf[1][i] = *(A_ptr + 32 * 32);
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
            half x_calc = __shfl_sync(0xffffffff, X_buf[0][2], i);
            if (!__heq(x_calc, ZERO)) sum = __hfma(A_buf[0][i], x_calc, sum);
            x_calc = __shfl_sync(0xffffffff, X_buf[1][2], i);
            if (!__heq(x_calc, ZERO)) sum = __hfma(A_buf[1][i], x_calc, sum);

            // load new A
            half x_load = __shfl_sync(0xffffffff, X_buf[0][1], i);
            if (!__heq(x_load, ZERO)) A_buf[0][i] = *A_ptr;
            x_load = __shfl_sync(0xffffffff, X_buf[1][1], i);
            if (!__heq(x_load, ZERO)) A_buf[1][i] = *(A_ptr + 32 * 32);
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
        half x_calc = __shfl_sync(0xffffffff, X_buf[0][2], i);
        if (!__heq(x_calc, ZERO)) sum = __hfma(A_buf[0][i], x_calc, sum);
        x_calc = __shfl_sync(0xffffffff, X_buf[1][2], i);
        if (!__heq(x_calc, ZERO)) sum = __hfma(A_buf[1][i], x_calc, sum);

        // load new A
        half x_load = __shfl_sync(0xffffffff, X_buf[0][1], i);
        if (!__heq(x_load, ZERO)) A_buf[0][i] = *A_ptr;
        x_load = __shfl_sync(0xffffffff, X_buf[1][1], i);
        if (!__heq(x_load, ZERO)) A_buf[1][i] = *(A_ptr + 32 * 32);
        A_ptr += 32;
    }
    A_ptr += 32 * 32;

    // tail stage 1 (no need to load new A)
    X_buf[0][2] = X_buf[0][1];
    X_buf[1][2] = X_buf[1][1];
    for (int i = 0; i < 32; i++) {
        half x_calc = __shfl_sync(0xffffffff, X_buf[0][2], i);
        if (!__heq(x_calc, ZERO)) sum = __hfma(A_buf[0][i], x_calc, sum);
        x_calc = __shfl_sync(0xffffffff, X_buf[1][2], i);
        if (!__heq(x_calc, ZERO)) sum = __hfma(A_buf[1][i], x_calc, sum);
    }

    __shared__ half reduce_sum[3][32];
    if (warp_id != 0)
        reduce_sum[warp_id - 1][lane_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        for (int i = 0;i < 3; i++){
            sum = __hadd(reduce_sum[i][lane_id], sum);
        }
        Y[blockIdx.x * 32 + lane_id] = __hadd(sum, B[blockIdx.x * 32 + lane_id]);
    }
}

void sparse_gemv_fp16_launcher(
    half *X, 
    half *W, 
    half *B,
    half *Output, 
    int M, 
    int N)
{
    dim3 blockSize(THREADS_PRE_BLOCK, 1, 1);
    dim3 gridSize(N / WARP_SIZE, 1, 1);
    asp_kernel_v2<<<gridSize, blockSize>>>(M, N, W, X, Output, B);
}