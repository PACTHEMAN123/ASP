#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include "utils.h"
#include <cuda_fp16.h>
#include <cstdio>

void sparse_gemv_fp16_launcher(
    half *X, 
    half *W, 
    half *B,
    half *Output, 
    int M, 
    int N
);

void sparse_gemv_fp16_gpu(
    int M,
    int N,
    at::Tensor x_tensor,
    at::Tensor w_tensor, 
    at::Tensor b_tensor,
    at::Tensor output_tensor) 
{
    // CHECK_INPUT(x_tensor);
    // CHECK_INPUT(w_tensor);
    // CHECK_INPUT(output_tensor);

    half *x = reinterpret_cast<half*>(x_tensor.data_ptr<at::Half>());
    half *w = reinterpret_cast<half*>(w_tensor.data_ptr<at::Half>());
    half *b = reinterpret_cast<half*>(b_tensor.data_ptr<at::Half>());
    half *o = reinterpret_cast<half*>(output_tensor.data_ptr<at::Half>());
    // int m = x_tensor.size(1);
    // int n = output_tensor.size(1);

    sparse_gemv_fp16_launcher(x, w, b, o, M, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sparse_gemv_fp16_gpu, "sparse gemv fp16 (CUDA)");
}
