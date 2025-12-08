#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include "utils.h"
#include <cstdio>

void sparse_gemv_fp32_launcher(
    float *X, 
    float *W, 
    float *Output, 
    int M, 
    int N
);

void sparse_gemv_fp32_gpu(
    int M,
    int N,
    at::Tensor x_tensor,
    at::Tensor w_tensor, 
    at::Tensor output_tensor) 
{
    // CHECK_INPUT(x_tensor);
    // CHECK_INPUT(w_tensor);
    // CHECK_INPUT(output_tensor);

    float *x = x_tensor.data_ptr<float>();
    float *w = w_tensor.data_ptr<float>();
    float *o = output_tensor.data_ptr<float>();
    // int m = x_tensor.size(1);
    // int n = output_tensor.size(1);

    sparse_gemv_fp32_launcher(x, w, o, M, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sparse_gemv_fp32_gpu, "sparse gemv (CUDA)");
}
