from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os
# os.environ['TORCH_CUDA_ARCH_LIST'] = "8.0;8.6;9.0;10.0;12.0"

cc_flag = []
# cc_flag += ["-gencode", "arch=compute_80,code=sm_80"]
cc_flag += ["-gencode", "arch=compute_120,code=sm_120"]

setup(
    name='sparse_gemv',
    packages=find_packages(),
    version='0.1.0',
    author='Jiayu Xiao',
    ext_modules=[
        CUDAExtension(
            'sparse_gemv_fp32', # operator name
            ['./ops/csrc/sparse_gemv_fp32.cpp',
             './ops/csrc/sparse_gemv_fp32_kernel.cu',
            ],
            extra_compile_args={
                "nvcc": cc_flag
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)