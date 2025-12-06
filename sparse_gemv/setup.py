from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os
os.environ['TORCH_CUDA_ARCH_LIST'] = "8.0;8.6;9.0;10.0;12.0"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

setup(
    name='sparse_gemv',
    packages=find_packages(),
    version='0.1.0',
    author='Jiayu Xiao',
    ext_modules=[
        CUDAExtension(
            'sparse_gemv_fp32', # operator name
            ['./ops/csrc/sparse_gemv_fp32.cpp',
             './ops/csrc/sparse_gemv_fp32_kernel.cu',]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)