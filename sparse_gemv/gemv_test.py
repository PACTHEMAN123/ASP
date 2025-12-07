# from torch import Tensor, nn
import torch

from ops import sparse_gemv_op
import time

class Timer:
    def __init__(self, op_name):
        self.begin_time = 0
        self.end_time = 0
        self.op_name = op_name

    def __enter__(self):
        torch.cuda.synchronize()
        self.begin_time = time.time()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        self.end_time = time.time()
        print(f"Average time cost of {self.op_name} is {(self.end_time - self.begin_time) * 1000:.4f} ms")

M = 4096
N = 4096

if __name__ == '__main__':
    x = torch.randn(1, M, device="cuda", dtype=torch.float32)
    w = torch.randn(M, N, device="cuda", dtype=torch.float32)
    
    expected = torch.matmul(x, w)
    torch.cuda.synchronize()
    out = sparse_gemv_op(x, w)
    print(f"out {out} shape {out.shape}")
    print(f"ans {expected} shape {expected.shape}")
    diff = expected - out
    print(diff)