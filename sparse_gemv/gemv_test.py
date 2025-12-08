# from torch import Tensor, nn
import torch

from ops import sparse_gemv_op
import time
import sparse_gemv_fp32

class Timer:
    def __init__(self, op_name, warmup=100, repeat=5000):
        self.op_name = op_name
        self.warmup = warmup
        self.repeat = repeat

    def __call__(self, fn, *args, **kwargs):
        # Warm-up 不计时
        for _ in range(self.warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(self.repeat):
            fn(*args, **kwargs)
        torch.cuda.synchronize()

        end = time.time()
        avg = (end - start) * 1000 / self.repeat
        print(f"[{self.op_name}] avg latency = {avg:.4f} ms")
        return avg

M = 4096
N = 4096

if __name__ == '__main__':
    x = torch.randn(1, M, device="cuda", dtype=torch.float32)
    w = torch.randn(M, N, device="cuda", dtype=torch.float32)
    w_sp_f32 = w.view(M, N // 32, 32).permute(1, 0, 2).reshape(N // 32, M * 32).contiguous()
    ans = x.new_zeros(1, N).to(x.device)

    # 直接算 ground truth
    expected = torch.matmul(x, w)
    torch.cuda.synchronize()

    # 测 matmul latency
    tm = Timer("matmul")
    tm(torch.matmul, x, w)

    # 测 sparse kernel latency
    ts = Timer("sparse_gemv")
    # ts(sparse_gemv_op, x, w_sp_f32)
    ts(sparse_gemv_fp32.forward, M, N, x, w_sp_f32, ans)

    out = sparse_gemv_op(x, w_sp_f32)  # 最后跑一次拿结果
    diff = (expected - out).abs().max()
    print("max diff:", diff.item())