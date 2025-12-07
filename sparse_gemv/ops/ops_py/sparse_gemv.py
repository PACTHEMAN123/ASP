import torch
from torch.autograd import Function
import sparse_gemv_fp32

class SparseGEMV(Function):

    @staticmethod
    def forward(ctx, x, w):
        """sum_double function forward.
        Args:
            array1 (torch.Tensor): [n,]
            array2 (torch.Tensor): [n,]
        
        Returns:
            ans (torch.Tensor): [n,]
        """
        x = x.float()
        w = w.float()
        M = x.shape[1]
        N = w.shape[1]
        ans = x.new_zeros(1, N).to(x.device)

        # transform w into our format
        w_sp = w.view(M, N // 32, 32).permute(1, 0, 2).reshape(N // 32, M * 32).contiguous()

        sparse_gemv_fp32.forward(x.contiguous(), w_sp, ans)
        ctx.mark_non_differentiable(ans) # the function is no need for backpropogation
        return ans

    @staticmethod
    def backward(ctx, g_out):
        return None   # the function is no need for backpropogation
    
sparse_gemv_op = SparseGEMV.apply