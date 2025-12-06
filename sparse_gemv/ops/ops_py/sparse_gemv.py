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
        ans = x.new_zeros(x.shape[0], w.shape[1]).to(x.device)
        sparse_gemv_fp32.forward(x.contiguous(), w.contiguous(), ans)

        print(f"ans shape {ans.shape}, sum {ans.sum(dim=-1)}")

        ctx.mark_non_differentiable(ans) # the function is no need for backpropogation
        return ans

    @staticmethod
    def backward(ctx, g_out):
        return None   # the function is no need for backpropogation
    
sparse_gemv_op = SparseGEMV.apply