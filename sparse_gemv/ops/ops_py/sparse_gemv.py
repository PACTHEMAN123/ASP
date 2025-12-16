import torch
from torch.autograd import Function
import sparse_gemv_fp32

class SparseGEMV(Function):

    @staticmethod
    def forward(ctx, x, w, ans):
        """sum_double function forward.
        Args:
            array1 (torch.Tensor): [n,]
            array2 (torch.Tensor): [n,]
        
        Returns:
            ans (torch.Tensor): [n,]
        """
        M = x.shape[1]
        N = w.shape[0] * 32

        sparse_gemv_fp32.forward(M, N, x, w, ans)
        ctx.mark_non_differentiable(ans) # the function is no need for backpropogation
        return ans

    @staticmethod
    def backward(ctx, g_out):
        return None   # the function is no need for backpropogation
    
class DenseGEMV(Function):
    @staticmethod
    def forward(ctx, x, w, bias):
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

        torch.matmul(x, w, out=ans)
        ctx.mark_non_differentiable(ans) # the function is no need for backpropogation
        return ans

    @staticmethod
    def backward(ctx, g_out):
        return None   # the function is no need for backpropogation
    
sparse_gemv_op = SparseGEMV.apply
dense_gemv_op = DenseGEMV.apply