import numpy as np
import torch
from sigKer_torch import SigKernelGramMat, SigKernelGramMat_naive

# =========================================================================================================================================
# Signature MMD distance (with gradients -- supports both CPU and GPU.)

# MMD(\mu,\nu) = ||ES(\mu)-ES(\nu)|| = 1/(A*A)\sum_i\sum_j k(x_i, x_j) + 1/(B*B)\sum_i\sum_j k(y_i, y_j) - 2/(A*B)\sum_i\sum_j k(x_i, y_j)
# =========================================================================================================================================
class SigMMD(torch.nn.Module):

    def __init__(self):
        super(SigMMD, self).__init__()

    def forward(self, X, Y):

        assert not Y.requires_grad, "the second input should not require grad"

        K_XX = SigKernelGramMat.apply(X,X,sym=True)
        
        K_YY = SigKernelGramMat.apply(Y,Y,sym=True)

        K_XY = SigKernelGramMat.apply(X,Y,sym=False)

        MMD_squared = torch.mean(K_XX) + torch.mean(K_YY) - 2.*torch.mean(K_XY)
        return MMD_squared


# =========================================================================================================================================
# Naive implementation of MMD distance with gradients ontain via pytorch automatic differentiation (slow, just for testing)
# =========================================================================================================================================
class SigMMD_naive(torch.nn.Module):

    def __init__(self):
        super(SigMMD_naive, self).__init__()

    def forward(self, X, Y):
        K_XX = SigKernelGramMat_naive(X,X)
        K_YY = SigKernelGramMat_naive(Y,Y)    
        K_XY = SigKernelGramMat_naive(X,Y)
        return torch.mean(K_XX)+torch.mean(K_YY)-2.*torch.mean(K_XY)