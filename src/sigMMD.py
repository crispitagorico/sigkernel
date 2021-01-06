import numpy as np
import torch
from sigKer_torch import SigKernelGramMat, SigKernelGramMat_naive

# =========================================================================================================================================
# Signature MMD distance (with gradients -- supports both CPU and GPU.)

# MMD(\mu,\nu) = ||ES(\mu)-ES(\nu)|| = 1/(A*A)\sum_i\sum_j k(x_i, x_j) + 1/(B*B)\sum_i\sum_j k(y_i, y_j) - 2/(A*B)\sum_i\sum_j k(x_i, y_j)
# =========================================================================================================================================
class SigMMD(torch.nn.Module):

    def __init__(self, n=0, solver=0):
        super(SigMMD, self).__init__()
        self.n = n
        self.solver = solver

    def forward(self, X, Y):

        assert not Y.requires_grad, "the second input should not require grad"

        K_XX = SigKernelGramMat.apply(X,X,self.n,self.solver,True)
        K_YY = SigKernelGramMat.apply(Y,Y,self.n,self.solver,True)
        K_XY = SigKernelGramMat.apply(X,Y,self.n,self.solver,False)

        dist = torch.mean(K_XX) + torch.mean(K_YY) - 2.*torch.mean(K_XY)

        return  torch.mean((X[:,0,:]-Y[:,0,:])**2) + dist


# =========================================================================================================================================
# Naive implementation of MMD distance with gradients ontain via pytorch automatic differentiation (slow, just for testing)
# =========================================================================================================================================
class SigMMD_naive(torch.nn.Module):

    def __init__(self, n=0, solver=0):
        super(SigMMD_naive, self).__init__()
        self.n = n
        self.solver = solver

    def forward(self, X, Y):

        K_XX = SigKernelGramMat_naive(X,X,self.n,self.solver)
        K_YY = SigKernelGramMat_naive(Y,Y,self.n,self.solver)    
        K_XY = SigKernelGramMat_naive(X,Y,self.n,self.solver)
        
        dist = torch.mean(K_XX) + torch.mean(K_YY) - 2.*torch.mean(K_XY) 

        return torch.mean((X[:,0,:]-Y[:,0,:])**2) + dist