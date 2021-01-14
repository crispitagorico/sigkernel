import numpy as np
import torch
from sigKer_torch import SigKernelGramMat, SigKernelGramMat_naive

# =========================================================================================================================================
# Signature MMD distance (with gradients -- supports both CPU and GPU.)

# MMD(\mu,\nu) = ||ES(\mu)-ES(\nu)|| = 1/(A*A)\sum_i\sum_j k(x_i, x_j) + 1/(B*B)\sum_i\sum_j k(y_i, y_j) - 2/(A*B)\sum_i\sum_j k(x_i, y_j)
# =========================================================================================================================================
class SigMMD(torch.nn.Module):

    def __init__(self, n=0, solver=0, rbf=False, sigma=1.):
        super(SigMMD, self).__init__()
        self.n = n
        self.solver = solver
        self.rbf = rbf
        self.sigma = sigma

    def forward(self, X, Y):

        assert not Y.requires_grad, "the second input should not require grad"

        if X.requires_grad:
            assert not rbf, 'Current backpropagation method only for linear signature kernel. For rbf signature kernel use naive implementation'


        K_XX = SigKernelGramMat.apply(X,X,self.n,self.solver,True,self.rbf,self.sigma)
        K_YY = SigKernelGramMat.apply(Y,Y,self.n,self.solver,True,self.rbf,self.sigma)
        K_XY = SigKernelGramMat.apply(X,Y,self.n,self.solver,False,self.rbf,self.sigma)

        dist = torch.mean(K_XX) + torch.mean(K_YY) - 2.*torch.mean(K_XY)

        return  torch.mean((X[:,0,:]-Y[:,0,:])**2) + dist


def c_alpha(m, alpha):
    return 4. * np.sqrt(-np.log(alpha) / m)


def hypothesis_test(y_pred, y_test, confidence_level=0.99, n=5, solver=1, rbf=False, sigma=1.):
    """Statistical test based on MMD distance to determine if 
       two sets of paths come from the same distribution.
    """

    m = max(y_pred.shape[0], y_test.shape[0])
    
    dist = SigMMD(n=n, solver=solver, rbf=rbf, sigma=sigma)

    TU = dist(y_pred, y_test)
    
    c = torch.tensor(c_alpha(m, confidence_level), dtype=y_pred.dtype)

    if TU > c:
        print(f'Hypothesis rejected: distribution are not equal with {confidence_level*100}% confidence')
    else:
        print(f'Hypothesis accepted: distribution are equal with {confidence_level*100}% confidence')


# =========================================================================================================================================
# Naive implementation of MMD distance with gradients ontain via pytorch automatic differentiation (slow, just for testing)
# =========================================================================================================================================
class SigMMD_naive(torch.nn.Module):

    def __init__(self, n=0, solver=0, rbf=False, sigma=1.):
        super(SigMMD_naive, self).__init__()
        self.n = n
        self.solver = solver
        self.rbf = rbf
        self.sigma = sigma

    def forward(self, X, Y):

        K_XX = SigKernelGramMat_naive(X,X,self.n,self.solver,self.rbf,self.sigma)
        K_YY = SigKernelGramMat_naive(Y,Y,self.n,self.solver,self.rbf,self.sigma)    
        K_XY = SigKernelGramMat_naive(X,Y,self.n,self.solver,self.rbf,self.sigma)
        
        dist = torch.mean(K_XX) + torch.mean(K_YY) - 2.*torch.mean(K_XY) 

        return torch.mean((X[:,0,:]-Y[:,0,:])**2) + dist