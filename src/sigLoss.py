import torch
from sigKer_torch import SigKernel, SigKernel_naive


# ===========================================================================================================
# Signature Loss function (with gradients -- supports both CPU and GPU.)

# L(x,y) = ||S(x)-S(y)||
# ===========================================================================================================
class SigLoss(torch.nn.Module):

    def __init__(self, n=0, solver=0, rbf=False, sigma=1., n_chunks=1):
        super(SigLoss, self).__init__()
        self.n = n
        self.solver = solver
        self.rbf = rbf
        self.sigma = sigma
        self.n_chunks = n_chunks
        
    def sig_distance(self,x,y):

        k_xx = SigKernel.apply(x,x,self.n,self.solver,self.rbf,self.sigma)
        k_yy = SigKernel.apply(y,y,self.n,self.solver,self.rbf,self.sigma)
        k_xy = SigKernel.apply(x,y,self.n,self.solver,self.rbf,self.sigma)

        dist = torch.mean(k_xx) + torch.mean(k_yy) - 2.*torch.mean(k_xy) 

        return torch.mean((x[:,0,:]-y[:,0,:])**2) + dist

    def forward(self, X, Y):

        assert not Y.requires_grad, "the second input should not require grad"

        if X.requires_grad:
            assert not self.rbf, 'Current backpropagation method only for linear signature kernel. For rbf signature kernel use naive implementation'


        if self.n_chunks==1:
            return self.sig_distance(X,Y)

        # "dyadic" partitioning
        dist = torch.tensor(0., dtype=torch.float64)
        for k in range(2, self.n_chunks+1):
            X_chunks = torch.chunk(X, k, dim=1)
            Y_chunks = torch.chunk(Y, k, dim=1)
            for x1,x2,y1,y2 in zip(X_chunks[:-1], X_chunks[1:], Y_chunks[:-1], Y_chunks[1:]):
                dist += self.sig_distance(torch.cat([x1,x2],dim=1),torch.cat([y1,y2],dim=1))

        return dist
# ===========================================================================================================


# ===========================================================================================================
# Naive implementation with pytorch auto-diff (slow, just for testing)
# ===========================================================================================================
class SigLoss_naive(torch.nn.Module):

    def __init__(self, n=0, solver=0, rbf=False, sigma=1., n_chunks=1):
        super(SigLoss_naive, self).__init__()
        self.n = n
        self.solver = solver
        self.rbf = rbf
        self.sigma = sigma
        self.n_chunks = n_chunks

    def sig_distance(self,x,y):

        k_xx = SigKernel_naive(x,x,self.n,self.solver,self.rbf,self.sigma)
        k_yy = SigKernel_naive(y,y,self.n,self.solver,self.rbf,self.sigma)
        k_xy = SigKernel_naive(x,y,self.n,self.solver,self.rbf,self.sigma)

        dist = torch.mean(k_xx) + torch.mean(k_yy) - 2.*torch.mean(k_xy)

        return torch.mean((x[:,0,:]-y[:,0,:])**2) + dist

    def forward(self, X, Y):

        if self.n_chunks==1:
            return self.sig_distance(X,Y)

        # "dyadic" partitioning
        dist = torch.tensor(0., dtype=torch.float64)
        for k in range(2, self.n_chunks+1):
            X_chunks = torch.chunk(X, k, dim=1)
            Y_chunks = torch.chunk(Y, k, dim=1)
            for x1,x2,y1,y2 in zip(X_chunks[:-1], X_chunks[1:], Y_chunks[:-1], Y_chunks[1:]):
                dist += self.sig_distance(torch.cat([x1,x2],dim=1),torch.cat([y1,y2],dim=1))

        return dist
# ===========================================================================================================