import torch
from sigKer_torch import SigKernel, SigKernel_naive


# ===========================================================================================================
# Signature Loss function (with gradients -- supports both CPU and GPU.)

# L(x,y) = ||S(x)-S(y)||
# ===========================================================================================================
class SigLoss(torch.nn.Module):

    def __init__(self, n_chunks=1, solver=0):
        super(SigLoss, self).__init__()
        self.n_chunks = n_chunks
        self.solver=solver
        
    def sig_distance(self,x,y):
        d = torch.mean( SigKernel.apply(x,x,self.solver)+ SigKernel.apply(y,y,self.solver)- 2.*SigKernel.apply(x,y,self.solver) )
        return d #+ torch.mean((x[:,0,:]-y[:,0,:])**2) #+ torch.mean(torch.abs(x[:,-1,:]-y[:,-1,:]))

    def forward(self, X, Y):

        assert not Y.requires_grad, "the second input should not require grad"

        if self.n_chunks==1:
            return self.sig_distance(X,Y)

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

    def __init__(self, n_chunks=1):
        super(SigLoss_naive, self).__init__()
        self.n_chunks = n_chunks

    def sig_distance(self,x,y):
        d = torch.mean( SigKernel_naive(x,x)+ SigKernel_naive(y,y) - 2.*SigKernel_naive(x,y) ) 
        return d #+ torch.mean(torch.abs(x[:,0,:]-y[:,0,:])) + torch.mean(torch.abs(x[:,-1,:]-y[:,-1,:]))

    def forward(self, X, Y):

        if self.n_chunks==1:
            return self.sig_distance(X,Y)

        dist = torch.tensor(0., dtype=torch.float64)
        for k in range(2, self.n_chunks+1):
            X_chunks = torch.chunk(X, k, dim=1)
            Y_chunks = torch.chunk(Y, k, dim=1)
            for x1,x2,y1,y2 in zip(X_chunks[:-1], X_chunks[1:], Y_chunks[:-1], Y_chunks[1:]):
                dist += self.sig_distance(torch.cat([x1,x2],dim=1),torch.cat([y1,y2],dim=1))

        return dist
# ===========================================================================================================