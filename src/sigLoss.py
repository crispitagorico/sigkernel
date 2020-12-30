import torch

from sigKer_torch import SigKernel, SigKernelCuda


# ===========================================================================================================
class SigLoss(torch.nn.Module):

    def __init__(self, n_chunks=1):
        super(SigLoss, self).__init__()
        self.n_chunks = n_chunks
        
    def sig_distance(self,x,y):
        if x.device.type=='cuda':
            d = torch.mean( SigKernelCuda.apply(x,x)+ SigKernelCuda.apply(y,y)- 2.*SigKernelCuda.apply(x,y) )
        else:
            d = torch.mean(SigKernel.apply(x, None)+ SigKernel.apply(y,None)- 2.*SigKernel.apply(x,y) )
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

    def __init__(self, n=0, n_chunks=2, method='explicit'):
        super(SigLoss_naive, self).__init__()
        self.n = n
        self.n_chunks = n_chunks
        self.method = method

    def sig_distance(self,x,y):
        d = torch.mean(SigKernel_naive(x,x,self.n,self.method)+ SigKernel_naive(y,y,self.n,self.method) - 2.*SigKernel_naive(x,y,self.n,self.method)) 
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

# ===========================================================================================================
def SigKernel_naive(X,Y,n=0,method='explicit'):

    A = len(X)
    M = X[0].shape[0]
    N = Y[0].shape[0]

    K_XY = torch.zeros((A, (2**n)*(M-1)+1, (2**n)*(N-1)+1)).type(torch.float64)
    K_XY[:, 0, :] = 1.
    K_XY[:, :, 0] = 1.

    for i in range(0, (2**n)*(M-1)):
        for j in range(0, (2**n)*(N-1)):

            ii = int(i / (2 ** n))
            jj = int(j / (2 ** n))

            inc_X_i = (X[:, ii + 1, :] - X[:, ii, :])/float(2**n)  
            inc_Y_j = (Y[:, jj + 1, :] - Y[:, jj, :])/float(2**n)  

            increment_XY = torch.einsum('ik,ik->i', inc_X_i, inc_Y_j)  

            if method == 'old':
                K_XY[:, i + 1, j + 1] = K_XY[:, i + 1, j].clone() + K_XY[:, i, j + 1].clone() + K_XY[:, i, j].clone()*(increment_XY.clone()-1.)
            elif method == 'explicit':
                K_XY[:, i + 1, j + 1] = ( K_XY[:, i + 1, j].clone() + K_XY[:, i, j + 1].clone() )*(1.+0.5*increment_XY.clone()+(1./12)*increment_XY.clone()**2) - K_XY[:, i, j].clone()*(1.-(1./12)*increment_XY.clone()**2)

    return K_XY[:, -1, -1]
# ===========================================================================================================