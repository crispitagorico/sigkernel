import numpy as np
import torch
from sigKer_fast import sig_kernels_fb_gram, sig_kernels_f_gram

class SigLoss(torch.nn.Module):

    def __init__(self, n=0):
        super(SigLoss, self).__init__()
        self.n = n

    def forward(self, X, Y):
        d = SigKernel.apply(X,X,self.n) + SigKernel.apply(Y,Y,self.n) - 2.*SigKernel.apply(X,Y,self.n)
        return torch.mean(d)


class SigKernel(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y=None, n=0):
        """
        input
         - X a list of A paths each of shape (M,D)
         - Y a list of B paths each of shape (N,D)

        computes by solving PDEs (forward) and by variation of parameter (backward)
         -  K_XY: a Gram matrix of pairwise kernel evaluations k(x_i,y_j)      (forward)
         -  K_dXY: ? ( dk_{x_{pq}}(x_i,y_i) )_{p=1,q=1}^{p=M,q=D}  (backward)
        """
        XX, YY, XY = False, False, False

        if Y is None:
            Y = X.detach().copy() # check that X is not detached
            if X.requires_grad:
                XX = True
            else:
                YY = True
        else:
            XY = True

        A = len(X)
        D = X[0].shape[1]
        M = X[0].shape[0]

        # 1. FORWARD
        if XX or XY:
            K, K_rev = sig_kernels_fb_gram(X.detach().numpy(),Y.detach().numpy(),n) 
            K_rev = torch.tensor(K_rev, dtype=torch.double)
        else:
            K =  sig_kernels_f_gram(X.detach().numpy(),Y.detach().numpy(),n) 
        K = torch.tensor(K, dtype=torch.double)
 
        # 2. GRADIENTS
        if XX or XY: # no need to compute this 
            # Need to get the increments of Y on the finer grid
            inc_Y = (Y[:,1:,:]-Y[:,:-1,:])/float(2**n)  #(A,M-1,D)  increments defined by the data
            inc_Y = tile(inc_Y,1,2**n)                  #(A,(2**n)*(M-1),D)  increments on the finer grid

            # Need to reorganize the K_rev matrix
            K_rev_rev = flip(K_rev,dim=1)
            K_rev_rev = flip(K_rev_rev,dim=2)

            KK = (K[:,:-1,:-1] * K_rev_rev[:,1:,1:])                       # (A,(2**n)*(M-1),(2**n)*(N-1))

            K_grad = KK[:,:,:,None]*inc_Y[:,None,:,:]                      # (A,(2**n)*(M-1),(2**n)*(N-1),D)

            K_grad = (1./(2**n))*torch.sum(K_grad,axis=2)                  # (A,(2**n)*(M-1),D)

            K_grad =  torch.sum(K_grad.reshape(A,M-1,2**n,D),axis=2)       # (A,M-1,D)

            ctx.save_for_backward((K_grad,XX,YY,XY))
        
        return K[:,:,-1,-1]  #(A,B)

    @staticmethod
    def backward(ctx, grad_output):

        """
        During the forward pass, the gradients with respect to each increment in each dimension has been computed.
        Here we derive the gradients with respect to the points of the time series.
        """

        grad_incr, XX, YY, XY = ctx.saved_tensors

        if XX or XY:
            A = grad_incr.shape[0]
            D = grad_incr.shape[2]
            grad_points = -torch.cat([grad_incr,torch.zeros((A, 1, D)).type(torch.float64)], dim=1) + torch.cat([torch.zeros((A, 1, D)).type(torch.float64), grad_incr], dim=1)

        if XX:
            return 2*grad_points, None, None
        if YY:
            return None, None, None
        if XY:
            return grad_points, None, None



def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)