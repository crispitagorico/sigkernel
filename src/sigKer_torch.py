import numpy as np
import torch
from sigKer_fast import sig_kernel_batch



class SigLoss(torch.nn.Module):

    def __init__(self, n=0, n_chunks=2):
        super(SigLoss, self).__init__()
        self.n = n
        self.n_chunks = n_chunks

    def sig_distance(self,x,y):
        d = torch.mean(SigKernel.apply(x,None,self.n) + SigKernel.apply(y,None,self.n) - 2.*SigKernel.apply(x,y,self.n))
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

#class SigLoss(torch.nn.Module):

#    def __init__(self, n=0):
#        super(SigLoss, self).__init__()
#        self.n = n

#    def forward(self, X, Y):
#        dist = SigKernel.apply(X,None,self.n) + SigKernel.apply(Y,None,self.n) - 2.*SigKernel.apply(X,Y,self.n)
#        return torch.mean(torch.sqrt(dist)) + torch.mean(torch.abs(X[:,0,:] - Y[:,0,:])) + torch.mean(torch.abs(X[:,-1,:] - Y[:,-1,:]))


class SigKernel(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y=None, n=0):
        """
        input
         - X a list of A paths each of shape (M,D)
         - Y a list of A paths each of shape (N,D)

        computes by solving PDEs (forward) and by variation of parameter (backward)
         -  K_XY: a vector of A pairwise kernel evaluations k(x_i,y_i)      (forward)
         -  K_dXY: A matrices ( dk_{x_{pq}}(x_i,y_i) )_{p=1,q=1}^{p=M,q=D}  (backward)
        """
        XX, YY, XY = False, False, False

        if Y is None:
            Y = X.detach().clone() 
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
            K, K_rev = sig_kernel_batch(X.detach().cpu().numpy(),Y.detach().cpu().numpy(),n,gradients=True) 
            K_rev = torch.tensor(K_rev, dtype=torch.double).to(X.device)
        else:
            K =  sig_kernel_batch(X.detach().cpu().numpy(),Y.detach().cpu().numpy(),n,gradients=False) 
        K = torch.tensor(K, dtype=torch.double).to(X.device)

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

            ctx.save_for_backward(K_grad)
        
        ctx.XX, ctx.YY, ctx.XY = XX, YY, XY

        return K[:,-1,-1]

    @staticmethod
    def backward(ctx, grad_output):

        """
        During the forward pass, the gradients with respect to each increment in each dimension has been computed.
        Here we derive the gradients with respect to the points of the time series.
        """

        XX, YY, XY = ctx.XX, ctx.YY, ctx.XY

        if XX or XY:
            grad_incr, = ctx.saved_tensors
            A = grad_incr.shape[0]
            D = grad_incr.shape[2]
            grad_points = -torch.cat([grad_incr,torch.zeros((A, 1, D)).type(torch.float64).to(grad_incr.device)], dim=1) + torch.cat([torch.zeros((A, 1, D)).type(torch.float64).to(grad_incr.device), grad_incr], dim=1)

        if XX:
            # remark1: grad_points=\sum_a dKa/dX, whilst dL/dX = \sum_a grad_output[a]*dKa/dX
            # where dKa/dX is a tensor of shape (A,M,N) with zeros everywhere except for Ka[a,:,:].
            # we need to 'inject grad_output' in grad_points, it corresponds to do grad_output[a]*grad_points[a,:,:]
            # remark2: KXX is bilinear, and grad_points is the gradient with respect to the left variable -> we need to multiply by 2
            return 2.*grad_output[:,None,None]*grad_points, None, None  
        if YY:
            return None, None, None
        if XY:
            # see remark 1
            return grad_output[:,None,None]*grad_points, None, None


# Naive implementation with pytorch auto-diff (slow)

class SigLoss_naive(torch.nn.Module):

    def __init__(self, n=0, n_chunks=2):
        super(SigLoss_naive, self).__init__()
        self.n = n
        self.n_chunks = n_chunks

    def sig_distance(self,x,y):
        d = torch.mean(torch.sqrt(SigKernel_naive(x,x,self.n) + SigKernel_naive(y,y,self.n) - 2.*SigKernel_naive(x,y,self.n))) 
        return d #+ torch.mean(torch.abs(x[:,0,:]-y[:,0,:])) + torch.mean(torch.abs(x[:,-1,:]-y[:,-1,:]))

    def forward(self, X, Y):

        if self.n_chunks==1:
            return self.sig_distance(X,Y)

        dist = torch.tensor(0., dtype=torch.float64)
        for k in range(2, self.n_chunks):
            X_chunks = torch.chunk(X, k, dim=1)
            Y_chunks = torch.chunk(Y, k, dim=1)
            for x1,x2,y1,y2 in zip(X_chunks[:-1], X_chunks[1:], Y_chunks[:-1], Y_chunks[1:]):
                dist += self.sig_distance(torch.cat([x1,x2],dim=1),torch.cat([y1,y2],dim=1))

        return dist

#class SigLoss_naive(torch.nn.Module):

#    def __init__(self, n=0):
#        super(SigLoss_naive, self).__init__()
#        self.n = n

#    def forward(self, X, Y):
#        d = SigKernel_naive(X,X,self.n) + SigKernel_naive(Y,Y,self.n) - 2.*SigKernel_naive(X,Y,self.n)
#        return torch.mean(d)


def SigKernel_naive(X,Y,n=0):
    """
    input
     - X a list of A paths each of shape (M,D)
     - Y a list of A paths each of shape (N,D)

    computes by solving PDEs (forward)
     -  K_XY: a vector of A pairwise kernel evaluations k(x_i,y_i)      (forward)
    """
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

            inc_X_i = (X[:, ii + 1, :] - X[:, ii, :])/float(2**n)  # (A,D)
            inc_Y_j = (Y[:, jj + 1, :] - Y[:, jj, :])/float(2**n)  # (A,D)

            increment_XY = torch.einsum('ik,ik->i', inc_X_i, inc_Y_j)  # (A) <-> A dots prod bwn R^D and R^D

            # implicit scheme 
            K_XY[:, i + 1, j + 1] = K_XY[:, i + 1, j].clone() + K_XY[:, i, j + 1].clone() - K_XY[:, i, j].clone() + ( K_XY[:, i + 1, j].clone() + K_XY[:, i, j + 1].clone() )*0.5*increment_XY.clone()*((1.-0.25*increment_XY.clone())**(-1))

            # explicit scheme
            #K_XY[:, i + 1, j + 1] = ( K_XY[:, i + 1, j].clone() + K_XY[:, i, j + 1].clone() )*(1.+0.5*increment_XY.clone()+(1./12)*increment_XY.clone()**2) - K_XY[:, i, j].clone()*(1.-(1./12)*increment_XY.clone()**2)

    return K_XY[:, -1, -1]


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(a.device)
    return torch.index_select(a, dim, order_index)


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)
