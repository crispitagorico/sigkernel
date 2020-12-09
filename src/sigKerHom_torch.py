import numpy as np
import torch
from sigKer_fast import sig_kernel_batch, sig_kernel_batch_

class SigLossHom(torch.nn.Module):

    def __init__(self, n=0, n_chunks=2, method='variation_parameters'):
        super(SigLossHom, self).__init__()
        self.n = n
        self.n_chunks = n_chunks
        self.method=method

    def sig_distance(self,x,y):
        d = torch.mean(torch.sqrt(SigKernelHom.apply(x,y,self.n,self.method)-1)) 
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


class SigKernelHom(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y, n=0, method='variation_parameters'):
        
        A = len(X)
        D = X.shape[2]
        M = X.shape[1]
        N = Y.shape[1]

        # 1. FORWARD
        Y_rev = flip(Y,1)
        Z = torch.cat([X,Y_rev], dim=1)
        if method=='variation_parameters':
            K, K_rev = sig_kernel_batch(Z.detach().cpu().numpy(),Z.detach().cpu().numpy(),n,implicit=False,gradients=True)
        else:
            K, K_rev = sig_kernel_batch_(Z.detach().cpu().numpy(),Z.detach().cpu().numpy(),n,implicit=False,gradients=True)
        K_rev = torch.tensor(K_rev, dtype=torch.double).to(X.device)
        K = torch.tensor(K, dtype=torch.double).to(X.device)

        if method=='variation_parameters':
            # 2. GRADIENTS
            # Need to get the increments of Y on the finer grid
            inc = (Z[:,1:,:]-Z[:,:-1,:])/float(2**n)  #(A,M-1,D)  increments defined by the data
            inc = tile(inc,1,2**n)                  #(A,(2**n)*(M-1),D)  increments on the finer grid

            # Need to reorganize the K_rev matrix
            K_rev_rev = flip(K_rev,dim=1)
            K_rev_rev = flip(K_rev_rev,dim=2)

            KK = (K[:,:-1,:-1] * K_rev_rev[:,1:,1:])                       # (A,(2**n)*(M-1),(2**n)*(N-1))

            K_grad = KK[:,:,:,None]*inc[:,None,:,:]                      # (A,(2**n)*(M-1),(2**n)*(N-1),D)

            K_grad = (1./(2**n))*torch.sum(K_grad,axis=2)                  # (A,(2**n)*(M-1),D)

            K_grad =  torch.sum(K_grad.reshape(A,M+N-1,2**n,D),axis=2)       # (A,M-1,D)

            ctx.save_for_backward(K_grad)

        else:
            ctx.save_for_backward(K_rev[:,:,:,-1,-1])
        
        ctx.M = M

        return K[:,-1,-1]

    @staticmethod
    def backward(ctx, grad_output):

        """
        During the forward pass, the gradients with respect to each increment in each dimension has been computed.
        Here we derive the gradients with respect to the points of the time series.
        """

        M = ctx.M

        grad_incr, = ctx.saved_tensors
        A = grad_incr.shape[0]
        D = grad_incr.shape[2]
        grad_points = -torch.cat([grad_incr,torch.zeros((A, 1, D)).type(torch.float64).to(grad_incr.device)], dim=1) + torch.cat([torch.zeros((A, 1, D)).type(torch.float64).to(grad_incr.device), grad_incr], dim=1)
            
        # remark1: grad_points=\sum_a dKa/dX, whilst dL/dX = \sum_a grad_output[a]*dKa/dX
        # where dKa/dX is a tensor of shape (A,M,N) with zeros everywhere except for Ka[a,:,:].
        # we need to 'inject grad_output' in grad_points, it corresponds to do grad_output[a]*grad_points[a,:,:]
        # remark2: KXX is bilinear, and grad_points is the gradient with respect to the left variable -> we need to multiply by 2
        return 2.*grad_output[:,None,None]*grad_points[:,:M,:], None, None, None 



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
