import numpy as np
import torch
import torch.cuda
from numba import cuda

from sigKer_fast import sig_kernel_batch_varpar, sig_kernel_Gram_matrix
from sigKer_cuda import compute_sig_kernel_batch_varpar_from_increments_cuda, compute_sig_kernel_Gram_mat_varpar_from_increments_cuda

class LinearKernel():
    """Linear kernel k: R^d x R^d -> R"""

    def batch_kernel(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch, length_X, dim),
                  - Y: torch tensor of shape (batch, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^i_t) of shape (batch, length_X, length_Y)
        """
        return torch.bmm(X, Y.permute(0,2,1))


class RBFKernel():
    """RBF kernel k: R^d x R^d -> R"""

    def __init__(self, sigma):
        self.sigma = sigma

    def batch_kernel(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch, length_X, dim),
                  - Y: torch tensor of shape (batch, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^i_t) of shape (batch, length_X, length_Y)
        """
        A = X.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        Xs = torch.sum(X**2, dim=2)
        Ys = torch.sum(Y**2, dim=2)
        dist = -2.*torch.bmm(X, Y.permute(0,2,1))
        dist += torch.reshape(Xs,(A,M,1)) + torch.reshape(Ys,(A,1,N))
        return torch.exp(-dist/self.sigma)


class _SigKernel(torch.autograd.Function):
    """Wrapper for signature kernel k_sig(x,y) = <S(f(x)),S(f(y))> where k(x,y) = <f(x),f(y)> is a given static kernel"""
 
    @staticmethod
    def forward(ctx, X, Y, static_kernel, dyadic_order):

        A = X.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        MM = (2**dyadic_order)*(M-1)
        NN = (2**dyadic_order)*(N-1)

        # computing dsdt k(X^i_s,Y^i_t)
        G_static = static_kernel.batch_kernel(X,Y)
        G_static_ = G_static[:,1:,1:] + G_static[:,:-1,:-1] - G_static[:,1:,:-1] - G_static[:,:-1,1:] 
        G_static_ = tile(tile(G_static_,1,2**dyadic_order)/float(2**dyadic_order),2,2**dyadic_order)/float(2**dyadic_order)

        # if on GPU
        if X.device.type=='cuda':

            assert max(MM+1,NN+1) < 1024, 'n must be lowered or data must be moved to CPU as the current choice of n makes exceed the thread limit'
            
            # cuda parameters
            threads_per_block = max(MM+1,NN+1)
            n_anti_diagonals = 2 * threads_per_block - 1

            # Prepare the tensor of output solutions to the PDE (forward)
            K = torch.zeros((A, MM+2, NN+2), device=G_static.device, dtype=G_static.dtype) 
            K[:,0,:] = 1.
            K[:,:,0] = 1. 

            # Compute the forward signature kernel
            compute_sig_kernel_batch_varpar_from_increments_cuda[A, threads_per_block](cuda.as_cuda_array(G_static_.detach()),
                                                                                       MM+1, NN+1, n_anti_diagonals,
                                                                                       cuda.as_cuda_array(K))
            K = K[:,:-1,:-1]

        # if on CPU
        else:
            K = torch.tensor(sig_kernel_batch_varpar(G_static_.detach().numpy()))

        ctx.save_for_backward(X,Y,G_static,K)
        ctx.static_kernel = static_kernel
        ctx.dyadic_order = dyadic_order

        return K[:,-1,-1]


    @staticmethod
    def backward(ctx, grad_output):
    
        X, Y, G_static, K = ctx.saved_tensors
        static_kernel = ctx.static_kernel
        dyadic_order = ctx.dyadic_order

        G_static_ = G_static[:,1:,1:] + G_static[:,:-1,:-1] - G_static[:,1:,:-1] - G_static[:,:-1,1:] 
        G_static_ = tile(tile(G_static_,1,2**dyadic_order)/float(2**dyadic_order),2,2**dyadic_order)/float(2**dyadic_order)

        A = X.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        MM = (2**dyadic_order)*(M-1)
        NN = (2**dyadic_order)*(N-1)
            
        # Reverse paths
        X_rev = torch.flip(X, dims=[1])
        Y_rev = torch.flip(Y, dims=[1])

        # computing dsdt k(X_rev^i_s,Y_rev^i_t) for variation of parameters
        G_static_rev = flip(flip(G_static_,dim=1),dim=2)

        # if on GPU
        if X.device.type=='cuda':

            # Prepare the tensor of output solutions to the PDE (backward)
            K_rev = torch.zeros((A, MM+2, NN+2), device=G_static_rev.device, dtype=G_static_rev.dtype) 
            K_rev[:,0,:] = 1.
            K_rev[:,:,0] = 1. 

            # cuda parameters
            threads_per_block = max(MM,NN)
            n_anti_diagonals = 2 * threads_per_block - 1

            # Compute signature kernel for reversed paths
            compute_sig_kernel_batch_varpar_from_increments_cuda[A, threads_per_block](cuda.as_cuda_array(G_static_rev.detach()), 
                                                                                       MM+1, NN+1, n_anti_diagonals,
                                                                                       cuda.as_cuda_array(K_rev))
            K_rev = K_rev[:,:-1,:-1]      

        # if on CPU
        else:
            K_rev = torch.tensor(sig_kernel_batch_varpar(G_static_rev.detach().numpy()))

        K_rev = flip(flip(K_rev,dim=1),dim=2)
        KK = K[:,:-1,:-1] * K_rev[:,1:,1:]     

        # finite difference step 
        h = 1e-9

        Xh = X[:,:,:,None] + h*torch.eye(D)[None,None,:]  
        Xh = Xh.permute(0,1,3,2)
        Xh = Xh.reshape(A,M*D,D)

        G_h = static_kernel.batch_kernel(Xh,Y) 
        G_h = G_h.reshape(A,M,D,N)
        G_h = G_h.permute(0,1,3,2) 

        Diff_1 = G_h[:,1:,1:,:] - G_h[:,1:,:-1,:] - (G_static[:,1:,1:])[:,:,:,None] + (G_static[:,1:,:-1])[:,:,:,None]
        Diff_1 =  tile( tile(Diff_1,1,2**dyadic_order)/float(2**dyadic_order),2, 2**dyadic_order)/float(2**dyadic_order)  
        Diff_2 = G_h[:,1:,1:,:] - G_h[:,1:,:-1,:] - (G_static[:,1:,1:])[:,:,:,None] + (G_static[:,1:,:-1])[:,:,:,None]
        Diff_2 += - G_h[:,:-1,1:,:] + G_h[:,:-1,:-1,:] + (G_static[:,:-1,1:])[:,:,:,None] - (G_static[:,:-1,:-1])[:,:,:,None]
        Diff_2 = tile(tile(Diff_2,1,2**dyadic_order)/float(2**dyadic_order),2,2**dyadic_order)/float(2**dyadic_order)  

        grad_1 = (KK[:,:,:,None] * Diff_1)/h
        grad_2 = (KK[:,:,:,None] * Diff_2)/h

        grad_1 = torch.sum(grad_1,axis=2)
        grad_1 = torch.sum(grad_1.reshape(A,M-1,2**dyadic_order,D),axis=2)
        grad_2 = torch.sum(grad_2,axis=2)
        grad_2 = torch.sum(grad_2.reshape(A,M-1,2**dyadic_order,D),axis=2)

        grad_prev = grad_1[:,:-1,:] + grad_2[:,1:,:]  # /¯¯
        grad_next = torch.cat([torch.zeros((A, 1, D), dtype=X.dtype, device=X.device), grad_1[:,1:,:]],dim=1)   # /
        grad_incr = grad_prev - grad_1[:,1:,:]
        grad_points = torch.cat([(grad_2[:,0,:]-grad_1[:,0,:])[:,None,:],grad_incr,grad_1[:,-1,:][:,None,:]],dim=1)
        return grad_output[:,None,None]*grad_points, None, None, None
# ===========================================================================================================


# ===========================================================================================================
# Signature Kernel Gram Matrix
# ===========================================================================================================
class SigKernelGramMat(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y, n=0, solver=0, sym=False, rbf=False, sigma=1.):

        A = X.shape[0]
        B = Y.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        MM = (2**n)*(M-1)
        NN = (2**n)*(N-1)

        if X.requires_grad:
            assert not rbf, 'Current backpropagation method only for linear signature kernel. For rbf signature kernel use naive implementation'

        # if on GPU
        if X.device.type=='cuda':

            assert max(MM,NN) < 1024, 'n must be lowered or data must be moved to CPU as the current choice of n makes exceed the thread limit'

            M_inc = increment_matrix_mmd(X,Y,rbf,sigma,n)

            # cuda parameters
            threads_per_block = max(MM+1,NN+1)
            n_anti_diagonals = 2 * threads_per_block - 1

            # Prepare the tensor of output solutions to the PDE (forward)
            G = torch.zeros((A, B, MM+2, NN+2), device=M_inc.device, dtype=M_inc.dtype) 
            G[:,:,0,:] = 1.
            G[:,:,:,0] = 1. 

            # Run the CUDA kernel.
            blockspergrid = (A,B)
            compute_sig_kernel_Gram_mat_varpar_from_increments_cuda[blockspergrid, threads_per_block](cuda.as_cuda_array(M_inc.detach()),
                                                                                                      MM+1, NN+1, n_anti_diagonals,
                                                                                                      cuda.as_cuda_array(G), solver)

            G = G[:,:,:-1,:-1]

        else:
            G = sig_kernel_Gram_matrix(X.detach().numpy(), Y.detach().numpy(), n=n, solver=solver, sym=sym, full=True, rbf=rbf, sigma=sigma)
            G = torch.tensor(G, dtype=X.dtype)

        ctx.save_for_backward(X,Y,G)      
        ctx.n = n
        ctx.solver = solver
        ctx.sym = sym

        return G[:,:,-1,-1]


    @staticmethod
    def backward(ctx, grad_output):

        X, Y, G = ctx.saved_tensors
        n = ctx.n
        solver = ctx.solver
        sym = ctx.sym

        A = X.shape[0]
        B = Y.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        MM = (2**n)*(M-1)
        NN = (2**n)*(N-1)

        # Reverse paths
        X_rev = torch.flip(X, dims=[1])
        Y_rev = torch.flip(Y, dims=[1])

        # if on GPU
        if X.device.type=='cuda':

            M_inc_rev = increment_matrix_mmd(X_rev,Y_rev,rbf=False,sigma=None,n=n)

            # Prepare the tensor of output solutions to the PDE (backward)
            G_rev = torch.zeros((A, B, MM+2, NN+2), device=M_inc_rev.device, dtype=M_inc_rev.dtype) 
            G_rev[:,:,0,:] = 1.
            G_rev[:,:,:,0] = 1. 

            # cuda parameters
            threads_per_block = max(MM+1,NN+1)
            n_anti_diagonals = 2 * threads_per_block - 1

            # Compute signature kernel for reversed paths
            blockspergrid = (A,B)
            compute_sig_kernel_Gram_mat_varpar_from_increments_cuda[blockspergrid, threads_per_block](cuda.as_cuda_array(M_inc_rev.detach()), 
                                                                                                      MM+1, NN+1, n_anti_diagonals,
                                                                                                      cuda.as_cuda_array(G_rev), solver)

            G_rev = G_rev[:,:,:-1,:-1]

        # if on CPU
        else:
            G_rev = sig_kernel_Gram_matrix(X_rev.detach().numpy(), Y_rev.detach().numpy(), n=n, solver=solver, sym=sym, full=True, rbf=False, sigma=None)
            G_rev = torch.tensor(G_rev, dtype=X.dtype)

        inc_Y = tile(Y[:,1:,:]-Y[:,:-1,:],1,2**n)/float(2**n)               # (B,(2**n)*(N-1),D)  increments on the finer grid

        G_rev = flip(flip(G_rev,dim=2),dim=3)

        GG = G[:,:,:-1,:-1] * G_rev[:,:,1:,1:]                              # (A,B,(2**n)*(M-1),(2**n)*(N-1))

        grad_incr = GG[:,:,:,:,None]*inc_Y[None,:,None,:,:]                 # (A,B,(2**n)*(M-1),(2**n)*(N-1),D)

        grad_incr = (1./(2**n))*torch.sum(grad_incr,axis=3)                 # (A,B,(2**n)*(M-1),D)

        grad_incr =  torch.sum(grad_incr.reshape(A,B,M-1,2**n,D),axis=3)    # (A,B,M-1,D)

        grad_points = -torch.cat([grad_incr,torch.zeros((A, B, 1, D), dtype=X.dtype, device=X.device)], dim=2) + torch.cat([torch.zeros((A, B, 1, D), dtype=X.dtype, device=X.device), grad_incr], dim=2)

        if sym:
            grad = (grad_output[:,:,None,None]*grad_points + grad_output.t()[:,:,None,None]*grad_points).sum(dim=1)
            return grad, None, None, None, None, None, None
        else:
            grad = (grad_output[:,:,None,None]*grad_points).sum(dim=1)
            return grad, None, None, None, None, None, None
# ===========================================================================================================


# ===========================================================================================================
# Various utility functions
# ===========================================================================================================
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)
# ===========================================================================================================
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(a.device)
    return torch.index_select(a, dim, order_index)
# ===========================================================================================================
def increment_matrix(X,Y,rbf,sigma,n):
    A = X.shape[0]
    M = X.shape[1]
    N = Y.shape[1]
    if rbf:
        Xs = torch.sum(X**2, dim=2)
        Ys = torch.sum(Y**2, dim=2)
        dist = -2.*torch.bmm(X, Y.permute(0,2,1))
        dist += torch.reshape(Xs,(A,M,1)) + torch.reshape(Ys,(A,1,N))
        M_inc = torch.exp(-dist/sigma)
        M_inc = M_inc[:,1:,1:] + M_inc[:,:-1,:-1] - M_inc[:,1:,:-1] - M_inc[:,:-1,1:] 
        M_inc = tile(tile(M_inc,1,2**n)/float(2**n),2,2**n)/float(2**n)
    else:
        inc_X = tile(X[:,1:,:]-X[:,:-1,:],1,2**n)/float(2**n)
        inc_Y = tile(Y[:,1:,:]-Y[:,:-1,:],1,2**n)/float(2**n)
        M_inc = torch.bmm(inc_X, inc_Y.permute(0,2,1))
    return M_inc
# ===========================================================================================================
def increment_matrix_mmd(X,Y,rbf,sigma,n):
    A = X.shape[0]
    B = Y.shape[0]
    M = X.shape[1]
    N = Y.shape[1]
    if rbf:
        Xs = torch.sum(X**2, dim=2)
        Ys = torch.sum(Y**2, dim=2)
        dist = -2.*torch.einsum('ipk,jqk->ijpq', X, Y)
        dist += torch.reshape(Xs,(A,1,M,1)) + torch.reshape(Ys,(1,B,1,N))
        M_inc = torch.exp(-dist/sigma)
        M_inc = M_inc[:,:,1:,1:] + M_inc[:,:,:-1,:-1] - M_inc[:,:,1:,:-1] - M_inc[:,:,:-1,1:] 
        M_inc = tile(tile(M_inc,2,2**n)/float(2**n),3,2**n)/float(2**n)
    else:
        inc_X = tile(X[:,1:,:]-X[:,:-1,:],1,2**n)/float(2**n)
        inc_Y = tile(Y[:,1:,:]-Y[:,:-1,:],1,2**n)/float(2**n)
        M_inc = torch.einsum('ipk,jqk->ijpq', inc_X, inc_Y)
    return M_inc
# ===========================================================================================================



# ===========================================================================================================
# Naive implementation of Signature Kernel with original finite difference scheme (slow, just for testing)
# ===========================================================================================================
def SigKernel_naive(X,Y,n=0,solver=0,rbf=False,sigma=1.):

    A = len(X)
    M = X[0].shape[0]
    N = Y[0].shape[0]

    K_XY = torch.zeros((A, (2**n)*(M-1)+1, (2**n)*(N-1)+1), dtype=X.dtype, device=X.device)
    K_XY[:, 0, :] = 1.
    K_XY[:, :, 0] = 1.

    M_inc = increment_matrix(X,Y,rbf,sigma,n)

    for i in range(0, (2**n)*(M-1)):
        for j in range(0, (2**n)*(N-1)):

            increment = M_inc[:,i,j].clone()

            k_10 = K_XY[:, i + 1, j].clone()
            k_01 = K_XY[:, i, j + 1].clone()
            k_00 = K_XY[:, i, j].clone()

            if solver==0:
                K_XY[:, i + 1, j + 1] = k_10 + k_01 + k_00*(increment-1.)
            elif solver==1:
                K_XY[:, i + 1, j + 1] = (k_10 + k_01)*(1.+0.5*increment+(1./12)*increment**2) - k_00*(1.-(1./12)*increment**2)
            else:
                #K_XY[:, i + 1, j + 1] = k_01+k_10-k_00 + ((0.5*inc)/(1.-0.25*increment))*(k_01+k_10)
                K_XY[:, i + 1, j + 1] = k_01 + k_10 - k_00 + (torch.exp(0.5*increment) - 1.)*(k_01 + k_10)
            
    return K_XY[:, -1, -1]
# ===========================================================================================================


# ===========================================================================================================
# Naive implementation of Signature Gram matrix with original finite difference scheme (slow, just for testing)
# ===========================================================================================================
def SigKernelGramMat_naive(X,Y,n=0,solver=0,rbf=False,sigma=1.):

    A = len(X)
    B = len(Y)
    M = X[0].shape[0]
    N = Y[0].shape[0]

    K_XY = torch.zeros((A,B, (2**n)*(M-1)+1, (2**n)*(N-1)+1), dtype=X.dtype, device=X.device)
    K_XY[:,:, 0, :] = 1.
    K_XY[:,:, :, 0] = 1.

    M_inc = increment_matrix_mmd(X,Y,rbf,sigma,n)

    for i in range(0, (2**n)*(M-1)):
        for j in range(0, (2**n)*(N-1)):

            increment = M_inc[:,:,i,j].clone()

            k_10 = K_XY[:, :, i + 1, j].clone()
            k_01 = K_XY[:, :, i, j + 1].clone()
            k_00 = K_XY[:, :, i, j].clone()

            if solver==0:
                K_XY[:, :, i + 1, j + 1] = k_10 + k_01 + k_00*(increment-1.)
            elif solver==1:
                K_XY[:, :, i + 1, j + 1] = (k_10 + k_01)*(1.+0.5*increment+(1./12)*increment**2) - k_00*(1.-(1./12)*increment**2)
            else:
                #K_XY[:, :, i + 1, j + 1] = k_01+k_10-k_00 + ((0.5*inc)/(1.-0.25*increment))*(k_01+k_10)
                K_XY[:, :, i + 1, j + 1] = k_01 + k_10 - k_00 + (torch.exp(0.5*increment) - 1.)*(k_01 + k_10)

    return K_XY[:,:, -1, -1]