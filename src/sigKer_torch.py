import numpy as np
import torch
import torch.cuda
from numba import cuda

from sigKer_fast import sig_kernel_batch_varpar, sig_kernel_Gram_varpar
from sigKer_cuda import compute_sig_kernel_batch_varpar_from_increments_cuda, compute_sig_kernel_Gram_mat_varpar_from_increments_cuda


# ===========================================================================================================
# Static kernels
# ===========================================================================================================

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

    def Gram_matrix(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^j_t) of shape (batch_X, batch_Y, length_X, length_Y)
        """
        return torch.einsum('ipk,jqk->ijpq', X, Y)

# ===========================================================================================================
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

    def Gram_matrix(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^j_t) of shape (batch_X, batch_Y, length_X, length_Y)
        """
        A = X.shape[0]
        B = Y.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        Xs = torch.sum(X**2, dim=2)
        Ys = torch.sum(Y**2, dim=2)
        dist = -2.*torch.einsum('ipk,jqk->ijpq', X, Y)
        dist += torch.reshape(Xs,(A,1,M,1)) + torch.reshape(Ys,(1,B,1,N))
        return torch.exp(-dist/self.sigma)


# ===========================================================================================================
# Signature Kernel wrapper
# ===========================================================================================================
class SigKernel():
    """Wrapper of the signature kernel k_sig(x,y) = <S(f(x)),S(f(y))> where k(x,y) = <f(x),f(y)> is a given static kernel"""

    def __init__(self,static_kernel, dyadic_order):
        self.static_kernel = static_kernel
        self.dyadic_order = dyadic_order

    def compute_kernel(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch, length_X, dim),
                  - Y: torch tensor of shape (batch, length_Y, dim)
           Output: 
                  - vector k(X^i_T,Y^i_T) of shape (batch,)
        """
        return _SigKernel.apply(X, Y, self.static_kernel, self.dyadic_order)

    def compute_Gram(self, X, Y, sym=False):
        """Input: 
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output: 
                  - matrix k(X^i_T,Y^j_T) of shape (batch_X, batch_Y)
        """
        return _SigKernelGram.apply(X, Y, self.static_kernel, self.dyadic_order, sym)

    def compute_distance(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch, length_X, dim),
                  - Y: torch tensor of shape (batch, length_Y, dim)
           Output: 
                  - vector ||S(X^i)_T - S(Y^i)_T||^2 of shape (batch,)
        """
        
        assert not Y.requires_grad, "the second input should not require grad"

        k_XX = self.compute_kernel(X, X)
        k_YY = self.compute_kernel(Y, Y)
        k_XY = self.compute_kernel(X, Y)

        return torch.mean(k_XX) + torch.mean(k_YY) - 2.*torch.mean(k_XY) 

    def compute_mmd(self):
        """Input: 
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output: 
                  - scalar: MMD signature distance between samples X and samples Y
        """

        assert not Y.requires_grad, "the second input should not require grad"

        K_XX = self.compute_Gram(X, X, sym=True)
        K_YY = self.compute_Gram(Y, Y, sym=True)
        K_XY = self.compute_Gram(X, Y, sym=False)

        return torch.mean(K_XX) + torch.mean(K_YY) - 2.*torch.mean(K_XY)



# ===========================================================================================================
# Signature kernel
# ===========================================================================================================
class _SigKernel(torch.autograd.Function):
    """Signature kernel k_sig(x,y) = <S(f(x)),S(f(y))> where k(x,y) = <f(x),f(y)> is a given static kernel"""
 
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
# Signature Kernel Gram Matrix
# ===========================================================================================================
class _SigKernelGram(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y, static_kernel, dyadic_order, sym=False):

        A = X.shape[0]
        B = Y.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        MM = (2**n)*(M-1)
        NN = (2**n)*(N-1)

        # computing dsdt k(X^i_s,Y^j_t)
        G_static = static_kernel.Gram_matrix(X,Y)
        G_static_ = G_static[:,:,1:,1:] + G_static[:,:,:-1,:-1] - G_static[:,:,1:,:-1] - G_static[:,:,:-1,1:] 
        G_static_ = tile(tile(G_static_,2,2**dyadic_order)/float(2**dyadic_order),3,2**dyadic_order)/float(2**dyadic_order)

        # if on GPU
        if X.device.type=='cuda':

            assert max(MM,NN) < 1024, 'n must be lowered or data must be moved to CPU as the current choice of n makes exceed the thread limit'

            # cuda parameters
            threads_per_block = max(MM+1,NN+1)
            n_anti_diagonals = 2 * threads_per_block - 1

            # Prepare the tensor of output solutions to the PDE (forward)
            G = torch.zeros((A, B, MM+2, NN+2), device=G_static.device, dtype=G_static.dtype) 
            G[:,:,0,:] = 1.
            G[:,:,:,0] = 1. 

            # Run the CUDA kernel.
            blockspergrid = (A,B)
            compute_sig_kernel_Gram_mat_varpar_from_increments_cuda[blockspergrid, threads_per_block](cuda.as_cuda_array(G_static_.detach()),
                                                                                                      MM+1, NN+1, n_anti_diagonals,
                                                                                                      cuda.as_cuda_array(G))

            G = G[:,:,:-1,:-1]

        else:
            G = torch.tensor(sig_kernel_Gram_varpar(G_static_.detach().numpy(), sym=sym), dtype=G_static.dtype)

        ctx.save_for_backward(X,Y,G,G_static)      
        ctx.sym = sym

        return G[:,:,-1,-1]


    @staticmethod
    def backward(ctx, grad_output):

        X, Y, G, G_static = ctx.saved_tensors
        sym = ctx.sym
        static_kernel = ctx.static_kernel
        dyadic_order = ctx.dyadic_order

        G_static_ = G_static[:,:,1:,1:] + G_static[:,:,:-1,:-1] - G_static[:,:,1:,:-1] - G_static[:,:,:-1,1:] 
        G_static_ = tile(tile(G_static_,2,2**dyadic_order)/float(2**dyadic_order),3,2**dyadic_order)/float(2**dyadic_order)

        A = X.shape[0]
        B = Y.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        MM = (2**dyadic_order)*(M-1)
        NN = (2**dyadic_order)*(N-1)
            
        # Reverse paths
        X_rev = torch.flip(X, dims=[1])
        Y_rev = torch.flip(Y, dims=[1])

        # computing dsdt k(X_rev^i_s,Y_rev^j_t) for variation of parameters
        G_static_rev = flip(flip(G_static_,dim=2),dim=3)

        # if on GPU
        if X.device.type=='cuda':

            # Prepare the tensor of output solutions to the PDE (backward)
            G_rev = torch.zeros((A, B, MM+2, NN+2), device=G_static_rev.device, dtype=G_static_rev.dtype) 
            G_rev[:,:,0,:] = 1.
            G_rev[:,:,:,0] = 1. 

            # cuda parameters
            threads_per_block = max(MM+1,NN+1)
            n_anti_diagonals = 2 * threads_per_block - 1

            # Compute signature kernel for reversed paths
            blockspergrid = (A,B)
            compute_sig_kernel_Gram_mat_varpar_from_increments_cuda[blockspergrid, threads_per_block](cuda.as_cuda_array(G_static_rev.detach()), 
                                                                                                      MM+1, NN+1, n_anti_diagonals,
                                                                                                      cuda.as_cuda_array(G_rev))

            G_rev = G_rev[:,:,:-1,:-1]

        # if on CPU
        else:
            G_rev = torch.tensor(sig_kernel_Gram_varpar(G_static_rev, sym=sym), dtype=G_static.dtype)

        G_rev = flip(flip(G_rev,dim=2),dim=3)
        GG = G[:,:,:-1,:-1] * G_rev[:,:,1:,1:]     

        # finite difference step 
        h = 1e-9

        Xh = X[:,:,:,None] + h*torch.eye(D)[None,None,:]  
        Xh = Xh.permute(0,1,3,2)
        Xh = Xh.reshape(A,M*D,D)

        G_h = static_kernel.Gram_matrix(Xh,Y) 
        G_h = G_h.reshape(A,B,M,D,N)
        G_h = G_h.permute(0,1,2,4,3) 

        Diff_1 = G_h[:,:,1:,1:,:] - G_h[:,:,1:,:-1,:] - (G_static[:,:,1:,1:])[:,:,:,:,None] + (G_static[:,:,1:,:-1])[:,:,:,:,None]
        Diff_1 =  tile(tile(Diff_1,2,2**dyadic_order)/float(2**dyadic_order),3,2**dyadic_order)/float(2**dyadic_order)  
        Diff_2 = G_h[:,:,1:,1:,:] - G_h[:,:,1:,:-1,:] - (G_static[:,:,1:,1:])[:,:,:,:,None] + (G_static[:,:,1:,:-1])[:,:,:,:,None]
        Diff_2 += - G_h[:,:,:-1,1:,:] + G_h[:,:,:-1,:-1,:] + (G_static[:,:,:-1,1:])[:,:,:,:,None] - (G_static[:,:,:-1,:-1])[:,:,:,:,None]
        Diff_2 = tile(tile(Diff_2,2,2**dyadic_order)/float(2**dyadic_order),3,2**dyadic_order)/float(2**dyadic_order)  

        grad_1 = (GG[:,:,:,:,None] * Diff_1)/h
        grad_2 = (GG[:,:,:,:,None] * Diff_2)/h

        grad_1 = torch.sum(grad_1,axis=3)
        grad_1 = torch.sum(grad_1.reshape(A,B,M-1,2**dyadic_order,D),axis=3)
        grad_2 = torch.sum(grad_2,axis=3)
        grad_2 = torch.sum(grad_2.reshape(A,B,M-1,2**dyadic_order,D),axis=3)

        grad_prev = grad_1[:,:,:-1,:] + grad_2[:,:,1:,:]  # /¯¯
        grad_next = torch.cat([torch.zeros((A, B, 1, D), dtype=X.dtype, device=X.device), grad_1[:,:,1:,:]], dim=2)   # /
        grad_incr = grad_prev - grad_1[:,:,1:,:]
        grad_points = torch.cat([(grad_2[:,:,0,:]-grad_1[:,:,0,:])[:,:,None,:],grad_incr,grad_1[:,:,-1,:][:,:,None,:]],dim=2)

        if sym:
            grad = (grad_output[:,:,None,None]*grad_points + grad_output.t()[:,:,None,None]*grad_points).sum(dim=1)
            return grad, None, None, None, None, None, None
        else:
            grad = (grad_output[:,:,None,None]*grad_points).sum(dim=1)
            return grad, None, None, None, None, None, None


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









# ===========================================================================================================
# Naive implementation of Signature Kernel with original finite difference scheme (slow, just for testing)
# ===========================================================================================================
def SigKernel_naive(X,Y,n=0,solver=1,rbf=False,sigma=1.):

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
# Naive implementation SigLoss with pytorch auto-diff (slow, just for testing)
# ===========================================================================================================
class SigLoss_naive(torch.nn.Module):

    def __init__(self, n=0, solver=0, rbf=False, sigma=1.):
        super(SigLoss_naive, self).__init__()
        self.n = n
        self.solver = solver
        self.rbf = rbf
        self.sigma = sigma

    def forward(self,x,y):

        k_xx = SigKernel_naive(x,x,self.n,self.solver,self.rbf,self.sigma)
        k_yy = SigKernel_naive(y,y,self.n,self.solver,self.rbf,self.sigma)
        k_xy = SigKernel_naive(x,y,self.n,self.solver,self.rbf,self.sigma)

        return torch.mean(k_xx) + torch.mean(k_yy) - 2.*torch.mean(k_xy)


# ===========================================================================================================
# Naive implementation of Signature Gram matrix with original finite difference scheme (slow, just for testing)
# ===========================================================================================================
def SigKernelGramMat_naive(X,Y,n=0,solver=1,rbf=False,sigma=1.):

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