import numpy as np
import torch
import torch.cuda
from numba import cuda

# ===========================================================================================================
# Static kernels
# ===========================================================================================================


class LinearKernel():
    """Linear kernel k: R^d x R^d -> R"""

    def __init__(self, scale=1.0):
        self.scale = scale
        
    def batch_kernel(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch, length_X, dim),
                  - Y: torch tensor of shape (batch, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^i_t) of shape (batch, length_X, length_Y)
        """
        return torch.bmm(self.scale*X, self.scale*Y.permute(0,2,1))

    def Gram_matrix(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^j_t) of shape (batch_X, batch_Y, length_X, length_Y)
        """
        return torch.einsum('ipk,jqk->ijpq', X, Y)


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

class RBF_CEXP_Kernel(RBFKernel):
    """RBF CEXP kernel k: H x H -> R"""

    def __init__(self, sigma1, sigma2, n_freqs):
        self.sigma1 = sigma1
        super().__init__(sigma2)
        self.n_freqs = n_freqs

    def batch_kernel(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch, length_X_t, length_x, dim),
                  - Y: torch tensor of shape (batch, length_Y_t, length_x, dim)
           Output: 
                  - matrix k(X^i_s,Y^i_t) of shape (batch, length_X_t, length_Y_t)
        """
        
        # transform
        CX = CEXP(X, self.n_freqs, self.sigma1)  #(batch, length_X_t, length_x, dim)
        CY = CEXP(Y, self.n_freqs, self.sigma1)  #(batch, length_Y_t, length_x, dim)
        
        CX = CX.reshape(X.shape[0], X.shape[1], -1)  #(batch, length_X_t, length_x x dim)
        CY = CY.reshape(Y.shape[0], Y.shape[1], -1)  #(batch, length_Y_t, length_x x dim)
        
        return super().batch_kernel(CX,CY) #(batch, length_X_t, length_Y_t)

    def Gram_matrix(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^j_t) of shape (batch_X, batch_Y, length_X, length_Y)
        """
        # transform
        CX = CEXP(X, self.n_freqs, self.sigma1)  #(batch, length_X_t, length_x, dim)
        CY = CEXP(Y, self.n_freqs, self.sigma1)  #(batch, length_Y_t, length_x, dim)
        
        CX = CX.reshape(X.shape[0], X.shape[1], -1)  #(batch, length_X_t, length_x x dim)
        CY = CY.reshape(Y.shape[0], Y.shape[1], -1)  #(batch, length_Y_t, length_x x dim)
        
        return super().Gram_matrix(CX,CY)  
    
class RBF_SQR_Kernel():
    """RBF SQR kernel k: H x H -> R"""

    def __init__(self, sigma1, sigma2):
        self.rbf1 = RBFKernel(sigma_1)
        self.rbf2 = RBFKernel(sigma_2)

    def batch_kernel(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch, length_X_t, length_x, dim),
                  - Y: torch tensor of shape (batch, length_Y_t, length_x dim)
           Output: 
                  - matrix k(X^i_s,Y^i_t) of shape (batch, length_X_t, length_Y_t)
        """
        X = X.reshape(X.shape[0], X.shape[1], -1)  #(batch, length_X_t, length_x x dim)
        Y = Y.reshape(Y.shape[0], Y.shape[1], -1)  #(batch, length_Y_t, length_x x dim)

        return self.rbf1.batch_kernel(X,Y)*self.rbf2.batch_kernel(X**2,Y**2)

    def Gram_matrix(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^j_t) of shape (batch_X, batch_Y, length_X, length_Y)
        """
        X = X.reshape(X.shape[0], X.shape[1], -1)  #(batch, length_X_t, length_x x dim)
        Y = Y.reshape(Y.shape[0], Y.shape[1], -1)  #(batch, length_Y_t, length_x x dim)

        return self.rbf1.Gram_matrix(X,Y)*self.rbf2.Gram_matrix(X**2,Y**2)
    
class Linear_ID_Kernel(LinearKernel):
    """COV kernel k: H x H -> R"""

    def __init__(self):
        super().__init__()

    def batch_kernel(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch, length_X_t, length_x, dim),
                  - Y: torch tensor of shape (batch, length_Y_t, length_x dim)
           Output: 
                  - matrix k(X^i_s,Y^i_t) of shape (batch, length_X_t, length_Y_t)
        """
        X = X.reshape(X.shape[0], X.shape[1], -1)  #(batch, length_X_t, length_x x dim)
        Y = Y.reshape(Y.shape[0], Y.shape[1], -1)  #(batch, length_Y_t, length_x x dim)

        return super().batch_kernel(X, Y)

    def Gram_matrix(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^j_t) of shape (batch_X, batch_Y, length_X, length_Y)
        """
        X = X.reshape(X.shape[0], X.shape[1], -1)  #(batch, length_X_t, length_x x dim)
        Y = Y.reshape(Y.shape[0], Y.shape[1], -1)  #(batch, length_Y_t, length_x x dim)

        return super().Gram_matrix(X,Y)
    
    
class RBF_ID_Kernel(RBFKernel):
    """COV kernel k: H x H -> R"""

    def __init__(self, sigma):
        super().__init__(sigma)

    def batch_kernel(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch, length_X_t, length_x, dim),
                  - Y: torch tensor of shape (batch, length_Y_t, length_x dim)
           Output: 
                  - matrix k(X^i_s,Y^i_t) of shape (batch, length_X_t, length_Y_t)
        """
        X = X.reshape(X.shape[0], X.shape[1], -1)  #(batch, length_X_t, length_x x dim)
        Y = Y.reshape(Y.shape[0], Y.shape[1], -1)  #(batch, length_Y_t, length_x x dim)

        return super().batch_kernel(X, Y)

    def Gram_matrix(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^j_t) of shape (batch_X, batch_Y, length_X, length_Y)
        """
        X = X.reshape(X.shape[0], X.shape[1], -1)  #(batch, length_X_t, length_x x dim)
        Y = Y.reshape(Y.shape[0], Y.shape[1], -1)  #(batch, length_Y_t, length_x x dim)

        return super().Gram_matrix(X,Y)
# ===========================================================================================================

def CEXP(X,n_freqs = 20, sigma=np.sqrt(10)):
    """
    Transforms an array of function values using the integral operator induced by the cos-exp kernel. 
    The function values are assumed to be on [0,1]
    
    Parameters:
    X - (batch, length_t, length_x, dim) array of function values
    n_freqs - number of frequencies to include in the sum
    sigma - bandwidth of the kernel
    
    Returns:
    cos_exp_X - (batch, length_t, length_x, dim) array of function values where each function has been passed
                through the integral operator induced by the cos-exp kernel
    """
    length_x = X.shape[2]
    obs_grid = torch.linspace(0,1,length_x, dtype=torch.float64).to(X.device)
    x_y = obs_grid[:,None] - obs_grid[None,:]  # length_x, length_x
    
    T_mat = cos_exp_kernel(x_y, n_freqs = n_freqs, sigma=sigma) # length_x, length_x

    cos_exp_X = (1./length_x)*torch.matmul(X.permute(0,1,3,2), T_mat)  # batch, length_t, dim, length_x
    
    return cos_exp_X.permute(0,1,3,2) # batch, length_t, length_x, dim

def cos_exp_kernel(x_y, n_freqs = 5, sigma=1):
    """
    The c-exp kernel
    
    Parameters:
    x_y - square matrix with entries x_y[i,j] = x_i - y_j 
    n_freqs - number of frequencies to include in the sum
    sigma - bandwidth of the kernel
    
    Returns:
    Kernel values given x,y
    """
    
    cos_term = torch.cos(2*torch.pi*x_y[:,:,None] * torch.arange(n_freqs)[None,None].to(x_y.device)).sum(dim=-1)
    
#     cos_term = torch.sum([torch.cos(2*torch.pi*n*(x-y)) for n in range(n_freqs)])
    
    return cos_term*torch.exp(-x_y**2/sigma)

