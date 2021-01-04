import iisignature
from esig import tosig as sig
import numpy as np
from scipy.ndimage.interpolation import shift
from scipy.integrate import odeint
import math


def white(steps, width, time=1.):
    mu, sigma = 0, math.sqrt(time / steps) 
    return np.random.normal(mu, sigma, (steps, width))

def brownian(steps, width, time=1.):
    path = np.zeros((steps + 1, width))
    np.cumsum(white(steps, width, time), axis=0, out=path[1:, :])
    return path

def brownian_perturbed(steps, width, time=1., amplitude=1.):
    path = brownian(steps, width, time)
    t = np.random.randint(steps)
    path[t:] = path[t:] + amplitude
    return path

def naive_sig_kernel(x,y,depth):
    sigx = iisignature.sig(x,depth,2)
    sigy = iisignature.sig(y,depth,2)
    k_true = np.ones((len(x),len(y)))
    for i,sx in enumerate(sigx):
        for j,sy in enumerate(sigy):
            k_true[i+1,j+1] = 1.+np.dot(sigx[i],sigy[j])
    return k_true

def truncated_sigKernel(X, num_levels, order=-1, difference=True, sigma=1.):
    """
    Computes the (truncated) signature kernel matrix of a given array of sequences. 
    
    Inputs:
    :X: a numpy array of shape (num_seq, len_seq, num_feat) of num_seq sequences of equal length, len_seq, with num_feat coordinates
    :num_levels: the number of signature levels used
    :order: the order of the signature kernel as per Kiraly and Oberhauser, order=num_levels gives the full signature kernel, while order < num_levels gives a lower order approximation. Defaults to order=-1, which means order=num_levels
    :difference: whether to difference the time series before computations (defaults to True)
    :sigma: a scalar or an np array of shape (num_levels+1); a multiplicative factor for each level
    
    Output:
    :K: a numpy array of shape (num_seq, num_seq)
    """
    order = num_levels if order < 1 else order
    sigma = sigma * np.ones((num_levels + 1,), dtype=X.dtype)
    
    if difference:
        X = np.diff(X, axis=1)
    
    num_seq, len_seq, num_feat = X.shape
    
    M = np.reshape(X.reshape((-1, num_feat)) @ X.reshape((-1, num_feat)).T, (num_seq, len_seq, num_seq, len_seq))
    K = sigma[0] * np.ones((num_seq, num_seq), dtype=X.dtype) + sigma[1] * np.sum(M, axis=(1, 3))
    R = M[None, None, ...]
    
    for m in range(1, num_levels):
        d = min(m+1, order)
        R_next = np.empty((d, d, num_seq, len_seq, num_seq, len_seq), dtype=X.dtype)
        R_next[0, 0] = M * shift(np.cumsum(np.cumsum(np.sum(R, axis=(0, 1)), axis=1), axis=3), shift=(0, 1, 0, 1))
        for j in range(1, d):
            R_next[0, j] = 1./(j+1) * M * shift(np.cumsum(np.sum(R[:, j-1], axis=0), axis=1), shift=(0, 1, 0, 0))
            R_next[j, 0] = 1./(j+1) * M * shift(np.cumsum(np.sum(R[j-1, :], axis=0), axis=3), shift=(0, 0, 0, 1))
            for i in range(1, d):
                R_next[i, j] = 1./((j+1)*(i+1)) * M * R[i-1, j-1]
        R = R_next
        K += sigma[m+1] * np.sum(R, axis=(0, 1, 3, 5))
    return K


def generate(M, N, h_x=0.8, h_y=0.8, scale=1., signature=False, BM=False, dim_BM=2):
    
    if BM:
        X = brownian(M-1, dim_BM, time=1.)
        Y = brownian(N-1, dim_BM, time=1.)

    else:
        fbm_generator_X = FBM(M-1, h_x)
        fbm_generator_Y = FBM(N-1, h_y)

        x = scale*fbm_generator_X.fbm()
        y = scale*fbm_generator_Y.fbm()

        X = AddTime().fit_transform([x])[0]
        Y = AddTime().fit_transform([y])[0]
    
    if signature:
        X = iisignature.sig(X,5,2)
        Y = iisignature.sig(Y,5,2)

        X0 = np.zeros_like(X[0,:].reshape(1,-1))
        X0[0,0] = 1.
        X = np.concatenate([X0, X])
        Y = np.concatenate([X0, Y])
        
    return X, Y