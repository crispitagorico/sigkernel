import iisignature
from esig import tosig as sig
import numpy as np
from scipy.ndimage.interpolation import shift
import math

def naive_SigKernel_iisig(X, Y, depth):
    sig_x = iisignature.sig(X, depth)
    sig_y = iisignature.sig(Y, depth)
    return 1. + np.sum([x*y for x,y in zip(sig_x, sig_y)], dtype=np.double)

def naive_SigKernel_esig(X, Y, depth):
    sig_x = sig.stream2sig(X, depth)
    sig_y = sig.stream2sig(Y, depth)
    return np.sum([x*y for x,y in zip(sig_x, sig_y)], dtype=np.float64)
    
def white(steps, width, time=1.):
    mu, sigma = 0, math.sqrt(time / steps) 
    return np.random.normal(mu, sigma, (steps, width))

def brownian(steps, width, time=1.):
    path = np.zeros((steps + 1, width))
    np.cumsum(white(steps, width, time), axis=0, out=path[1:, :])
    return path

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