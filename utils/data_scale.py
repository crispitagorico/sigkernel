import iisignature
from scipy.optimize import brentq as brentq
import numpy as np

''' MAIN FUNCTIONS USED FOR PATH RE-SCALING '''

def scale_path(x,M,a,level_sig):
    '''
        This function computes a single scaling factor \theta_x (path-dependent) and rescales the path x 
        , i.e. return x_new: t -> \theta_x*x_t, such that ||S_{<level_sig}(x_new)|| < M(1+1/a).
        
        Inputs:
            - x: an array (L,D) representing a path where L is the length and D is the state-space dimension
            - (int) level_sig: the truncation level to compute the norm of S_{<level_sig}(x)
            - (int) M: the first parameter used to define the maximum norm allowed 
            - (float) a: the second parameter used to define the maximum norm allowed 
        Outputs:
            - x_new: the rescaled path 
    '''
    D = x.shape[1]

    maxi = M*(1.+1./a)

    sig = iisignature.sig(x, level_sig) # computes the signature of the path x

    norm_levels = get_norm_level_sig(sig, D, level_sig) # gets the norm of each tensor in S(x) truncated at level level_sig
    
    norm = np.sum(norm_levels) # gets the norm of S(x) truncated at level level_sig

    psi = psi_tilde(norm,M,a) # computes an intermediary number which is used to find the scale

    theta_x = brentq(poly,0, 10000, args=(psi, norm_levels,level_sig)) # computes the scale 

    return theta_x*x 

''' UTILS FUNCTIONS ''' 

def get_norm_level_sig(sig, d, level_sig):
    ''' 
        This function computes the norm of each tensor in the truncated signature in input
        INPUTS:
            - (array) sig: a flat signature
            - (int) d: the original state-space dimension of the path
            - (int) level_sig: the truncation level of the signature
        OUTPUT:
            - (list) norms: a list containing the norms of each level of the truncated signature in input
    '''
    norms = [1.]

    for k in range(1,level_sig+1):

        start = int(((1 - d ** k) / (1 - d)) - 1)
        end = int((1 - d ** (k + 1)) / (1 - d) - 1)

        norms.append(np.sum(sig[start:end]**2))

    return norms

def psi_tilde(x, M, a):
    if x <= M:
        return x
    else:
        return M + pow(M, 1. + a) * (pow(M, -a) - pow(x, -a)) / a


def poly(x,psi,coef,level_sig):
    return np.sum([coef[i]*x**(2*i) for i in range(level_sig+1)])-psi