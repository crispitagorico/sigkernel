# cython: boundscheck=False
# cython: wraparound=False

import numpy as np

def forward_step_explicit(double k_00, double k_01, double k_10, double increment):
	return (k_10 + k_01)*(1.+0.5*increment+(1./12)*increment**2) - k_00*(1.-(1./12)*increment**2)

def forward_step_implicit(double k_00, double k_01, double k_10, double increment):
	return k_01+k_10-k_00 + ((0.5*increment)/(1.-0.25*increment))*(k_01+k_10)

def sig_kernel_batch(double[:,:,:] x, double[:,:,:] y, int n=0, bint implicit=True, bint gradients=True):

	cdef int A = x.shape[0]
	cdef int M = x.shape[1]
	cdef int N = y.shape[1]
	cdef int D = x.shape[2]

	cdef double increment, increment_rev
	cdef double factor = 2**(2*n)

	cdef int i, j, k, l, ii, jj
	cdef int MM = (2**n)*(M-1)
	cdef int NN = (2**n)*(N-1)

	cdef double[:,:,:] K = np.zeros((A,MM+1,NN+1), dtype=np.float64)

	if gradients:
		
		cdef double[:,:,:] K_rev = np.zeros((A,MM+1,NN+1), dtype=np.float64)
		
		for l in range(A):
			
			for i in range(MM+1):
				K[l,i,0] = 1.
				K_rev[l,i,0] = 1.
	
			for j in range(NN+1):
				K[l,0,j] = 1.
				K_rev[l,0,j] = 1.

			for i in range(MM):
				for j in range(NN):

					ii = int(i/(2**n))
					jj = int(j/(2**n))

					increment = 0.
					increment_rev = 0.
					for k in range(D):
						increment += (x[l,ii+1,k]-x[l,ii,k])*(y[l,jj+1,k]-y[l,jj,k])/factor
						increment_rev += (x[l,(M-1)-(ii+1),k]-x[l,(M-1)-ii,k])*(y[l,(N-1)-(jj+1),k]-y[l,(N-1)-jj,k])/factor

					if implicit:
						K[l,i+1,j+1] = forward_step_implicit(K[i,j], K[i,j+1], K[i+1,j], increment)
						K_rev[l,i+1,j+1] = forward_step_implicit(K_rev[i,j], K_rev[i,j+1], K_rev[i+1,j], increment_rev)
					else:
						K[l,i+1,j+1] = forward_step_explicit(K[i,j], K[i,j+1], K[i+1,j], increment)
						K_rev[l,i+1,j+1] = forward_step_explicit(K_rev[i,j], K_rev[i,j+1], K_rev[i+1,j], increment_rev)
	
	else:

		for l in range(A):
			
			for i in range(MM+1):
				K[l,i,0] = 1.
	
			for j in range(NN+1):
				K[l,0,j] = 1.

			for i in range(MM):
				for j in range(NN):

					ii = int(i/(2**n))
					jj = int(j/(2**n))

					increment = 0.
					for k in range(D):
						increment += (x[l,ii+1,k]-x[l,ii,k])*(y[l,jj+1,k]-y[l,jj,k])/factor

					if implicit:
						K[l,i+1,j+1] = forward_step_implicit(K[i,j], K[i,j+1], K[i+1,j], increment)
					else:
						K[l,i+1,j+1] = forward_step_explicit(K[i,j], K[i,j+1], K[i+1,j], increment)

	if gradients:
		return K, K_rev
	else:
		return K

