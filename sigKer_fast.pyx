# cython: boundscheck=False
# cython: wraparound=False

import numpy as np

def sig_kernel(double[:,:] x, double[:,:] y, int n=0):
	
	cdef int D = x.shape[1]
	cdef int M = x.shape[0]
	cdef int N = y.shape[0]

	cdef double increment
	cdef double factor = 2**(2*n)

	cdef int i, j, k, ii, jj
	cdef int MM = (2**n)*(M-1)
	cdef int NN = (2**n)*(N-1)

	cdef double[:,:] K = np.zeros((MM+1,NN+1), dtype=np.float64)

	for i in range(MM+1):
		K[i,0] = 1.

	for j in range(NN+1):
		K[0,j] = 1.

	for i in range(MM):
		for j in range(NN):

			ii = int(i/(2**n))
			jj = int(j/(2**n))

			increment = 0.
			for k in range(D):
				increment += (x[ii+1,k]-x[ii,k])*(y[jj+1,k]-y[jj,k])/factor

			K[i+1,j+1] = K[i,j+1] + K[i+1,j] + increment*K[i,j] - K[i,j]

	return K[MM,NN]


def sig_distance(double[:,:] x, double[:,:] y, int n=0):
	cdef double a = sig_kernel(x,x,n)
	cdef double b = sig_kernel(y,y,n)
	cdef double c = sig_kernel(x,y,n)
	return a + b - 2.*c


def sig_kernels_fb_batch(double[:,:,:] x, double[:,:,:] y, int n=0):

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

				K[l,i+1,j+1] = K[l,i,j+1] + K[l,i+1,j] + (increment-1.)*K[l,i,j]
				K_rev[l,i+1,j+1] = K_rev[l,i,j+1] + K_rev[l,i+1,j] + (increment_rev-1.)*K_rev[l,i,j]

	return K, K_rev


def sig_kernels_f_batch(double[:,:,:] x, double[:,:,:] y, int n=0):

	cdef int A = x.shape[0]
	cdef int M = x.shape[1]
	cdef int N = y.shape[1]
	cdef int D = x.shape[2]

	cdef double increment
	cdef double factor = 2**(2*n)

	cdef int i, j, k, l, ii, jj
	cdef int MM = (2**n)*(M-1)
	cdef int NN = (2**n)*(N-1)

	cdef double[:,:,:] K = np.zeros((A,MM+1,NN+1), dtype=np.float64)

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
				increment_inv = 0.
				for k in range(D):
					increment += (x[l,ii+1,k]-x[l,ii,k])*(y[l,jj+1,k]-y[l,jj,k])/factor

				K[l,i+1,j+1] = K[l,i,j+1] + K[l,i+1,j] + (increment-1.)*K[l,i,j]

	return K