# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport exp
import numpy as np

from cython.parallel import prange


def forward_step(double k_00, double k_01, double k_10, double increment):
	return k_10 + k_01 + k_00*(increment-1.)

def forward_step_explicit(double k_00, double k_01, double k_10, double increment):
	return (k_10 + k_01)*(1.+0.5*increment+(1./12)*increment**2) - k_00*(1.-(1./12)*increment**2)

def forward_step_implicit(double k_00, double k_01, double k_10, double increment):
	# return k_01+k_10-k_00 + ((0.5*increment)/(1.-0.25*increment))*(k_01+k_10)
	return k_01+k_10-k_00 + (exp(0.5*increment)-1.)*(k_01+k_10)


def sig_kernel(double[:,:] x, double[:,:] y, int n=0, int solver=0, bint full=False, bint rbf=False, double sigma=1.):

	cdef int M = x.shape[0]
	cdef int N = y.shape[0]
	cdef int D = x.shape[1]

	cdef double increment
	cdef double d1, d2, d3, d4 
	cdef double factor = 2**(2*n)

	cdef int i, j, k, l, ii, jj

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
			d1 = 0.
			d2 = 0.
			d3 = 0.
			d4 = 0.
			for k in range(D):

				if rbf:
					d1 = d1 + (x[ii+1,k]-y[jj+1,k])**2
					d2 = d2 + (x[ii+1,k]-y[jj,k])**2
					d3 = d3 + (x[ii,k]-y[jj+1,k])**2
					d4 = d4 + (x[ii,k]-y[jj,k])**2
				else:
					increment = increment + (x[ii+1,k]-x[ii,k])*(y[jj+1,k]-y[jj,k])

			if rbf:
				increment = ( exp(-d1/sigma) - exp(-d2/sigma) - exp(-d3/sigma) + exp(-d4/sigma) )/factor
			else:
				increment = increment/factor

			if solver==0:
				K[i+1,j+1] = forward_step(K[i,j], K[i,j+1], K[i+1,j], increment)
			elif solver==1:
				K[i+1,j+1] = forward_step_explicit(K[i,j], K[i,j+1], K[i+1,j], increment)
			else:
				K[i+1,j+1] = forward_step_implicit(K[i,j], K[i,j+1], K[i+1,j], increment)

	if full:
		return np.array(K)
	else:
		return K[MM,NN]


def sig_distance(double[:,:] x, double[:,:] y, int n=0, int solver=0, bint rbf=False, double sigma=1.):
	cdef double a = sig_kernel(x,x,n,solver,False,rbf,sigma)
	cdef double b = sig_kernel(y,y,n,solver,False,rbf,sigma)
	cdef double c = sig_kernel(x,y,n,solver,False,rbf,sigma)
	return a + b - 2.*c


def mmd_distance(double[:,:,:] x, double[:,:,:] y, int n=0, int solver=0, bint rbf=False, double sigma=1.):
	cdef double[:,:] K_XX = sig_kernel_Gram_matrix(x,x,n,solver,True,False,rbf,sigma)
	cdef double[:,:] K_YY = sig_kernel_Gram_matrix(y,y,n,solver,True,False,rbf,sigma)
	cdef double[:,:] K_XY = sig_kernel_Gram_matrix(x,y,n,solver,False,False,rbf,sigma)
	return np.mean(K_XX) + np.mean(K_YY) - 2.*np.mean(K_XY)


def sig_kernel_batch_varpar(double[:,:,:] G_static):

	cdef int A = G_static.shape[0]
	cdef int M = G_static.shape[1]
	cdef int N = G_static.shape[2]
	cdef int i, j, l
	
	cdef double[:,:,:] K = np.zeros((A,M+1,N+1), dtype=np.float64)
		
	for l in range(A):

		for i in range(M+1):
			K[l,i,0] = 1.
	
		for j in range(N+1):
			K[l,0,j] = 1.

		for i in range(M):
			for j in range(N):

				# K[l,i+1,j+1] = K[l,i+1,j] + K[l,i,j+1] + K[l,i,j]*(G_static[l,i,j] - 1.)
				K[l,i+1,j+1] = (K[l,i+1,j] + K[l,i,j+1])*(1. + 0.5*G_static[l,i,j]+(1./12)*G_static[l,i,j]**2) - K[l,i,j]*(1. - (1./12)*G_static[l,i,j]**2)
				# K[l,i+1,j+1] = K[l,i+1,j] + K[l,i,j+1] - K[l,i,j] + (exp(0.5*G_static[l,i,j])-1.)*(K[l,i+1,j] + K[l,i,j+1])

	return np.array(K)


def sig_kernel_Gram_varpar(double[:,:,:,:] G_static, bint sym=False):

	cdef int A = G_static.shape[0]
	cdef int B = G_static.shape[1]
	cdef int M = G_static.shape[2]
	cdef int N = G_static.shape[3]
	cdef int i, j, l, m

	cdef double[:,:,:,:] K = np.zeros((A,B,M+1,N+1), dtype=np.float64)

	if sym:
		# for l in prange(A,nogil=True):
		for l in range(A):
			for m in range(l,A):

				for i in range(M+1):
					K[l,m,i,0] = 1.
					K[m,l,i,0] = 1.
	
				for j in range(N+1):
					K[l,m,0,j] = 1.
					K[m,l,0,j] = 1.

				for i in range(M):
					for j in range(N):

						#K[l,m,i+1,j+1] = K[l,m,i+1,j] + K[l,m,i,j+1] + K[l,m,i,j]*(G_static[l,m,i,j]-1.)
						K[l,m,i+1,j+1] = (K[l,m,i+1,j] + K[l,m,i,j+1])*(1.+0.5*G_static[l,m,i,j]+(1./12)*G_static[l,m,i,j]**2) - K[l,m,i,j]*(1.-(1./12)*G_static[l,m,i,j]**2)
						#K[l,m,i+1,j+1] = K[l,m,i+1,j] + K[l,m,i,j+1] - K[l,m,i,j] + (exp(0.5*G_static[l,m,i,j])-1.)*(K[l,m,i+1,j] + K[l,m,i,j+1])

						K[m,l,j+1,i+1] = K[l,m,i+1,j+1]

	else:
		# for l in prange(A,nogil=True):
		for l in range(A):
			for m in range(B):

				for i in range(M+1):
					K[l,m,i,0] = 1.
	
				for j in range(N+1):
					K[l,m,0,j] = 1.

				for i in range(M):
					for j in range(N):

						#K[l,m,i+1,j+1] = K[l,m,i+1,j] + K[l,m,i,j+1] + K[l,m,i,j]*(G_static[l,m,i,j] - 1.)
						K[l,m,i+1,j+1] = (K[l,m,i+1,j] + K[l,m,i,j+1])*(1. + 0.5*G_static[l,m,i,j]+(1./12)*G_static[l,m,i,j]**2) - K[l,m,i,j]*(1. - (1./12)*G_static[l,m,i,j]**2)
						#K[l,m,i+1,j+1] = K[l,m,i+1,j] + K[l,m,i,j+1] - K[l,m,i,j] + (exp(0.5*G_static[l,m,i,j])-1.)*(K[l,m,i+1,j] + K[l,m,i,j+1])
	
	return np.array(K)
