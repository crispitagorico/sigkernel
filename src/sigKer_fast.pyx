# cython: boundscheck=False
# cython: wraparound=False

import numpy as np

def forward_step(double k_00, double k_01, double k_10, double increment):
	return k_10 + k_01 + k_00*(increment-1.)

def forward_step_explicit(double k_00, double k_01, double k_10, double increment):
	return (k_10 + k_01)*(1.+0.5*increment+(1./12)*increment**2) - k_00*(1.-(1./12)*increment**2)

def forward_step_implicit(double k_00, double k_01, double k_10, double increment):
	return k_01+k_10-k_00 + ((0.5*increment)/(1.-0.25*increment))*(k_01+k_10)

def forward_step_gradient(double k_00, double k_01, double k_10, double increment, double k_00_, double k_01_, double k_10_, double k_11_, double increment_):
	return k_10 + k_01 + k_00*(increment-1.) + 0.25*increment_*(k_00_ + k_01_ + k_10_ + k_11_)

def forward_step_explicit_gradient(double k_00, double k_01, double k_10, double increment, double k_00_, double k_01_, double k_10_, double k_11_, double increment_):
	return (k_10 + k_01)*(1.+0.5*increment+(1./12)*increment**2) - k_00*(1.-(1./12)*increment**2) + 0.25*increment_*(k_00_ + k_01_ + k_10_ + k_11_)

def forward_step_implicit_gradient(double k_00, double k_01, double k_10, double increment, double k_00_, double k_01_, double k_10_, double k_11_, double increment_):
	return k_01+k_10-k_00 + ((0.5*increment)/(1.-0.25*increment))*(k_01+k_10) + 0.25*increment_*(k_00_ + k_01_ + k_10_ + k_11_)


def sig_kernel(double[:,:] x, double[:,:] y, int n=0, int solver=0, bint full=False):

	cdef int M = x.shape[0]
	cdef int N = y.shape[0]
	cdef int D = x.shape[1]

	cdef double increment
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
			for k in range(D):
				increment = increment + (x[ii+1,k]-x[ii,k])*(y[jj+1,k]-y[jj,k])/factor 

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


def sig_distance(double[:,:] x, double[:,:] y, int n=0, int solver=0):
	cdef double a = sig_kernel(x,x,n,solver)
	cdef double b = sig_kernel(y,y,n,solver)
	cdef double c = sig_kernel(x,y,n,solver)
	return a + b - 2.*c


def mmd_distance(double[:,:,:] x, double[:,:,:] y, int n=0):
	cdef double[:,:] K_XX = Gram_matrix(x,x,n,sym=True)
	cdef double[:,:] K_YY = Gram_matrix(y,y,n,sym=True)
	cdef double[:,:] K_XY = Gram_matrix(x,y,n,sym=False)
	return (np.mean(K_XX) + np.mean(K_YY) - 2.*np.mean(K_XY))**(0.5)


def Gram_matrix(double[:,:,:] x, double[:,:,:] y, int n=0, bint sym=False):

	cdef int A = x.shape[0]
	cdef int B = y.shape[0]
	cdef int M = x.shape[1]
	cdef int N = y.shape[1]
	cdef int D = x.shape[2]

	cdef double increment
	cdef double factor = 2**(2*n)

	cdef int i, j, k, l, ii, jj
	cdef int MM = (2**n)*(M-1)
	cdef int NN = (2**n)*(N-1)

	cdef double[:,:,:,:] K = np.zeros((A,B,MM+1,NN+1), dtype=np.float64)

	if sym:
		for l in range(A):
			for m in range(l,A):

				for i in range(MM+1):
					K[l,m,i,0] = 1.
					K[m,l,i,0] = 1.
	
				for j in range(NN+1):
					K[l,m,0,j] = 1.
					K[m,l,0,j] = 1.

				for i in range(MM):
					for j in range(NN):

						ii = int(i/(2**n))
						jj = int(j/(2**n))

						increment = 0.
						for k in range(D):
							increment += (x[l,ii+1,k]-x[l,ii,k])*(y[m,jj+1,k]-y[m,jj,k])/factor
						
						K[l,m,i+1,j+1] = forward_step_implicit(K[l,m,i,j], K[l,m,i,j+1], K[l,m,i+1,j], increment)
						K[m,l,i+1,j+1] = K[l,m,i+1,j+1]


	else:
		for l in range(A):
			for m in range(B):

				for i in range(MM+1):
					K[l,m,i,0] = 1.
	
				for j in range(NN+1):
					K[l,m,0,j] = 1.

				for i in range(MM):
					for j in range(NN):

						ii = int(i/(2**n))
						jj = int(j/(2**n))

						increment = 0.
						for k in range(D):
							increment += (x[l,ii+1,k]-x[l,ii,k])*(y[m,jj+1,k]-y[m,jj,k])/factor
	
						K[l,m,i+1,j+1] = forward_step_implicit(K[l,m,i,j], K[l,m,i,j+1], K[l,m,i+1,j], increment)
	
	return np.array(K[:,:,MM,NN])


def sig_kernel_batch(double[:,:,:] x, double[:,:,:] y, int n=0, int solver=0, bint gradients=True):

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

	if gradients:
		
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

					if solver==2:
						K[l,i+1,j+1] = forward_step_implicit(K[l,i,j], K[l,i,j+1], K[l,i+1,j], increment)
						K_rev[l,i+1,j+1] = forward_step_implicit(K_rev[l,i,j], K_rev[l,i,j+1], K_rev[l,i+1,j], increment_rev)
					elif solver==1:
						K[l,i+1,j+1] = forward_step(K[l,i,j], K[l,i,j+1], K[l,i+1,j], increment)
						K_rev[l,i+1,j+1] = forward_step(K_rev[l,i,j], K_rev[l,i,j+1], K_rev[l,i+1,j], increment_rev)
					elif solver==0:
						K[l,i+1,j+1] = forward_step(K[l,i,j], K[l,i,j+1], K[l,i+1,j], increment)
						K_rev[l,i+1,j+1] = forward_step(K_rev[l,i,j], K_rev[l,i,j+1], K_rev[l,i+1,j], increment_rev)

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

					if solver==2:
						K[l,i+1,j+1] = forward_step_implicit(K[l,i,j], K[l,i,j+1], K[l,i+1,j], increment)
					elif solver==1:
						K[l,i+1,j+1] = forward_step_explicit(K[l,i,j], K[l,i,j+1], K[l,i+1,j], increment)
					elif solver==0:
						K[l,i+1,j+1] = forward_step(K[l,i,j], K[l,i,j+1], K[l,i+1,j], increment)


	if gradients:
		return np.array(K), np.array(K_rev)
	return np.array(K)


def sig_kernel_batch_(double[:,:,:] x, double[:,:,:] y, int n=0, int solver=0, bint gradients=True):

	cdef int A = x.shape[0]
	cdef int M = x.shape[1]
	cdef int N = y.shape[1]
	cdef int D = x.shape[2]

	cdef double increment, inc_Y
	cdef double factor = 2**(2*n)

	cdef int i, j, k, l, ii, jj, m, d
	cdef int MM = (2**n)*(M-1)
	cdef int NN = (2**n)*(N-1)

	cdef double[:,:,:] K = np.zeros((A,MM+1,NN+1), dtype=np.float64)
	cdef double[:,:,:,:,:] K_rev = np.zeros((A,M-1,D,MM+1,NN+1), dtype=np.float64)

	if gradients:
		
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
					increment_rev = 0.
					for k in range(D):
						increment += (x[l,ii+1,k]-x[l,ii,k])*(y[l,jj+1,k]-y[l,jj,k])/factor

					if solver==2:
						K[l,i+1,j+1] = forward_step_implicit(K[l,i,j], K[l,i,j+1], K[l,i+1,j], increment)
						for m in range(M-1):
							for d in range(D):
								if m==ii:
									inc_Y = (y[l,jj+1,d]-y[l,jj,d])/factor
								else:
									inc_Y = 0.
								K_rev[l,m,d,i+1,j+1] = forward_step_implicit_gradient(K_rev[l,m,d,i,j], K_rev[l,m,d,i,j+1], K_rev[l,m,d,i+1,j], increment, K[l,i,j], K[l,i+1,j], K[l,i,j+1], K[l,i+1,j+1], inc_Y)
					elif solver==1:
						K[l,i+1,j+1] = forward_step_explicit(K[l,i,j], K[l,i,j+1], K[l,i+1,j], increment)
						for m in range(M-1):
							for d in range(D):
								if m==ii:
									inc_Y = (y[l,jj+1,d]-y[l,jj,d])/factor
								else:
									inc_Y = 0.
								K_rev[l,m,d,i+1,j+1] = forward_step_gradient(K_rev[l,m,d,i,j], K_rev[l,m,d,i,j+1], K_rev[l,m,d,i+1,j], increment, K[l,i,j], K[l,i+1,j], K[l,i,j+1], K[l,i+1,j+1], inc_Y)
					elif solver==0:
						K[l,i+1,j+1] = forward_step(K[l,i,j], K[l,i,j+1], K[l,i+1,j], increment)
						for m in range(M-1):
							for d in range(D):
								if m==ii:
									inc_Y = (y[l,jj+1,d]-y[l,jj,d])/factor
								else:
									inc_Y = 0.
								K_rev[l,m,d,i+1,j+1] = forward_step_gradient(K_rev[l,m,d,i,j], K_rev[l,m,d,i,j+1], K_rev[l,m,d,i+1,j], increment, K[l,i,j], K[l,i+1,j], K[l,i,j+1], K[l,i+1,j+1], inc_Y)
		
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

					if solver==2:
						K[l,i+1,j+1] = forward_step_implicit(K[l,i,j], K[l,i,j+1], K[l,i+1,j], increment)
					elif solver==1:
						K[l,i+1,j+1] = forward_step_explicit(K[l,i,j], K[l,i,j+1], K[l,i+1,j], increment)
					elif solver==0:
						K[l,i+1,j+1] = forward_step(K[l,i,j], K[l,i,j+1], K[l,i+1,j], increment)

	if gradients:
		return np.array(K), np.array(K_rev)
	return np.array(K)