<h1 align='center'>sigkernel</h1>
<h2 align='center'>Differentiable computations for the signature-PDE-kernel on CPU and GPU</h2>

This library provides differentiable computation in PyTorch for the [signature-PDE-kernel](https://arxiv.org/abs/2006.14794) both on CPU and GPU. Automatic differentiation is done efficiently without backpropagating through the PDE solver.

This allows to build state-of-the-art kernel-methods such as Support Vector Machines or Gaussian Processes for high-dimensional, irregularly-sampled, multivariate time series.

---

## Installation

```bash
pip install git+https://github.com/crispitagorico/sigkernel.git
```

Requires PyTorch >=1.6.0, Numba >= 0.50 and Cython >= 0.29.

## How to use the library

```python
import torch
import sigkernel

# Specify the static kernel (for linear kernel use sigkernel.LinearKernel())
static_kernel = sigkernel.RBFKernel(sigma=0.5)

# Specify dyadic order for PDE solver (int > 0, default 0, the higher the more accurate but slower)
dyadic_order = 5

# Initialize the corresponding signature kernel
signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order, _naive_solver)

# Synthetic data
batch, len_x, len_y, dim = 5, 10, 20, 2
x = torch.rand((batch,len_x,dim), dtype=torch.float64, device='cuda') # shape (batch,len_x,dim)
y = torch.rand((batch,len_y,dim), dtype=torch.float64, device='cuda') # shape (batch,len_y,dim)

# Compute signature kernel "batch-wise" (i.e. k(x_1,y_1),...,k(x_batch, y_batch))
k_batch = signature_kernel.compute_kernel(x,y)

# Compute signature kernel Gram matrix (i.e. k(x_i,y_j) for i,j=1,...,batch), also works for different batch_x != batch_y)
k_batch = signature_kernel.compute_Gram(x,y,sym=False)

# Compute MMD distance between samples x ~ P and samples y ~ Q, where P,Q are two distributions on path space
d_mmd = signature_kernel.compute_mmd(x,y)

# and to backpropagate through the MMD distance simply call .backward(), like any other PyTorch loss function
d_mmd.backward()
```

## Examples for paper [The signature kernel is the solution of a Goursat PDE](https://arxiv.org/abs/2006.14794)
To run the specific examples navigate to folder `./examples` and install the requirements with

+ `pip install -r requirements.txt`

## Citation

```bibtex
@article{salvi2020computing,
  title={The signature kernel is the solution of a Goursat PDE},
  author={Salvi, Cristopher and Cass, Thomas and Foster, James and Lyons, Terry and Yang, Weixin},
  journal={arXiv preprint arXiv:2006.14794},
  year={2020}
}
```

<!-- 
-->

