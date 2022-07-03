<h1 align='center'>sigkernel</h1>
<h2 align='center'>Differentiable computations for the signature-PDE-kernel on CPU and GPU</h2>

This library provides differentiable computation in PyTorch for the [signature-PDE-kernel](https://arxiv.org/abs/2006.14794) both on CPU and GPU. Automatic differentiation is done efficiently by solving a second "adjoint" PDE so without backpropagating through the PDE solver.

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
signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)


# Synthetic data
batch, len_x, len_y, dim = 5, 10, 20, 2
X = torch.rand((batch,len_x,dim), dtype=torch.float64, device='cuda') # shape (batch,len_x,dim)
Y = torch.rand((batch,len_y,dim), dtype=torch.float64, device='cuda') # shape (batch,len_y,dim)
G = torch.rand((batch,len_x,dim), dtype=torch.float64, device='cuda') # shape (batch,len_x,dim)


# Compute signature kernel "batch-wise" (i.e. k(x_1,y_1),...,k(x_batch, y_batch))
K = signature_kernel.compute_kernel(X,Y)


# Compute signature kernel and directional derivative along a batch of paths g,  i.e. D_{g_1}k(x_1,y_1),...,D_{g_batch}k(x_batch, y_batch)), 
# where the directional derivatives are with respect to the first variable.
K, K_diff = signature_kernel.compute_kernel_and_derivative(X,Y,G)


# Compute signature kernel Gram matrix (i.e. k(x_i,y_j) for i,j=1,...,batch), also works for different batch_x != batch_y)
K_Gram = signature_kernel.compute_Gram(X,Y,sym=False)


# Compute MMD distance between samples x ~ X and samples y ~ Y, where X,Y are two distributions on path space...
MMD = signature_kernel.compute_mmd(X,Y)
# ... and to backpropagate through the MMD distance simply call .backward(), like any other PyTorch loss function
MMD.backward()


# Compute scoring rule between X and a sample path y, i.e. S_sig(X,y) = E[k(X,X)] - 2E[k(X,y] ...
y = Y[0]
scoring_rule = signature_kernel.compute_scoring_rule(X,y)
# ... and expected scoring rule between X and Y, i.e. S(X,Y) = E_Y[S_sig(X,y)]
scoring_rule = signature_kernel.compute_expected_scoring_rule(X,y)
```

## Examples for paper [The signature kernel is the solution of a Goursat PDE](https://arxiv.org/abs/2006.14794)
To run the specific examples navigate to folder `./examples` and install the requirements with

+ `pip install -r requirements.txt`

#### UEA time series classification
To train all models and all datasets run (takes ~8 hours)

+ `python3 time_series_classification.py --train`

To test all models and all datasets run 

+ `python3 time_series_classification.py --test`

To print results final results run

+ `python3 time_series_classification.py --print`

#### Bitcoin prices predictions
Jupyter notebook `bitcoin_predictions.ipynb`.

#### Recombination
Jupyetr notebook `recombination.ipynb`.


## Citation

```bibtex
@article{salvi2020computing,
  title={The Signature Kernel is the solution of a Goursat PDE},
  author={Salvi, Cristopher and Cass, Thomas and Foster, James and Lyons, Terry and Yang, Weixin},
  journal={arXiv preprint arXiv:2006.14794},
  year={2020}
}
```

<!-- 
-->

