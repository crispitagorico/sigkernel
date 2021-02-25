<h1 align='center'>sigkernel</h1>
<h2 align='center'>Differentiable GPU-capable computations for the signature-PDE-kernel</h2>

This library provides differentiable GPU-capable computation in PyTorch for the [signature-PDE-kernel](https://arxiv.org/abs/2006.14794). Automatic differentiation is done efficiently without backpropagating through the PDE solver.

This allows to build state-of-the-art kernel-methods such as SVM or Gaussian Processes for irregularly sampled multivariate time series.

---

## Installation

```bash
pip install git+https://github.com/crispitagorico/sigkernel
```

Requires PyTorch >=1.6.0, numba >= 0.50 and Cython >= 0.29.

## Examples
To run the specific examples in folder `./examples`, install the requirements with

+ `pip install -r requirements.txt`

## Citation

```bibtex
@article{cass2020computing,
  title={The Signature Kernel is the solution of a Goursat PDE},
  author={Salvi, Cristopher and Cass, Thomas and Lyons, Terry and Yang, Weixin},
  journal={arXiv preprint arXiv:2006.14794},
  year={2020}
}
```

<!-- 
-->

