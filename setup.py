import io
import os
import setuptools
from Cython.Build import cythonize

here = os.path.realpath(os.path.dirname(__file__))

name = 'sigkernel'

version = '0.0.1'

author = 'Cristopher Salvi'

author_email = 'crispitagorico@gmail.com'

description = "Differentiable signature-PDE-kernel computations for PyTorch with GPU support."

with io.open(os.path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
    readme = f.read()

url = "https://github.com/crispitagorico/sigkernel"

license = "Apache-2.0"

classifiers = ["Intended Audience :: Developers",
               "Intended Audience :: Financial and Insurance Industry",
               "Intended Audience :: Information Technology",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: Apache Software License",
               "Natural Language :: English",
               "Operating System :: MacOS :: MacOS X",
               "Operating System :: Microsoft :: Windows",
               "Operating System :: Unix",
               "Programming Language :: Python :: 3",
               "Programming Language :: Python :: Implementation :: CPython",
               "Topic :: Scientific/Engineering :: Artificial Intelligence",
               "Topic :: Scientific/Engineering :: Information Analysis",
               "Topic :: Scientific/Engineering :: Mathematics"]

python_requires = "~=3.6"

install_requires = ["numba >= 0.50", "torch >= 1.6.0", "scikit-learn", "scipy", "tslearn", "iisignature", "joblib", "tqdm"]

ext_modules = [
    setuptools.Extension(
        name="cython_backend",
        sources=["sigkernel/cython_backend.pyx"],
        #extra_compile_args=['/openmp'],
        #extra_link_args=['/openmp'],
    )
]

setuptools.setup(name=name,
                 version=version,
                 author=author,
                 author_email=author_email,
                 maintainer=author,
                 maintainer_email=author_email,
                 description=description,
                 long_description=readme,
                 url=url,
                 license=license,
                 classifiers=classifiers,
                 zip_safe=False,
                 python_requires=python_requires,
                 install_requires=install_requires,
                 ext_modules=cythonize(ext_modules),
                 packages=[name])