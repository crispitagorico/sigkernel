from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "cython_backend",
        ["cython_backend.pyx"],
        #extra_compile_args=['/openmp'],
        #extra_link_args=['/openmp'],
    )
]

setup(
    name='cython_backend',
    ext_modules=cythonize(ext_modules),
)