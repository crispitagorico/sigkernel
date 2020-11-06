from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "sigKer_fast",
        ["sigKer_fast.pyx"],
        #extra_compile_args=['/openmp'],
        #extra_link_args=['/openmp'],
    )
]

setup(
    name='sigKer-fast',
    ext_modules=cythonize(ext_modules),
)