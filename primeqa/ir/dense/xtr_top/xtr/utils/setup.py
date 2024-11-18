from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension("xtr_inference", ["xtr_inference.pyx"],
              include_dirs=[numpy.get_include()])
]

setup(
    ext_modules=cythonize(ext_modules)
)
