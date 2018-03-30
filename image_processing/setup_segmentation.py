#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import shutil

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension('segmentation',
                             sources=['segmentation.pyx'],
                             include_dirs=[numpy.get_include()])],
)

# Run from terminal: python setup_segmentation.py build_ext --inplace