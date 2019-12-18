#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:13:14 2019

@author: nate
"""


from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import build_ext
import numpy

ext_modules = [ Extension("metric",
                          ["metric.pyx"],
                          libraries=["m"],
                          extra_compile_args = ["-ffast-math"])]

setup(name="metric",
      cmdclass = {"build_ext": build_ext},
      include_dirs=[numpy.get_include()],
      ext_modules = ext_modules)