#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 12:09:17 2020

@author: nate
"""

""" test use of scipy rotation """

# from pyTex.utils import XYZtoSPH, normalize, symmetrise
# import numpy as np
# import matplotlib.pyplot as plt

# hkl = [1,1,0]
# hkl2 = [3,1,2]

# symHKL = symmetrise('m-3m',hkl)
# normHKL = normalize(symHKL)

# symHKL2 = symmetrise('m-3m', hkl2)
# normHKL2 = normalize(symHKL2)

# sphProj = XYZtoSPH(normHKL,proj='stereo')
# sphProj2 = XYZtoSPH(normHKL2,proj='stereo')

# fig = plt.figure()
# ax = fig.add_subplot(111,polar=True)

# ax.scatter(sphProj[:,0],sphProj[:,1])
# ax.scatter(sphProj2[:,0],sphProj2[:,1])
# ax.set_yticklabels([])
# ax.set_xticklabels([])
# ax.set_ylim([0,1])
# ax.grid(False)

# plt.savefig('test.png', transparent=True)

from scipy.spatial.transform import Rotation as R

test = R.from_euler('x',90,degrees=True)