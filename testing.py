#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 12:09:17 2020

@author: nate
"""

""" test use of scipy rotation """

from pyTex.base import bunge
from scipy.spatial.transform import Rotation as rot
import numpy as np

crystalSym = 'm-3m'
sampleSym = '1'
cellSize = np.deg2rad(5)

od = bunge(cellSize, crystalSym, sampleSym)

phi1 = od.phi1cen
Phi  = od.Phicen
phi2 = od.phi2cen

test = rot.from_euler('ZXZ', (phi1, Phi, phi2))