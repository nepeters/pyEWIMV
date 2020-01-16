#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:52:34 2020

@author: nate
"""

import sys
import numpy as np

sys.path.append('/mnt/2E7481AA7481757D/Users/Nate/pyTex/')

from pyTex import bunge

cellSize = np.deg2rad(5)
crystalSym = 'm-3m'
sampleSym = '1'

od = bunge(cellSize,crystalSym,sampleSym)
cellVol = od.cellVolume