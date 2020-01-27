#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 21:35:33 2020

@author: nate
"""

"""
MAUD ODF import 
"""

from pyTex import bunge
import numpy as np

crystalSym = 'm-3m'
sampleSym = '1'
cellSize = np.deg2rad(5)

od_file = '/mnt/c/Users/Nate/Dropbox/ORNL/EWIMVvsMTEX/MAUD EWIMV exports/NOMAD_Al_4Datasets_NoSSabs_odf_5res'

maud_od = bunge.loadMAUD(od_file,
                         np.deg2rad(5),
                         'm-3m',
                         '1')

cV = maud_od.res * maud_od.res * ( np.cos( maud_od.Phi - ( maud_od.res/2 ) ) - np.cos( maud_od.Phi + ( maud_od.res/2 ) ) )

print(maud_od.index(cellVolume=cV))
print(maud_od.entropy(cellVolume=cV))
maud_od.plot3d()
