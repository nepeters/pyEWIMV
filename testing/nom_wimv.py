#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 12:26:45 2019

@author: nate
"""

#testing nomad data with WIMV

import sys,os

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0,'/home/nate/wimv')

from classes import poleFigure, bunge
from inversion import wimv
from utils.orientation import eu2om

crystalSym = 'm-3m'
sampleSym = '1'
cellSize = np.deg2rad(5)

hkls = []
files = []

datadir = "/home/nate/wimv/Data/NOMAD Aluminum - no abs/pole figures/combined"

for file in os.listdir(datadir):
    
    pfName = file.split(')')[0].split('(')[1]
    
    try:
        hkls.append(tuple([int(c) for c in pfName]))
        files.append(os.path.join(datadir,file))
    except: #not hkls
        continue
    
    sortby = [sum([c**2 for c in h]) for h in hkls]
    hkls = [x for _, x in sorted(zip(sortby,hkls), key=lambda pair: pair[0])]
    files = [x for _, x in sorted(zip(sortby,files), key=lambda pair: pair[0])]
    
chi = R.from_euler('x',(13),degrees=True).as_dcm() 
om = R.from_euler('y',(90),degrees=True).as_dcm() 
phi = R.from_euler('z',(-88),degrees=True).as_dcm()

rot = chi.dot(phi).dot(om)
pf = poleFigure(files,hkls,crystalSym,'nd')
pf.rotate(rot)

pf.interpolate(cellSize,intMethod='linear')
pf.normalize()
#pf.plot(pfs=3)

y = pf.y
h = pf.symHKL

#od = bunge(cellSize, crystalSym, sampleSym)
#
#test = wimv(pf, od)

# %%


