#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:22:39 2019

@author: nate
"""

import os,sys

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import rowan as quat

from pymatgen.core import Lattice, Structure

sys.path.insert(0,'/media/nate/2E7481AA7481757D/Users/Nate/Dropbox/wimv2/')

from pyTex import poleFigure, bunge
from pyTex.orientation import eu2quat, quat2eu 
from pyTex.utils import symmetrise, normalize, genSym, XYZtoSPH
from pyTex.diffrac import calc_XRDreflWeights

dir_path = os.path.dirname(os.path.realpath('__file__'))

P = 1

crystalSym = 'm-3m'
sampleSym = '1'
cellSize = np.deg2rad(5)
theta = np.deg2rad(7.5)

hkls = []
files = []

# datadir = os.path.join(dir_path,'Data','NOMAD Aluminum - no abs','pole figures','combined')
datadir = '/media/nate/2E7481AA7481757D/Users/Nate/Dropbox/ORNL/Texture/NRSF2/mtex_export'

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

""" rotate """

# rot = R.from_euler('XZY',(13,-88,90), degrees=True).as_dcm()

pf = poleFigure(files,hkls,crystalSym,'nd')
# pf.rotate(rot)

test = np.copy(pf.y[0])
test = test[:,[0,1,2]]

test_sph = XYZtoSPH(test,upperOnly=True)

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
ax.set_ylim([0,np.pi/2])

ax.scatter(test_sph[:,0],test_sph[:,1])
