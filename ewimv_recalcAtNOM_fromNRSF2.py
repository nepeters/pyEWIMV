#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 22:10:29 2020

@author: nate
"""

"""
EWIMV arbitrary grid
"""

"""
Sample coordnate system: z || up, x ⟂ y
Crystal coordinate system: z || [001], x || [100], y || [010]
this goes scattering vector -> intersection in bunge
"""

import os,sys

import numpy as np
from scipy.spatial.transform import Rotation as R

from pyTex import poleFigure, bunge
from pyTex.inversion import e_wimv

dir_path = os.path.dirname(os.path.realpath('__file__'))

P = 1

#sample information
sampleName = 'recalcAtNOM_fromNRSF2'
rad_type = 'nd'
def_al = {'name': 'Al',
          'composition': [dict(ion='Al', pos=[0, 0, 0]),
                          dict(ion='Al', pos=[0.5, 0, 0.5]),
                          dict(ion='Al', pos=[0.5, 0.5, 0]),
                          dict(ion='Al', pos=[0, 0.5, 0.5])],
          'lattice': dict(abc=[4.0465, 4.0465, 4.0465], abg=[90, 90, 90]),
          'debye-waller': False,
          'massNorm': False}

crystalSym = 'm-3m'
sampleSym = '1'
cellSize = np.deg2rad(5)
od = bunge(cellSize, crystalSym, sampleSym)

#tube radius
theta = np.deg2rad(7)
#tube exponent
tube_exp = 1

""" pole figures exported from MTEX """

hkls = []
files = []

datadir = '/mnt/c/Users/Nate/Dropbox/ORNL/EWIMVvsMTEX/MTEX_recalcAtNOM_fromNRSF2'

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
    
pf = poleFigure(files,hkls,crystalSym,'mtex')

#perform E-WIMV iterations
recalc_pf, calc_od = e_wimv( pf, od, theta, tube_exp, rad_type, def_al, iterations=8 )
final_iter = max(list(calc_od.keys()))

#recalculate (111), (200), (220)
hkls = [(1,1,1),(2,0,0),(2,2,0)]
recalc_pf_new = calc_od[final_iter-1].calcPF( hkls, theta, tube_exp, tube_proj=True )

#plot recalculated pole figures
cl = np.arange(0,6.5,0.5)
# recalc_pf[final_iter-1].plot(pfs=3,contourlevels=cl,cmap='viridis_r',proj='none')
recalc_pf_new.plot(contourlevels=cl,cmap='viridis',proj='earea')

# #plot ODF section
# # calc_od[iterations-1].sectionPlot('phi2',np.deg2rad(90))

# #calculate texture index & entropy
# print(sampleName)
# print('iterations: '+str(final_iter-1))
# print(calc_od[final_iter-1].index())
# print(calc_od[final_iter-1].entropy())

## export data
# calc_od[final_iter-1].export('/mnt/c/Users/Nate/Dropbox/ORNL/EWIMVvsMTEX/EWIMV exports (abs corr)/'+sampleName+'.odf',vol_norm=True)
# recalc_pf[final_iter-1].export('/mnt/c/Users/Nate/Dropbox/ORNL/EWIMVvsMTEX/EWIMV exports (abs corr)/',sampleName=sampleName)
# recalc_pf_new.export('/mnt/c/Users/Nate/Dropbox/ORNL/EWIMVvsMTEX/EWIMV exports (abs corr)/',sampleName=sampleName)



