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
sampleName = 'Al_NRSF2_5x7'
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

""" NRSF2 .jul """

#define pole figures
data_path = os.path.join(dir_path, 'Data', 'HB2B - Aluminum')
hkls = np.array([(2,2,2), (3,1,1), (4,0,0)])
pf222path = os.path.join(data_path, 'HB2B_exp129_3Chi_222.jul')
pf311path = os.path.join(data_path, 'HB2B_exp129_3Chi_311.jul')
pf400path = os.path.join(data_path, 'HB2B_exp129_3Chi_400.jul')

#load pole figures
pf = poleFigure([pf222path,pf311path,pf400path], hkls, crystalSym, 'jul')

rot = R.from_euler('XZX', (90,90,90), degrees=True).as_matrix()

""" peak-fitted pole figures """

# hkls = []
# files = []

# # datadir = os.path.join(dir_path,'Data','NOMAD Aluminum - no abs','combined')
# # datadir = os.path.join(dir_path,'Data','NOMAD Nickel - full abs - peak int','pole figures','combined')
# # datadir = os.path.join(dir_path,'Data','NOMAD Aluminum - no abs - peak int','combined')
# # datadir = '/media/nate/2E7481AA7481757D/Users/Nate/Dropbox/ORNL/Texture/NRSF2/mtex_export'
# # datadir = '/mnt/c/Users/Nate/pyReducePF/pole figures/pole figures peak int Al absCorr/combined'
# datadir = '/mnt/c/Users/Nate/pyReducePF/pole figures/pole figures integ int Al absCorr/combined'

# for file in os.listdir(datadir):
    
#     pfName = file.split(')')[0].split('(')[1]
    
#     try:
#         hkls.append(tuple([int(c) for c in pfName]))
#         files.append(os.path.join(datadir,file))
#     except: #not hkls
#         continue
    
#     sortby = [sum([c**2 for c in h]) for h in hkls]
#     hkls = [x for _, x in sorted(zip(sortby,hkls), key=lambda pair: pair[0])]
#     files = [x for _, x in sorted(zip(sortby,files), key=lambda pair: pair[0])]
    

# pf = poleFigure(files,hkls,crystalSym,'sparse')

# rot = R.from_euler('XZY',(13,-88,90), degrees=True).as_matrix()

#rotate pole figures
pf.rotate(rot)

#perform E-WIMV iterations
recalc_pf, calc_od = e_wimv( pf, od, theta, tube_exp, rad_type, def_al, iterations=8 )
final_iter = max(list(calc_od.keys()))

#recalculate (111), (200), (220)
hkls = [(1,1,1),(2,0,0),(2,2,0)]
recalc_pf_new = calc_od[final_iter-1].calcPF( hkls, theta, tube_exp, tube_proj=True )

#plot recalculated pole figures
cl = np.arange(0,10.5,0.5)
# recalc_pf[final_iter-1].plot(pfs=3,contourlevels=cl,cmap='viridis_r',proj='none')
recalc_pf_new.plot(contourlevels=cl,cmap='viridis_r',proj='none')

#plot ODF section
# calc_od[iterations-1].sectionPlot('phi2',np.deg2rad(90))

#calculate texture index & entropy
print(sampleName)
print('iterations: '+str(final_iter-1))
print(calc_od[final_iter-1].index())
print(calc_od[final_iter-1].entropy())

## export data
calc_od[final_iter-1].export('/mnt/c/Users/Nate/Dropbox/ORNL/EWIMVvsMTEX/EWIMV exports (abs corr)/'+sampleName+'.odf',vol_norm=True)
# recalc_pf[final_iter-1].export('/mnt/c/Users/Nate/Dropbox/ORNL/EWIMVvsMTEX/EWIMV exports (abs corr)/',sampleName=sampleName)
recalc_pf_new.export('/mnt/c/Users/Nate/Dropbox/ORNL/EWIMVvsMTEX/EWIMV exports (abs corr)/',sampleName=sampleName)



