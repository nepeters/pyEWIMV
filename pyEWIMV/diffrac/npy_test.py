#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 09:37:33 2020

@author: nate
"""

import sys,os

from neutronpy import Material
import numpy as np

sys.path.append('/home/nate/projects/pyTex/')
from pyTex.utils import symmetrise, normalize

dir_path = '/home/nate/projects/pyTex/'

""" peak-fitted pole figures """

hkls = []
files = []

# datadir = os.path.join(dir_path,'Data','NOMAD Aluminum - no abs','combined')
# datadir = os.path.join(dir_path,'Data','NOMAD Nickel - full abs - peak int','pole figures','combined')
datadir = os.path.join(dir_path,'Data','NOMAD Aluminum - no abs - peak int','combined')
# datadir = '/media/nate/2E7481AA7481757D/Users/Nate/Dropbox/ORNL/Texture/NRSF2/mtex_export'

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

crystalSym = 'm-3m'

symHKL = symmetrise(crystalSym, hkls)
symHKL = normalize(symHKL)

def_al = {'name': 'Al',
          'composition': [dict(ion='Al', pos=[0, 0, 0]),
                          dict(ion='Al', pos=[0.5, 0, 0.5]),
                          dict(ion='Al', pos=[0.5, 0.5, 0]),
                          dict(ion='Al', pos=[0, 0.5, 0.5])],
          'lattice': dict(abc=[4.0495, 4.0495, 4.0495], abg=[90, 90, 90]),
          'debye-waller': False,
          'massNorm': False}

al = Material(def_al)

str_fac = {}

for fi,fam in enumerate(symHKL):
    
    str_fac[hkls[fi]] = []
    
    for h in fam:
        
        str_fac[hkls[fi]].append( np.abs( al.calc_nuc_str_fac(h) )**2 )
        
    str_fac[hkls[fi]] = np.average(str_fac[hkls[fi]])
    
    
#normalize
norm = 1 / np.max(list(str_fac.values()))
    
for fi,fam in enumerate(symHKL):
    
    str_fac[hkls[fi]] *= norm