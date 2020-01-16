#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:16:57 2019

@author: nate
"""

import sys,os
import numpy as np

filepath = os.path.dirname(os.path.abspath(__file__))

from pyTex import poleFigure as _poleFigure
from pyTex import bunge as _bunge
from pyTex.inversion import wimv

crystalSym = 'm-3m'
sampleSym = '1'
cellSize = np.deg2rad(2.5)
hkls = [(1,1,1),(2,0,0),(2,2,0)]

bkgd111path = os.path.join(filepath,'Data','XRD Aluminum','32.5_ bkgd.xrdml')
bkgd200path = os.path.join(filepath,'Data','XRD Aluminum','55_bkgd.xrdml')
bkgd220path = os.path.join(filepath,'Data','XRD Aluminum','55_bkgd.xrdml')

bkgds = [bkgd111path,bkgd200path,bkgd220path]
bkgd = _poleFigure(bkgds, hkls, crystalSym, 'xrdml',subtype='bkgd')

def111path = os.path.join(filepath,'Data','XRD Aluminum','defocus_38.xrdml')
def200path = os.path.join(filepath,'Data','XRD Aluminum','defocus_45.xrdml')
def220path = os.path.join(filepath,'Data','XRD Aluminum','defocus_65.xrdml')

defs = [def111path,def200path,def220path]
defocus = _poleFigure(defs, hkls, crystalSym, 'xrdml',subtype='defocus')

pf111path = os.path.join(filepath,'Data','XRD Aluminum','111pf_2T=38.xrdml')
pf200path = os.path.join(filepath,'Data','XRD Aluminum','200pf_2T=45.xrdml')
pf220path = os.path.join(filepath,'Data','XRD Aluminum','220pf_2theta=65.xrdml')

pfs = [pf111path,pf200path,pf220path]
pf = _poleFigure(pfs, hkls, crystalSym, 'xrdml')

pf.correct(bkgd=bkgd,defocus=defocus)
pf.normalize()

od = _bunge(cellSize, crystalSym, sampleSym)

pf_grid = pf.grid(full=True)

recalc_pf, calc_od, pf_od, od_pf, prnt_str = wimv(pf, od, ret_pointer=True )

recalc_pf[11].plot(cmap='viridis_r',contourlevels=np.arange(1,10), proj='none')

# %%

#fibre_out = {}
#
#for hi, h in enumerate(hkls):
#    
#    fibre_out[hi] = {}
#    
#    for pf_cell in pf_od[hi].keys():
#        
#        od_cells = np.array(pf_od[hi][pf_cell])
#        od_idx = np.unravel_index(od_cells.astype(int),od.bungeList.shape)
#        
#        temp = np.column_stack((od.phi1cen[od_idx], od.Phicen[od_idx], od.phi2cen[od_idx]))
#        
#        fibre_out[hi][pf_cell] = temp
#        
#import pickle
#f = open("file.pkl","wb")
#pickle.dump(fibre_out,f)
#f.close()


# %%

#""" plotting test """
#
#import mayavi.mlab as mlab
#mlab.figure(bgcolor=(1,1,1))
#
# ## grid cen ##
#gd = mlab.points3d(od.phi1cen,od.Phicen,od.phi2cen,scale_factor=1,mode='point',color=(1,0,0))
#gd.actor.property.render_points_as_spheres = True
#gd.actor.property.point_size = 5
#
# ## grid ##
#gd = mlab.points3d(od.phi1,od.Phi,od.phi2,scale_factor=1,mode='point',color=(0,0,0))
#gd.actor.property.render_points_as_spheres = True
#gd.actor.property.point_size = 3
