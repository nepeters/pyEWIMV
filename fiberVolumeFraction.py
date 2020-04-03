#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 19:09:44 2020

@author: nate
"""


## testing fiber

import numpy as np
from scipy.spatial.transform import Rotation as R
from pyTex.orientation import symmetrise as symOri
from pyTex.orientation import om2eu, eu2om
from pyTex.utils import genSymOps
from pyTex import bunge

od = bunge(np.deg2rad(5), 'm-3m', '1')

hkl = np.array([2, 1, 3])
hkl = np.divide(hkl, np.linalg.norm(hkl))

test = R.from_euler('ZXZ', [35.3,45.0,90.0], degrees=True)
testSym = symOri(test.as_matrix(),'m-3m', '1')

#try transpose
testSym = testSym.transpose((0,2,1))

# convert to euler
testSym_eu = om2eu(testSym)

# pick fundamental zone
fz = (testSym_eu[:,0] <= 2*np.pi) & (testSym_eu[:,1] <= np.pi/2) & (testSym_eu[:,2] <= np.pi/2)
fz_idx = np.nonzero(fz)
g_fz = testSym[fz_idx[0],:,:]

# generate crystal sym ops
crysSymOps = genSymOps('m-3m')
smplSymOps = genSymOps('1')

# create Nx3 array of grid points
eu_grid = np.array([od.phi1cen.flatten(),od.Phicen.flatten(),od.phi2cen.flatten()]).T
g_grid  = eu2om(eu_grid,out='mdarray_2')
g_grid  = g_grid.transpose((2,0,1))

trace = {}
misori = {}
mo_cell = []

for gi,g in enumerate(g_fz):    
    
    trace[gi] = []
    misori[gi] = []
    k = 0
    
    for crys_op in crysSymOps:
        
        # for smpl_op in smplSymOps:
    
        temp = g @ g_grid
        test = crys_op @ temp 
        trace[gi].append( np.trace( test,axis1=1,axis2=2 ) )
        
        #calculate misorientation
        mo = np.arccos( np.clip( (trace[gi][k] - 1)/2, -1, 1) )
        
        #criteria
        crit = np.where(mo <= np.deg2rad(10))
        # crit = np.argmin(mo)
        
        #store cell id, misorientation angle for each sym equiv.
        misori[gi].append( np.array( [crit[0], mo[crit]] ).T )
        k+=1
            
    # concatenate, pull true min from sym equiv.
    misori[gi] = np.vstack(misori[gi])
    # mo_cell.append( misori[gi][ np.argmin(misori[gi][:,1]), 0 ].astype(int) )
    mo_cell.append(np.unique(misori[gi],axis=0)[:,0].T)    
        
    # misori.append(np.argmin(np.arccos((np.vstack(trace)-1)/2)))
    # k+=1

mo_cell = np.unique(np.hstack(mo_cell).astype(int))

# %%

# plotting

import mayavi.mlab as mlab
from pyTex import bunge

od = bunge(np.deg2rad(5), 'm-3m', '1')

mlab.figure(bgcolor=(1,1,1))

gd = mlab.points3d(od.phi1cen,od.Phicen,od.phi2cen,mode='point',scale_factor=1,color=(0.25,0.25,0.25))
gd.actor.property.render_points_as_spheres = True
gd.actor.property.point_size = 2

pts = mlab.points3d(testSym_eu[fz_idx,0],testSym_eu[fz_idx,1],testSym_eu[fz_idx,2],mode='point',scale_factor=1,color=(0,1,0))
pts.actor.property.render_points_as_spheres = True
pts.actor.property.point_size = 10

pts = mlab.points3d(eu_grid[mo_cell,0],eu_grid[mo_cell,1],eu_grid[mo_cell,2],mode='point',scale_factor=1,color=(1,0,0))
pts.actor.property.render_points_as_spheres = True
pts.actor.property.point_size = 10