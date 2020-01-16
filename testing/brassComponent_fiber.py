#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 21:08:50 2020

@author: nate
"""

import sys

import numpy as np
from scipy.spatial.transform import Rotation as R
import rowan as quat
from numba import jit,prange
from tqdm import tqdm

sys.path.append('/mnt/2E7481AA7481757D/Users/Nate/pyTex/')

from pyTex import bunge
from pyTex.orientation import quat2eu, eu2quat
from pyTex.utils import symmetrise, normalize

P = 1

crystalSym = 'm-3m'
sampleSym = '1'
cellSize = np.deg2rad(5)
theta = np.deg2rad(7)

hkls = (1,1,1)
symHKL = symmetrise(crystalSym, hkls)
symHKL = [normalize(symHKL)]

od = bunge(cellSize, crystalSym, sampleSym)

""" quaternion grid """
bungeAngs = np.zeros(( np.product(od.phi1cen.shape), 3 ))

for ii,i in enumerate(np.ndindex(od.phi1cen.shape)):
    
    bungeAngs[ii,:] = np.array((od.phi1cen[i],od.Phicen[i],od.phi2cen[i]))

qgrid = eu2quat(bungeAngs).T

omega = np.radians(np.arange(0,360+5,5))
""" use sklearn KDTree for reduction of points for query (euclidean) """
from sklearn.neighbors import KDTree
qgrid_pos = np.copy(qgrid)
qgrid_pos[qgrid_pos[:,0] < 0] *= -1
tree = KDTree(qgrid_pos)

rad = ( 1 - np.cos(theta) ) / 2
euc_rad = 4*np.sin(theta)**2

""" calculate y's for given component """
brass = R.from_euler('ZXZ',(0,45,0),degrees=True)
brass_y = brass.apply(symHKL[0])

@jit(nopython=True,parallel=True)
def quatMetricNumba(a, b):
    
    """ from DOI 10.1007/s10851-009-0161-2, #4 """
    
    dist = np.zeros((len(a),len(b)))
    
    for bi in prange(len(b)):
        
        dist[:,bi] = 1 - np.abs(np.dot(a,b[bi]))
    
    return dist

def calcFibre(symHKL,yset,qgrid,omega,rad,tree,euc_rad):
    
    fibre_e = {}
    fibre_q = {}
    
    nn_gridPts = {}
    nn_gridDist = {}
    
    egrid_trun = {}
        
    for fi,fam in enumerate(tqdm(symHKL)):
        
        fibre_e[fi] = {}
        fibre_q[fi] = {}
        
        nn_gridPts[fi] = {}
        nn_gridDist[fi] = {}
        
        egrid_trun[fi] = {}
        
        """ set proper iterator """
        if isinstance(yset,dict): it = yset[fi]
        else: it = yset
        
        q1_n = [quat.from_axis_angle(h, omega) for h in fam]
    
        for yi,y in enumerate(it):
            
            axis = np.cross(fam,y)
            angle = np.arccos(np.dot(fam,y))
            
            q0_n = quat.from_axis_angle(axis, angle)
            # q0_n = quat.normalize(q0)
            
            qfib = np.zeros((len(q1_n[0]),len(q0_n),4))
            
            for sym_eq,(qA,qB) in enumerate(zip(q0_n,q1_n)):
                
                temp = quat.multiply(qA, qB)
                
                qfib[:,sym_eq,:] = temp
              
            phi1, Phi, phi2 = quat2eu(qfib)
            
            phi1 = np.where(phi1 < 0, phi1 + 2*np.pi, phi1) #brnng back to 0 - 2pi
            Phi = np.where(Phi < 0, Phi + np.pi, Phi) #brnng back to 0 - pi
            phi2 = np.where(phi2 < 0, phi2 + 2*np.pi, phi2) #brnng back to 0 - 2pi
            
            eu_fib = np.stack( (phi1, Phi, phi2), axis=2 )
            eu_fib = np.reshape( eu_fib, (eu_fib.shape[0]*eu_fib.shape[1], eu_fib.shape[2]) ) #new method       
    
            fz = (eu_fib[:,0] < od._phi1max) & (eu_fib[:,1] < od._Phimax) & (eu_fib[:,2] < od._phi2max)
            fz_idx = np.nonzero(fz)
            
            fibre_e[fi][yi] = eu_fib[fz]
    
            fib_idx = np.unravel_index(fz_idx[0], (qfib.shape[0],qfib.shape[1]))
            
            fibre_q[fi][yi] = qfib[fib_idx]
            
            """ reduce geodesic query size """
            qfib_pos = np.copy(qfib[fib_idx])
            qfib_pos[qfib_pos[:,0] < 0] *= -1
            
            query = np.concatenate(tree.query_radius(qfib_pos,euc_rad))
            query_uni = np.unique(query)
            qgrid_trun = qgrid[query_uni]
            qgrid_trun_idx = np.arange(len(qgrid))[query_uni] #store indexes to retrieve original grid pts later
            
            """ distance calc """
            temp = quatMetricNumba(qgrid_trun,qfib[fib_idx])
            """ find tube """
            tube = (temp <= rad)
            temp = np.column_stack((np.argwhere(tube)[:,0],temp[tube]))
            
            """ round very small values """
            temp = np.round(temp, decimals=7)
            
            """ move values at zero to very small (1E-5) """
            temp[:,1] = np.where(temp[:,1] == 0, 1E-5, temp[:,1])
            
            """ sort by min distance """
            temp = temp[np.argsort(temp[:,1],axis=0)]
            """ return unique pts (first in list) """
            uni_pts = np.unique(temp[:,0],return_index=True)
            
            nn_gridPts[fi][yi] = qgrid_trun_idx[uni_pts[0].astype(int)]
            nn_gridDist[fi][yi] = temp[uni_pts[1],1]
            
            egrid_trun[fi][yi] = bungeAngs[query_uni]
            
    # return nn_gridPts, nn_gridDist
    return fibre_e, nn_gridPts

brassFibre, nn_gridPts  = calcFibre(symHKL,brass_y,qgrid,omega,rad,tree,euc_rad)

# %%

### FIBER PLOT ###

import mayavi.mlab as mlab

mlab.figure(bgcolor=(1,1,1))

pf_num = 0

## grid ##
gd = mlab.points3d(bungeAngs[:,0],bungeAngs[:,1],bungeAngs[:,2],scale_factor=1,mode='point',color=(0,0,0))
gd.actor.property.render_points_as_spheres = True
gd.actor.property.point_size = 3

## all fibres ##
allBrassFibre = []
for k,v in brassFibre[pf_num].items():
    tubePts = nn_gridPts[pf_num][k]
    allBrassFibre.append(v)
    # allBrassFibre.append(bungeAngs[tubePts.astype(int),:])
    
allBrassFibre = np.vstack(allBrassFibre)
gd = mlab.points3d(allBrassFibre[:,0],allBrassFibre[:,1],allBrassFibre[:,2],scale_factor=1,mode='point',color=(1,0,0))
gd.actor.property.render_points_as_spheres = True
gd.actor.property.point_size = 10

# ## lit point ##
# gd2 = mlab.points3d(0,0,0,scale_factor=1,mode='point',color=(1,0,0))
# gd2.actor.property.render_points_as_spheres = True
# gd2.actor.property.point_size = 7

# ## manual fibre ##
# gd3 = mlab.points3d(0,0,0,scale_factor=1,mode='point',color=(1,0,0))
# gd3.actor.property.render_points_as_spheres = True
# gd3.actor.property.point_size = 7

plt_list = list(brassFibre[pf_num].keys())
plt_list.sort()

# @mlab.animate(delay=100)
# def anim():
#     while True:
        
#         for yi in plt_list:
                
#             gd2.mlab_source.reset( x = brassFibre[pf_num][yi][:,0],
#                                    y = brassFibre[pf_num][yi][:,1],
#                                    z = brassFibre[pf_num][yi][:,2])
            
#             tubePts = nn_gridPts[pf_num][yi]
            
#             gd3.mlab_source.reset( x = bungeAngs[tubePts.astype(int),0],
#                                    y = bungeAngs[tubePts.astype(int),1],
#                                    z = bungeAngs[tubePts.astype(int),2])
        
#             yield
            
# anim()

#for yi in range(len(pf.y[pf_num])):
#    
#        gd = mlab.points3d(fibre[pf_num][yi][:,0],fibre[pf_num][yi][:,1],fibre[pf_num][yi][:,2],scale_factor=1,mode='point',color=(1,0,0))
#        gd.actor.property.render_points_as_spheres = True
#        gd.actor.property.point_size = 5    

mlab.show(stop=True)