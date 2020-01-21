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
from math import pi

# sys.path.append('/mnt/2E7481AA7481757D/Users/Nate/pyTex/')
sys.path.append('/home/nate/projects/pyTex/')

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

pi2 = pi/2

polar_stepN = 15
polar_step = pi2 / (polar_stepN-1)

polar = np.arange(0,polar_stepN) * polar_step
r = np.sin(polar)
azi_stepN = np.ceil(2.0*pi*r / polar_step)
azi_stepN[0] = 0.9995 #single point at poles
azi_step = 2*pi / azi_stepN

pts = []

for azi_n,pol in zip(azi_stepN,polar):
    
    azi = np.linspace(0,2*pi,azi_n)
    pol = np.ones((len(azi)))*pol

    x = np.sin(pol) * np.cos(azi)
    y = np.sin(pol) * np.sin(azi)
    z = np.cos(pol)

    pts.append(np.array((x,y,z)).T)

xyz_pf = np.vstack(pts) 

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
        qfib_all = {}
        
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
            
            # fibre_e[fi][yi] = eu_fib[fz]
            fibre_e[fi][yi] = eu_fib
    
            fib_idx = np.unravel_index(fz_idx[0], (qfib.shape[0],qfib.shape[1]))
            
            fibre_q[fi][yi] = qfib[fib_idx]
            qfib_all[yi] = qfib
            
            # """ reduce geodesic query size """
            # qfib_pos = np.copy(qfib[fib_idx])
            # qfib_pos[qfib_pos[:,0] < 0] *= -1
            
            # query = np.concatenate(tree.query_radius(qfib_pos,euc_rad))
            # query_uni = np.unique(query)
            # qgrid_trun = qgrid[query_uni]
            # qgrid_trun_idx = np.arange(len(qgrid))[query_uni] #store indexes to retrieve original grid pts later
            
            # """ distance calc """
            # temp = quatMetricNumba(qgrid_trun,qfib[fib_idx])
            # """ find tube """
            # tube = (temp <= rad)
            # temp = np.column_stack((np.argwhere(tube)[:,0],temp[tube]))
            
            # """ round very small values """
            # temp = np.round(temp, decimals=7)
            
            # """ move values at zero to very small (1E-5) """
            # temp[:,1] = np.where(temp[:,1] == 0, 1E-5, temp[:,1])
            
            # """ sort by min distance """
            # temp = temp[np.argsort(temp[:,1],axis=0)]
            # """ return unique pts (first in list) """
            # uni_pts = np.unique(temp[:,0],return_index=True)
            
            # nn_gridPts[fi][yi] = qgrid_trun_idx[uni_pts[0].astype(int)]
            # nn_gridDist[fi][yi] = temp[uni_pts[1],1]
            
            # egrid_trun[fi][yi] = bungeAngs[query_uni]
            
    # return nn_gridPts, nn_gridDist
    # return qfib_all, nn_gridPts
    return qfib_all, fibre_e, nn_gridPts

brassFibre, brassFibre_e, nn_gridPts  = calcFibre(symHKL,xyz_pf,qgrid,omega,rad,tree,euc_rad)

# %%

""" Marc DeGraef method J. Appl. Cryst. (2019) 52 """

# brass_y2 = np.zeros((1,3,8))
HxY = {}
ome = {}

cphi = np.cos(omega/2)
sphi = np.sin(omega/2)

q0 = {}
q = {}
qf = {}

fibre_marc = {}
fibre_marc_e = {}

for yi,by in enumerate(xyz_pf): 
    
    # brass_y2[:,:,yi] = by
    HxY[yi] = np.cross(symHKL[0],by)
    ome[yi] = np.arccos(np.dot(symHKL[0],by))
    
    q0[yi] = {}
    q[yi] = {}
    qf[yi] = {}
    fibre_marc[yi] = np.zeros_like(brassFibre[yi])
    
    for hi,h in enumerate(HxY[yi]):
        
        q0[yi][hi] = quat.normalize(np.hstack( [ np.cos(ome[yi][hi]/2), np.sin(ome[yi][hi]/2) * h ] ))
        q[yi][hi]  = quat.normalize(np.hstack( [ cphi[:, np.newaxis], np.tile( by, (len(cphi),1) ) * sphi[:, np.newaxis] ] ))
        
        qf[yi][hi] = quat.multiply(q[yi][hi], q0[yi][hi])
    
        for qi in range(qf[yi][hi].shape[0]):
            
            fibre_marc[yi][qi,hi,:] = qf[yi][hi][qi,:]
            
    phi1, Phi, phi2 = quat2eu(fibre_marc[yi])
    
    phi1 = np.where(phi1 < 0, phi1 + 2*np.pi, phi1) #brnng back to 0 - 2pi
    Phi = np.where(Phi < 0, Phi + np.pi, Phi) #brnng back to 0 - pi
    phi2 = np.where(phi2 < 0, phi2 + 2*np.pi, phi2) #brnng back to 0 - 2pi
    
    eu_fib = np.stack( (phi1, Phi, phi2), axis=2 )
    fibre_marc_e[yi] = np.reshape( eu_fib, (eu_fib.shape[0]*eu_fib.shape[1], eu_fib.shape[2]) ) #new method    
    
    eu_fib = np.reshape( eu_fib, (eu_fib.shape[0]*eu_fib.shape[1], eu_fib.shape[2]) ) #new method       
    
    fz = (eu_fib[:,0] <= od._phi1max) & (eu_fib[:,1] <= od._Phimax) & (eu_fib[:,2] <= od._phi2max)
    fz_idx = np.nonzero(fz)
    
    fib_idx = np.unravel_index(fz_idx[0], (fibre_marc[yi].shape[0],fibre_marc[yi].shape[1]))
    
    """ distance calc """
    temp = quatMetricNumba(qgrid,fibre_marc[yi][fib_idx])
    
# %%
        
""" compare """

test = {}

for yi in qf.keys():
    
    test[yi] = {}
    
    for hi in qf[yi].keys():
        
        test[yi][hi] = (qf[yi][hi],brassFibre[yi][:,hi,:])

# %%

### FIBER PLOT ###

import mayavi.mlab as mlab

mlab.figure(bgcolor=(1,1,1))

pf_num = 0

## grid ##
gd = mlab.points3d(bungeAngs[:,0],bungeAngs[:,1],bungeAngs[:,2],scale_factor=1,mode='point',color=(0,0,0))
gd.actor.property.render_points_as_spheres = True
gd.actor.property.point_size = 3

# ## all fibres ##
# allBrassFibre = []
# for k,v in brassFibre[pf_num].items():
#     tubePts = nn_gridPts[pf_num][k]
#     allBrassFibre.append(v)
#     # allBrassFibre.append(bungeAngs[tubePts.astype(int),:])
    
# allBrassFibre = np.vstack(allBrassFibre)
# gd = mlab.points3d(allBrassFibre[:,0],allBrassFibre[:,1],allBrassFibre[:,2],scale_factor=1,mode='point',color=(1,0,0))
# gd.actor.property.render_points_as_spheres = True
# gd.actor.property.point_size = 10

## lit point ##
gd2 = mlab.points3d(fibre_marc_e[pf_num][:,0],fibre_marc_e[pf_num][:,1],fibre_marc_e[pf_num][:,2],scale_factor=1,mode='point',color=(1,0,0))
gd2.actor.property.render_points_as_spheres = True
gd2.actor.property.point_size = 7

## manual fibre ##
gd3 = mlab.points3d(brassFibre_e[0][pf_num][:,0],brassFibre_e[0][pf_num][:,1],brassFibre_e[0][pf_num][:,2],scale_factor=1,mode='point',color=(0,1,0))
gd3.actor.property.render_points_as_spheres = True
gd3.actor.property.point_size = 7

# plt_list = list(brassFibre[pf_num].keys())
# plt_list.sort()

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