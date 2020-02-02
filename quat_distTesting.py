#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 21:29:13 2020

@author: nate
"""

"""
quaternion distance speed testing
"""

import numpy as np
from numba import jit,prange
from sklearn.neighbors import KDTree

from pyTex.utils import symmetrise, normalize
from pyTex.orientation import eu2quat, quat2eu 
from pyTex.base import poleFigure, bunge

from tqdm import tqdm
import rowan as quat


@jit(nopython=True,parallel=True)
def quatMetricNumba(a, b):
    
    """ from DOI 10.1007/s10851-009-0161-2, #4 """
    
    dist = np.zeros((len(a),len(b)))
    
    for bi in prange(len(b)):
        
        dist[:,bi] = 1 - np.abs(np.dot(a,b[bi]))
    
    return dist

def quatMetric(a, b, rad):
    
    """ from DOI 10.1007/s10851-009-0161-2, #4 """
    
    dist = np.zeros((len(a),len(b)))
    
    for bi in range(len(b)):
        
        dist[:,bi] = 1 - np.abs(np.dot(a,b[bi]))

    tube = (dist <= rad)
    temp = np.column_stack((np.argwhere(tube)[:,0],dist[tube]))
    
    return temp 

def quatMetric_reshape(a, b):
    
    dist = 1 - np.abs(np.dot(a, b))
    
    return dist
    
def quatMetric_tensorDot(a, b):
    
    dist = 1 - np.abs(np.tensordot(a,b,axes=([-1],[1])))
    
    return dist

def calcFibre(symHKL,yset,qgrid,phi,rad,tree,euc_rad):
    
    cphi = np.cos(phi/2)
    sphi = np.sin(phi/2)
    
    q0 = {}
    q = {}
    qf = {}
    
    axis = {}
    omega = {}
    
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
        
        q0[fi] = {}
        q[fi] = {}
        
        axis[fi] = {}
        omega[fi] = {}
        
        """ set proper iterator """
        if isinstance(yset,dict): it = yset[fi]
        else: it = yset
        
        for yi,y in enumerate(it): 
            
            axis[fi][yi] = np.cross(fam,y)
            axis[fi][yi] = axis[fi][yi] / np.linalg.norm(axis[fi][yi],axis=1)[:,None]
            omega[fi][yi] = np.arccos(np.dot(fam,y))
            
            q0[fi][yi] = {}
            q[fi][yi] = {}
            qf[yi] = {}
            qfib = np.zeros((len(phi),len(fam),4))
            
            for hi,HxY in enumerate(axis[fi][yi]):
            
                q0[fi][yi][hi] = np.hstack( [ np.cos(omega[fi][yi][hi]/2), np.sin(omega[fi][yi][hi]/2) * HxY ] )
                q[fi][yi][hi]  = np.hstack( [ cphi[:, np.newaxis], np.tile( y, (len(cphi),1) ) * sphi[:, np.newaxis] ] )
                
                qf[yi][hi] = quat.multiply(q[fi][yi][hi], q0[fi][yi][hi])
            
                for qi in range(qf[yi][hi].shape[0]):
                    
                    qfib[qi,hi,:] = qf[yi][hi][qi,:]
              
            phi1, Phi, phi2 = quat2eu(qfib)
            
            phi1 = np.where(phi1 < 0, phi1 + 2*np.pi, phi1) #brnng back to 0 - 2pi
            Phi = np.where(Phi < 0, Phi + np.pi, Phi) #brnng back to 0 - pi
            phi2 = np.where(phi2 < 0, phi2 + 2*np.pi, phi2) #brnng back to 0 - 2pi
            
            eu_fib = np.stack( (phi1, Phi, phi2), axis=2 )
            eu_fib = np.reshape( eu_fib, (eu_fib.shape[0]*eu_fib.shape[1], eu_fib.shape[2]) ) #new method       
    
            fz = (eu_fib[:,0] <= od._phi1max) & (eu_fib[:,1] <= od._Phimax) & (eu_fib[:,2] <= od._phi2max)
            fz_idx = np.nonzero(fz)
            
            fibre_e[fi][yi] = eu_fib[fz]    
            fib_idx = np.unravel_index(fz_idx[0], (qfib.shape[0],qfib.shape[1]))            
            fibre_q[fi][yi] = qfib[fib_idx]
            
            """ euclidean distance calculation - KDTree """
            
            qfib_pos = np.copy(qfib[fib_idx])
            qfib_pos[qfib_pos[:,0] < 0] *= -1
            
            # returns tuple - first array are points, second array is distances
            query = tree.query_radius(qfib_pos,euc_rad,return_distance=True)
        
            # concatenate arrays
            query = np.column_stack([np.concatenate(ar) for ar in query])
            
            # sort by minimum distance - unique function takes first appearance of index
            query_sort = query[np.argsort(query[:,1],axis=0)]
            
            # return unique points
            uni_pts = np.unique(query_sort[:,0],return_index=True)
            
            nn_gridPts[fi][yi] = uni_pts[0].astype(int)
            nn_gridDist[fi][yi] = query_sort[uni_pts[1],1]
            
            """ geodesic distance calculation - dot product """
            
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
            
            # # egrid_trun[fi][yi] = bungeAngs[query_uni]
            
    return qfib[fib_idx], temp

# %% MAIN
    
#reflections
hkls = np.array([(2,2,2), (3,1,1), (4,0,0)])
hkls = normalize(hkls)
symHKL = symmetrise('m-3m', hkls)

#od
od = bunge(np.deg2rad(5), 'm-3m', '1')

#rotation around path in space
phi = np.linspace(0,2*np.pi,73)

#y
pf_grid, alp, bet = poleFigure.grid(res=np.deg2rad(5),
                                    radians=True,
                                    cen=True,
                                    ret_ab=True)

#calculate pole figure y's
sph = np.array((np.ravel(alp),np.ravel(bet))).T

#convert to xyz
xyz_pf = np.zeros((sph.shape[0],3))
xyz_pf[:,0] = np.sin( sph[:,0] ) * np.cos( sph[:,1] )
xyz_pf[:,1] = np.sin( sph[:,0] ) * np.sin( sph[:,1] )
xyz_pf[:,2] = np.cos( sph[:,0] ) 

""" quaternion grid """

bungeAngs = np.zeros(( np.product(od.phi1cen.shape), 3 ))

for ii,i in enumerate(np.ndindex(od.phi1cen.shape)):
    
    bungeAngs[ii,:] = np.array((od.phi1cen[i],od.Phicen[i],od.phi2cen[i]))

q_grid = eu2quat(bungeAngs).T

""" use sklearn KDTree for reduction of points for query (euclidean) """
from sklearn.neighbors import KDTree
qgrid_pos = np.copy(q_grid)
qgrid_pos[qgrid_pos[:,0] < 0] *= -1
tree = KDTree(qgrid_pos)

theta = np.deg2rad(7)

rad = np.sqrt( 2 * ( 1 - np.cos(0.5*theta) ) )
euc_rad = np.sqrt( 4 * np.sin(0.25*theta)**2 )

q_path, temp = calcFibre(symHKL,xyz_pf,q_grid,phi,rad,tree,euc_rad)

q_test = quat.random.rand(300)
q_testMD = q_test.reshape((1,4,300))

q_test_pos = np.copy(q_test)
q_test_pos[q_test_pos[:,0] < 0] *= -1



