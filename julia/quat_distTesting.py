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

from pyTex.utils import symmetrise, normalize, genSymOps
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
pf_grid, alp, bet = poleFigure.genGrid(res=np.deg2rad(5),
                                    radians=True,
                                    centered=True,
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
qgrid_pos = np.copy(q_grid)
qgrid_pos[qgrid_pos[:,0] < 0] *= -1
tree = KDTree(qgrid_pos)

theta = np.deg2rad(7)

rad = np.sqrt( 2 * ( 1 - np.cos(0.5*theta) ) )
euc_rad = np.sqrt( 4 * np.sin(0.25*theta)**2 )

# q_path = calcFibre(hkls,xyz_pf,q_grid,phi,rad,tree,euc_rad)

symOps = genSymOps('m-3m')
symOps = np.unique(np.swapaxes(symOps,2,0),axis=0)

""" only use proper rotations """
""" complicated, simplify? """

proper = np.where( np.linalg.det(symOps) == 1 ) #proper orthogonal
quatSymOps = quat.from_matrix(symOps[proper])
quatSymOps = np.tile(quatSymOps[:,:,np.newaxis],(1,1,len(phi)))
quatSymOps = quatSymOps.transpose((0,2,1))

# qf_sym = quat.multiply(qf[0],quatSymOps)
# qf_fz = np.argmin(qf_sym,axis=0)s

# q_test = quat.random.rand(300)
# q_testMD = q_test.reshape((1,4,300))

# q_test_pos = np.copy(q_test)
# q_test_pos[q_test_pos[:,0] < 0] *= -1

# %%

cphi = np.cos(phi/2)
sphi = np.sin(phi/2)

q0 = {}
q = {}
qf = {}
q_tubeTest = {}

axis = {}
omega = {}

fibre_e = {}
fibre_q = {}
tube_e = {}

nn_gridPts = {}
nn_gridDist = {}

egrid_trun = {}
    
for fi,fam in enumerate(tqdm(hkls)):
    
    fibre_e[fi] = {}
    fibre_q[fi] = {}
    tube_e[fi] = {}
    
    nn_gridPts[fi] = {}
    nn_gridDist[fi] = {}
    
    egrid_trun[fi] = {}
    
    q0[fi] = {}
    q[fi] = {}
    
    qf[fi] = {}
    q_tubeTest[fi] = {}
    
    axis[fi] = {}
    omega[fi] = {}
    
    """ set proper iterator """
    if isinstance(xyz_pf,dict): it = xyz_pf[fi]
    else: it = xyz_pf
    
    for yi,y in enumerate(it): 
        
        axis[fi][yi] = np.cross(fam,y)
        axis[fi][yi] = axis[fi][yi] / np.linalg.norm(axis[fi][yi],axis=-1)
        omega[fi][yi] = np.arccos(np.dot(fam,y))
        
        q0[fi][yi] = np.hstack( [ np.cos(omega[fi][yi]/2), np.sin(omega[fi][yi]/2) * axis[fi][yi] ] )
        q[fi][yi]  = np.hstack( [ cphi[:, np.newaxis], np.tile( y, (len(cphi),1) ) * sphi[:, np.newaxis] ] )
        qf[fi][yi] = quat.multiply(q[fi][yi], q0[fi][yi])
        
        q_tubeTest[fi][yi] = quat.multiply(qf[fi][yi],quat.from_axis_angle((1,1,0), theta))
        # qfib = quat.multiply(quatSymOps, qf[yi])
        qfib = quat.multiply(qf[fi][yi], quatSymOps)
        qfib = qfib.transpose((1,0,2))
        
        phi1, Phi, phi2 = quat2eu(qfib)
        
        qtube = quat.multiply(q_tubeTest[fi][yi], quatSymOps)
        qtube = qtube.transpose((1,0,2))
        
        phi1_t, Phi_t, phi2_t = quat2eu(qtube)
        
        # q0[fi][yi] = {}
        # q[fi][yi] = {}
        # qf[yi] = {}
        
        # qfib = np.zeros((len(phi),len(fam),4))
        
        # for hi,HxY in enumerate(axis[fi][yi]):
        
        #     q0[fi][yi][hi] = np.hstack( [ np.cos(omega[fi][yi][hi]/2), np.sin(omega[fi][yi][hi]/2) * HxY ] )
        #     q[fi][yi][hi]  = np.hstack( [ cphi[:, np.newaxis], np.tile( y, (len(cphi),1) ) * sphi[:, np.newaxis] ] )
            
        #     qf[yi][hi] = quat.multiply(q[fi][yi][hi], q0[fi][yi][hi])
        
        #     for qi in range(qf[yi][hi].shape[0]):
                
        #         qfib[qi,hi,:] = qf[yi][hi][qi,:]
          
        # phi1, Phi, phi2 = quat2eu(qfib)
        
        phi1 = np.where(phi1 < 0, phi1 + 2*np.pi, phi1) #brnng back to 0 - 2pi
        Phi = np.where(Phi < 0, Phi + np.pi, Phi) #brnng back to 0 - pi
        phi2 = np.where(phi2 < 0, phi2 + 2*np.pi, phi2) #brnng back to 0 - 2pi
        eu_fib = np.stack( (phi1, Phi, phi2), axis=2 )
        
        phi1_t = np.where(phi1_t < 0, phi1_t + 2*np.pi, phi1_t) #brnng back to 0 - 2pi
        Phi_t = np.where(Phi_t < 0, Phi_t + np.pi, Phi_t) #brnng back to 0 - pi
        phi2_t = np.where(phi2_t < 0, phi2_t + 2*np.pi, phi2_t) #brnng back to 0 - 2pi
        eu_tube = np.stack ( (phi1_t, Phi_t, phi2_t), axis=2 )

        # #largest scalar component quaternion
        # fz_quat = np.argmax(qfib,axis=1)
        # fibre_e[fi][yi] = np.zeros((qfib.shape[0],3))
        # fibre_q[fi][yi] = np.zeros((qfib.shape[0],4))
        
        # for i,fzi in enumerate(fz_quat[:,0]):
            
        #     fibre_e[fi][yi][i,:] = eu_fib[i,fzi,:]
        #     fibre_q[fi][yi][i,:] = qfib[i,fzi,:]
            
        #unique values based on rectilinear bunge box
        eu_fib = np.reshape( eu_fib, (eu_fib.shape[0]*eu_fib.shape[1], eu_fib.shape[2]) ) #new method       
        fz = (eu_fib[:,0] <= od._phi1max) & (eu_fib[:,1] <= od._Phimax) & (eu_fib[:,2] <= od._phi2max)
        fz_idx = np.nonzero(fz)        
        fibre_e[fi][yi],uni_path_idx = np.unique(eu_fib[fz],return_index=True,axis=0)
        
        eu_tube = np.reshape( eu_tube, (eu_tube.shape[0]*eu_tube.shape[1], eu_tube.shape[2]) ) #new method  
        fz_tube = (eu_tube[:,0] <= od._phi1max) & (eu_tube[:,1] <= od._Phimax) & (eu_tube[:,2] <= od._phi2max)
        fz_tube_idx = np.nonzero(fz_tube) 
        tube_e[fi][yi] = np.unique(eu_tube[fz],axis=0)
        
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
        # qgrid_trun = q_grid[query_uni]
        # qgrid_trun_idx = np.arange(len(q_grid))[query_uni] #store indexes to retrieve original grid pts later
        
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
            
# %%

### FIBER PLOT ###

# """ import matlab """

# from scipy.io import loadmat

# mtex_fib = loadmat('/home/nate/Dropbox/ORNL/Texture/NRSF2/bunList.mat')['bun_list']

pf_num = 0
y_num = 155

import mayavi.mlab as mlab
# import matplotlib.pyplot as plt
# fig = plt.figure()

mlab.figure(bgcolor=(1,1,1))

## grid ##
# gd = mlab.points3d(bungeAngs[:,0],bungeAngs[:,1],bungeAngs[:,2],scale_factor=1,mode='point',color=(0,0,0))
# gd.actor.property.render_points_as_spheres = True
# gd.actor.property.point_size = 3
  
# ## lit point ##
# gd2 = mlab.points3d(0,0,0,scale_factor=1,mode='point',color=(1,0,0))
# gd2.actor.property.render_points_as_spheres = True
# gd2.actor.property.point_size = 5

# test = np.sort(fibre_e[pf_num][955],axis=1)
# test = np.around(fibre_e[pf_num][455],decimals=8)
# test = fibre_e[pf_num][455][fibre_e[pf_num][455][:,0].argsort()]

test = np.copy(fibre_e[pf_num][y_num ])
test_tube = np.copy(tube_e[pf_num][y_num ])
tube = np.copy(bungeAngs[nn_gridPts[pf_num][y_num].astype(int)])

sz = np.arange(0,len(test))

gd4 = mlab.points3d(tube[:,0],
                    tube[:,1],
                    tube[:,2],
                    scale_factor=1,
                    mode='point',
                    color=(1,0,0))

gd4.actor.property.render_points_as_spheres = True
gd4.actor.property.point_size = 3

gd2 = mlab.points3d(test[:,0],
                    test[:,1],
                    test[:,2],
                    scale_factor=1,
                    mode='point',
                    color=(0,0,1))

gd2.actor.property.render_points_as_spheres = True
gd2.actor.property.point_size = 8

gd3 = mlab.points3d(test_tube[:,0],
                    test_tube[:,1],
                    test_tube[:,2],
                    scale_factor=1,
                    mode='point',
                    color=(0,1,0))

gd3.actor.property.render_points_as_spheres = True
gd3.actor.property.point_size = 8




# ## mtex fibre ##
# gd3 = mlab.points3d(mtex_fib[:,0],
#                     mtex_fib[:,1],
#                     mtex_fib[:,2],
#                     scale_factor=1,
#                     mode='point',
#                     color=(0,0,1))

# gd3.actor.property.render_points_as_spheres = True
# gd3.actor.property.point_size = 5

# ## trun grid ##
# gd3 = mlab.points3d(0,0,0,scale_factor=1,mode='point',color=(0,0,1))
# gd3.actor.property.render_points_as_spheres = True
# gd3.actor.property.point_size = 5   

# plt_list = list(fibre_full_e[pf_num].keys())
# plt_list.sort()

# """ cube (111) pts """

# from scipy.spatial.distance import cdist

# azi = np.deg2rad(np.array((45,135,225,315)))
# pol = np.deg2rad(np.array((35,35,35,35)))
# x = np.sin(pol) * np.cos(azi)
# y = np.sin(pol) * np.sin(azi)
# z = np.cos(pol)
# pts = np.array((x,y,z)).T

# dist_mat = cdist(xyz_pf,pts)
# plt_list = np.argmin(dist_mat,axis=0)

# @mlab.animate(delay=100)
# def anim():
#     while True:
        
#         for yi in plt_list:
                
# #            gd2.mlab_source.reset( x = fibre_wimv[pf_num][yi][:,0],
# #                                   y = fibre_wimv[pf_num][yi][:,1],
# #                                   z = fibre_wimv[pf_num][yi][:,2])
            
#             gd2.mlab_source.reset( x = fibre_full_e[pf_num][yi][:,0],
#                                   y = fibre_full_e[pf_num][yi][:,1],
#                                   z = fibre_full_e[pf_num][yi][:,2])
            
#             # gd2.mlab_source.reset( x = egrid_trun[pf_num][yi][:,0],
#             #                         y = egrid_trun[pf_num][yi][:,1],
#             #                         z = egrid_trun[pf_num][yi][:,2])
            
#             tubePts = nn_gridPts_full[pf_num][yi]
            
#             gd3.mlab_source.reset( x = bungeAngs[tubePts.astype(int),0],
#                                     y = bungeAngs[tubePts.astype(int),1],
#                                     z = bungeAngs[tubePts.astype(int),2])
        
#             yield
            
# anim()

#for yi in range(len(pf.y[pf_num])):
#    
#        gd = mlab.points3d(fibre[pf_num][yi][:,0],fibre[pf_num][yi][:,1],fibre[pf_num][yi][:,2],scale_factor=1,mode='point',color=(1,0,0))
#        gd.actor.property.render_points_as_spheres = True
#        gd.actor.property.point_size = 5    

mlab.show(stop=True)

# %%

# from scipy.spatial.transform import Rotation as R

# sciR_quatSymOps = R.from_quat(quatSymOps[:,0,:])
    
# qf_sym = sciR_quatSymOps.
