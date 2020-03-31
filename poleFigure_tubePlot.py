#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:55:47 2020

@author: nate
"""

import os

from math import pi
import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import rowan as quat

from pyTex.utils import genSymOps, SPHtoXYZ
from pyTex.orientation import eu2quat, quat2eu 
from pyTex.base import poleFigure, bunge

#load in nomad pole figures

hkls = []
files = []
datadir = '/mnt/c/Users/Nate/pyReducePF/pole figures/pole figures integ int Al absCorr/combined'

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
    
pf = poleFigure(files,hkls,'m-3m','sparse')
rot = R.from_euler('XZY',(13,-88,90), degrees=True).as_matrix()
pf.rotate(rot)

od = bunge(np.deg2rad(5), 'm-3m', '1')
theta = np.deg2rad(5)

#path projection

""" quaternion grid """

bungeAngs = np.zeros(( np.product(od.phi1cen.shape), 3 ))

for ii,i in enumerate(np.ndindex(od.phi1cen.shape)):
    
    bungeAngs[ii,:] = np.array((od.phi1cen[i],od.Phicen[i],od.phi2cen[i]))

q_grid = eu2quat(bungeAngs).T

""" use sklearn KDTree for reduction of points for query (euclidean) """
qgrid_pos = np.copy(q_grid)
qgrid_pos[qgrid_pos[:,0] < 0] *= -1
tree = KDTree(qgrid_pos)

#rotation around path in space
phi = np.linspace(0,2*np.pi,73)

rad = np.sqrt( 2 * ( 1 - np.cos(0.5*theta) ) )
euc_rad = np.sqrt( 4 * np.sin(0.25*theta)**2 )

# q_path = calcFibre(hkls,xyz_pf,q_grid,phi,rad,tree,euc_rad)

symOps = genSymOps('m-3m')
symOps = np.unique(np.swapaxes(symOps,2,0),axis=0)

""" sym ops """

proper = np.where( np.linalg.det(symOps) == 1 ) #proper orthogonal
quatSymOps = quat.from_matrix(symOps[proper])
quatSymOps = np.tile(quatSymOps[:,:,np.newaxis],(1,1,len(phi)))
quatSymOps = quatSymOps.transpose((0,2,1))

""" create small grid centered around [001] """

polar_stepN = 9
polar_step = np.deg2rad(5) / (polar_stepN-1)

polar = np.arange(0,polar_stepN) * polar_step
r = np.sin(polar)
azi_stepN = np.ceil(2.0*pi*r / polar_step)
azi_stepN[0] = 1 #single point at poles
azi_step = 2*np.pi / azi_stepN

pts = []

for azi_n,pol in zip(azi_stepN.astype(int),polar):
    
    azi = np.linspace(0,2*pi,azi_n)
    pol = np.ones((len(azi)))*pol

    x = np.sin(pol) * np.cos(azi)
    y = np.sin(pol) * np.sin(azi)
    z = np.cos(pol)

    pts.append(np.array((x,y,z)).T)

xyz_sphere = np.vstack(pts)

# %%

cphi = np.cos(phi/2)
sphi = np.sin(phi/2)

q0 = {}
q = {}
qf = {}

fibre_e = {}
fibre_q = {}
tube_e = {}

nn_gridPts = {}
nn_gridDist = {}
egrid_trun = {}
    
for fi,fam in enumerate(pf._normHKLs[:1]):
    
    fibre_e[fi] = {}
    fibre_q[fi] = {}
    tube_e[fi] = {}
    
    q0[fi] = {}
    q[fi] = {}
    qf[fi] = {}
    
    nn_gridPts[fi] = {}
    nn_gridDist[fi] = {}
    egrid_trun[fi] = {}
    
    mag = {}
    grid = {}
    
    for yi,y in enumerate(tqdm(pf.y[fi],position=0,desc='pole figure y',leave=True)): 
        
        axis = np.cross(fam,y)
        axis = axis / np.linalg.norm(axis,axis=-1)
        omega = np.arccos(np.dot(fam,y))
        
        q0[fi][yi] = np.hstack( [ np.cos(omega/2), np.sin(omega/2) * axis ] )
        q[fi][yi]  = np.hstack( [ cphi[:, np.newaxis], np.tile( y, (len(cphi),1) ) * sphi[:, np.newaxis] ] )
        qf[fi][yi] = quat.multiply(q[fi][yi], q0[fi][yi])
        
        qfib = quat.multiply(qf[fi][yi], quatSymOps)
        qfib = qfib.transpose((1,0,2))        
        phi1, Phi, phi2 = quat2eu(qfib)
        
        phi1 = np.where(phi1 < 0, phi1 + 2*np.pi, phi1) #brnng back to 0 - 2pi
        Phi = np.where(Phi < 0, Phi + np.pi, Phi) #brnng back to 0 - pi
        phi2 = np.where(phi2 < 0, phi2 + 2*np.pi, phi2) #brnng back to 0 - 2pi
        eu_fib = np.stack( (phi1, Phi, phi2), axis=2 )
            
        #unique values based on rectilinear bunge box
        eu_fib = np.reshape( eu_fib, (eu_fib.shape[0]*eu_fib.shape[1], eu_fib.shape[2]) ) #new method       
        fz = (eu_fib[:,0] <= od._phi1max) & (eu_fib[:,1] <= od._Phimax) & (eu_fib[:,2] <= od._phi2max)
        fz_idx = np.nonzero(fz)        
        fibre_e[fi][yi],uni_path_idx = np.unique(eu_fib[fz],return_index=True,axis=0)
        
        fib_idx = np.unravel_index(fz_idx[0], (qfib.shape[0],qfib.shape[1]))            
        fibre_q[fi][yi] = qfib[fib_idx]
        
        """ tube comparison """
        
        # q_cen = R.from_quat(qfib[fib_idx][:,[1,2,3,0]])
        
        # outsideTube = False
        
        # ## iterate on this single y until outside of radius
        # while outsideTube is False:
            
        #     #calculate rotation required to pull small grid into y
        #     align = R.align_vectors(y[None,:],xyz_sphere[0][None,:])
        #     #rotate grid into alignment
        #     test_grid = align[0].apply(xyz_sphere)
            
        #     q_test = {}
        #     q_comp = {}
        #     q_mag = []
            
        #     for yti, y_test in enumerate(test_grid):
                
        #         axis = np.cross(fam,y_test)
        #         axis = axis / np.linalg.norm(axis,axis=-1)
        #         omega = np.arccos(np.dot(fam,y_test))
                
        #         q0t = np.hstack( [ np.cos(omega/2), np.sin(omega/2) * axis ] )
        #         qt  = np.hstack( [ cphi[:, np.newaxis], np.tile( y_test, (len(cphi),1) ) * sphi[:, np.newaxis] ] )
        #         qft = quat.multiply(qt, q0t)
                
        #         qfib_t = quat.multiply(qft, quatSymOps)
        #         qfib_t = qfib_t.transpose((1,0,2))        
        #         phi1, Phi, phi2 = quat2eu(qfib_t)
                
        #         phi1 = np.where(phi1 < 0, phi1 + 2*np.pi, phi1) #brnng back to 0 - 2pi
        #         Phi = np.where(Phi < 0, Phi + np.pi, Phi) #brnng back to 0 - pi
        #         phi2 = np.where(phi2 < 0, phi2 + 2*np.pi, phi2) #brnng back to 0 - 2pi
        #         eu_fib = np.stack( (phi1, Phi, phi2), axis=2 )
            
        #         #unique values based on rectilinear bunge box
        #         eu_fib = np.reshape( eu_fib, (eu_fib.shape[0]*eu_fib.shape[1], eu_fib.shape[2]) ) #new method       
        #         fz = (eu_fib[:,0] <= od._phi1max) & (eu_fib[:,1] <= od._Phimax) & (eu_fib[:,2] <= od._phi2max)
        #         fz_idx = np.nonzero(fz)        
                
        #         fib_idx = np.unravel_index(fz_idx[0], (qfib_t.shape[0],qfib_t.shape[1]))            
        #         q_test[yti] = R.from_quat(qfib_t[fib_idx][:,[1,2,3,0]])
                
        #         q_comp[yti] = q_cen * q_test[yti].inv() 
        #         q_mag.append(q_comp[yti].magnitude())
            
        #     mag[yi] = np.mean(np.vstack(q_mag),axis=1)
        #     grid[yi] = np.copy(test_grid)
        #     outsideTube = True
            
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


# %%
        
#calculate wimv pointer
pf.res = np.deg2rad(1.5)
od._calcPointer('wimv', pf)  

# %%      

pf_num = 0
yi = 50

#plotting
import mayavi.mlab as mlab

#full grid
fullPFgrid, xyz_pf = pf.genGrid(res=pf.res,
                                radians=True,
                                centered=True,
                                ret_xyz=True)

gd = mlab.points3d(xyz_pf[:,0],xyz_pf[:,1],xyz_pf[:,2],mode='point',scale_factor=1,color=(0,0,0))
gd.actor.property.render_points_as_spheres = True
gd.actor.property.point_size = 5

cells = mlab.points3d(0,0,0,0,mode='point', colormap='viridis',scale_factor=0.25)
cells.actor.property.render_points_as_spheres = True
cells.actor.property.point_size = 6

vec = mlab.points3d(0,0,0,mode='point',scale_factor=1,color=(0,1,0))
vec.actor.property.render_points_as_spheres = True
vec.actor.property.point_size = 5

#pointer
pter = od.pointer['full']['od to pf'][pf_num]

@mlab.animate(delay=800)
def anim():
    while True:
        
        for yi,y_plt in enumerate(pf.y[pf_num]):
            
            #invert y
            if y_plt[2] < 0: y_plt *= -1
            
            #tube pts
            tube_pts = nn_gridPts[pf_num][yi]
            
            pf_cells   = []
            pf_cellWgt = [] 
            
            #accumulate pole fig points
            for i,pt in enumerate(tube_pts):
                
                #od cell is defined in pointer
                if pt in pter:
                    
                    pf_cells.append(pter[pt])
                    pf_cellWgt.append(nn_gridDist[pf_num][yi][i])
            
            #return unique index
            pf_cells, uni_idx = np.unique(np.vstack(pf_cells),return_index=True)
            
            #pull cell wgts
            col,row = np.divmod(uni_idx,len(pf_cellWgt))
            
            weights = np.vstack(pf_cellWgt)[col].flatten()
            weights /= np.max(weights)
                
            cells.mlab_source.reset(x = xyz_pf[pf_cells.astype(int),0],
                                    y = xyz_pf[pf_cells.astype(int),1],
                                    z = xyz_pf[pf_cells.astype(int),2],
                                    scalars = weights)
            
            vec.mlab_source.reset(x = y_plt[0],
                                  y = y_plt[1],
                                  z = y_plt[2])
        
            yield
            
anim()
mlab.show(stop=True)


# %% 
        
        
""" tube plot """
            
# import mayavi.mlab as mlab

# #full grid

# fullPFgrid, xyz_pf = pf.genGrid(res=np.deg2rad(5),
#                                 radians=True,
#                                 centered=True,
#                                 ret_xyz=True)


            
# gd = mlab.points3d(xyz_pf[:,0],xyz_pf[:,1],xyz_pf[:,2],mode='point',scale_factor=1,color=(0,0,0))
# gd.actor.property.render_points_as_spheres = True
# gd.actor.property.point_size = 5

# for yi,y_plt in grid.items():

#     #subset pts
#     y_plt[y_plt[:,2] < 0] *= -1
    
#     insideTube = (mag[yi] < theta)
#     #scale pts
#     s = mag[yi]/theta
    
#     gd = mlab.points3d(y_plt[insideTube,0],y_plt[insideTube,1],y_plt[insideTube,2],s[insideTube],mode='point', colormap="viridis", scale_factor=1)
#     gd.actor.property.render_points_as_spheres = True
#     gd.actor.property.point_size = 6
    
#     #vector
    
#     gd = mlab.points3d(y_plt[0,0],y_plt[0,1],y_plt[0,2],mode='point',scale_factor=1,color=(0,1,0))
#     gd.actor.property.render_points_as_spheres = True
#     gd.actor.property.point_size = 10
