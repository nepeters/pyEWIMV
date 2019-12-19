#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 18:01:30 2019

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

import os

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import rowan as quat

from pymatgen.core import Lattice, Structure

from pyTex import poleFigure, bunge
from pyTex.orientation import eu2quat, quat2eu 
from pyTex.utils import symmetrise, normalize, genSym
from pyTex.diffrac import calc_XRDreflWeights

dir_path = os.path.dirname(os.path.realpath('__file__'))

P = 1

crystalSym = 'm-3m'
sampleSym = '1'
cellSize = np.deg2rad(5)
theta = np.deg2rad(5)

hkls = []
files = []

datadir = os.path.join(dir_path,'Data','NOMAD Aluminum - no abs','pole figures','combined')
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
    
""" refl weights """

a = 4.046 #Angstrom
latt = Lattice.cubic(a)
structure = Structure(latt, ["Al", "Al", "Al", "Al"], [[0, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]])
rad = 'CuKa'

refl_wgt = calc_XRDreflWeights(structure, hkls, rad='MoKa')

""" ones for refl_wgt """

refl_wgt = {}

for hi,h in enumerate(hkls):
    refl_wgt[hi] = 1

hkl_str = [''.join(tuple(map(str,h))) for h in hkls]

""" rotate """

rot = R.from_euler('XZY',(13,-90,90), degrees=True).as_dcm()

pf = poleFigure(files,hkls,crystalSym,'nd')
pf.rotate(rot)

od = bunge(cellSize, crystalSym, sampleSym)
hkls = np.array(hkls)

omega = np.radians(np.arange(0,360+5,5))
    
""" symmetry after """

fibre_e = {}
fibre_q = {}
weights = {}
hkls = normalize(hkls)

""" search for unique hkls to save time """

hkls_loop, uni_hkls_idx, hkls_loop_idx = np.unique(hkls,axis=0,return_inverse=True,return_index=True)
symHKL_loop = symmetrise(crystalSym, hkls_loop)
symHKL_loop = normalize(symHKL_loop)

symOps = genSym(crystalSym)
symOps = np.unique(np.swapaxes(symOps,2,0),axis=0)

""" only use proper rotations """
""" complicated, simplify? """

proper = np.where( np.linalg.det(symOps) == 1 ) #proper orthogonal
quatSymOps = quat.from_matrix(symOps[proper])
quatSymOps = np.tile(quatSymOps[:,:,np.newaxis],(1,1,len(omega)))
quatSymOps = quatSymOps.transpose((2,0,1))

""" gen quats from bunge grid """

bungeAngs = np.zeros(( np.product(od.phi1cen.shape), 3 ))

for ii,i in enumerate(np.ndindex(od.phi1cen.shape)):
    
    bungeAngs[ii,:] = np.array((od.phi1cen[i],od.Phicen[i],od.phi2cen[i]))

qgrid = eu2quat(bungeAngs).T

""" calculate pf grid XYZ for fibre """

pf_grid, alp, bet = pf.grid(full=True, ret_ab=True)

#calculate pole figure y's
sph = np.array(np.divmod(np.ravel(pf_grid),pf_grid.shape[1])).T
sph = sph * pf.res
sph[:,0] = np.where(sph[:,0] == 0, 0.004363323, sph[:,0]) #1/4deg tilt off center to avoid issues

#convert to xyz
xyz_pf = np.zeros((sph.shape[0],3))
xyz_pf[:,0] = np.sin( sph[:,0] ) * np.cos( sph[:,1] )
xyz_pf[:,1] = np.sin( sph[:,0] ) * np.sin( sph[:,1] )
xyz_pf[:,2] = np.cos( sph[:,0] )

fibre_full_e = {}
fibre_full_q = {}

nn_gridPts_full = {}
nn_gridDist_full = {}

# %%

"""
loop can be improved:
    eliminate replicate paths
    multiprocess?
"""

""" start pointer loop """

from numba import jit,prange
@jit(nopython=True,parallel=True)
def quatMetricNumba(a, b):
    
    """ from DOI 10.1007/s10851-009-0161-2, #4 """
    
    dist = np.zeros((len(a),len(b)))
    
    for bi in prange(len(b)):
        
        dist[:,bi] = 1 - np.abs(np.dot(a,b[bi]))
    
    return dist

""" use sklearn KDTree for reduction of points for query (euclidean) """
from sklearn.neighbors import KDTree
qgrid_pos = np.copy(qgrid)
qgrid_pos[qgrid_pos[:,0] < 0] *= -1
tree = KDTree(qgrid_pos)

rad = ( 1 - np.cos(theta) ) / 2
euc_rad = 4*np.sin(1.25*theta)**2
    
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
            
    return nn_gridPts, nn_gridDist

nn_gridPts, nn_gridDist = calcFibre(pf.symHKL,pf.y,qgrid,omega,rad,tree,euc_rad)
tempPts_full, tempDist_full = calcFibre(symHKL_loop,xyz_pf,qgrid,omega,rad,tree,euc_rad)
 
for i,hi in enumerate(hkls_loop_idx):

    nn_gridPts_full[i] = tempPts_full[hi]
    nn_gridDist_full[i] = tempDist_full[hi]

    
# %%        

""" extract distances -> weights, construct pointer 
    pf_od: pf --> pf to od
    od_pf: od --> od to pf
    each entry stored as dict; ['cell'] is cell #s ['weights'] is weights """
    
tube_exp = 0.5

pf_od = {}
pf_od_full = {}
odwgts_tot = np.zeros( ( len(hkls), od.bungeList.shape[0]*od.bungeList.shape[1]*od.bungeList.shape[2] ) )

test = []

for hi, h in enumerate(hkls):
    
    pf_od[hi] = {}
    pf_od_full[hi] = {}
    
    for yi in range(len(nn_gridPts[hi].keys())):
        
        od_cells = nn_gridPts[hi][yi]
        #TODO: work on scaling towards max distance
        scaled_dist = nn_gridDist[hi][yi]
        weights = 1 / ( ( scaled_dist )**tube_exp )
        
        if np.any(weights < 0): raise ValueError('neg weight')
        if np.any(weights == 0): raise ValueError('zero weight')
        
        pf_od[hi][yi] = {'cell': od_cells, 'weight': weights}
        
        odwgts_tot[hi,od_cells.astype(int)] += weights
        
    for yi in range(len(nn_gridPts_full[hi].keys())):
        
        od_cells = nn_gridPts_full[hi][yi]
        #TODO: work on scaling towards max distance
        scaled_dist = nn_gridDist_full[hi][yi]
        weights = 1 / ( ( scaled_dist )**tube_exp )
        
        if np.any(weights < 0): raise ValueError('neg weight')
        if np.any(weights == 0): raise ValueError('zero weight')
        
        pf_od_full[hi][yi] = {'cell': od_cells, 'weight': weights}
        
odwgts_tot = np.where(odwgts_tot == 0, 1, odwgts_tot)
odwgts_tot = 1 / odwgts_tot

# %%
    
""" e-wimv iteration start """
        
od_data = np.ones( od.bungeList.shape[0]*od.bungeList.shape[1]*od.bungeList.shape[2] )
calc_od = {}
recalc_pf = {}

recalc_pf_full = {}
    
numPoles = pf._numHKL
numHKLs = [len(fam) for fam in pf.symHKL]

iterations = 9

for i in tqdm(range(iterations)):
    
    """ first iteration, skip recalc of PF """
    
    if i == 0: #first iteration is direct from PFs
        
        od_data = np.ones( od.bungeList.shape[0]*od.bungeList.shape[1]*od.bungeList.shape[2] )
        calc_od[0] = np.ones( (od_data.shape[0], numPoles) )        
        
        for fi in range(numPoles): 
            
            temp = np.ones(( od.bungeList.shape[0]*od.bungeList.shape[1]*od.bungeList.shape[2], len(pf.y[fi]) ))
            
            for yi in range(len(pf.y[fi])):
                    
                od_cells = pf_od[fi][yi]['cell']
                wgts = pf_od[fi][yi]['weight']                   
        
                temp[od_cells.astype(int), yi] *= abs(pf.data[fi][yi])
            
            """ zero to 1E-5 """
            temp = np.where(temp==0,1E-5,temp)
            """ log before sum instead of product """
            temp = np.log(temp)
            n = np.count_nonzero(temp,axis=1)
            n = np.where(n == 0, 1, n)
            calc_od[0][:,fi] = np.exp((np.sum(temp,axis=1)*refl_wgt[fi])/n)
        
        calc_od[0] = np.product(calc_od[0],axis=1)**(1/numPoles)
        #place into OD object
        calc_od[0] = bunge(od.res, od.cs, od.ss, weights=calc_od[0])
        calc_od[0].normalize()
        
    """ recalculate poles """
    recalc_pf[i] = {}
    
    for fi in range(numPoles):
        
        recalc_pf[i][fi] = np.zeros(len(pf.y[fi]))
        
        for yi in range(len(pf.y[fi])):
            
            if yi in pf_od[fi]: #pf_cell is defined
                
                od_cells = np.array(pf_od[fi][yi]['cell'])

                recalc_pf[i][fi][yi] = ( 1 / (2*np.pi) ) * ( 1 / sum(pf_od[fi][yi]['weight']) ) * np.sum( pf_od[fi][yi]['weight'] * calc_od[i].weights[od_cells.astype(int)] )    
   
    """ recalculate full pole figures """
    recalc_pf_full[i] = np.zeros((pf_grid.shape[0],pf_grid.shape[1],numPoles))
    
    for fi in range(numPoles):
        
        for pf_cell in np.ravel(pf_grid):
            
            if pf_cell in pf_od_full[fi]: #pf_cell is defined
                
                od_cells = np.array(pf_od_full[fi][pf_cell]['cell'])
                ai, bi = np.divmod(pf_cell, pf_grid.shape[1])
                recalc_pf_full[i][int(ai),int(bi),fi] = ( 1 / np.sum(pf_od_full[fi][pf_cell]['weight']) ) * np.sum( pf_od_full[fi][pf_cell]['weight'] * calc_od[i].weights[od_cells.astype(int)] )
        
    recalc_pf_full[i] = poleFigure(recalc_pf_full[i], pf.hkl, od.cs, 'recalc', resolution=5)
    recalc_pf_full[i].normalize()    
        
    """ (i+1)th inversion """

    od_data = np.ones( od.bungeList.shape[0]*od.bungeList.shape[1]*od.bungeList.shape[2] )
    calc_od[i+1] = np.zeros( (od_data.shape[0], numPoles) )        
    
    for fi in range(numPoles):

        temp = np.ones(( od.bungeList.shape[0]*od.bungeList.shape[1]*od.bungeList.shape[2], len(pf.y[fi]) ))
        
        for yi in range(len(pf.y[fi])):
            
            od_cells = pf_od[fi][yi]['cell']
            wgts = pf_od[fi][yi]['weight']
                
            if recalc_pf[i][fi][yi] == 0: continue
            else: temp[od_cells.astype(int), yi] = ( abs(pf.data[fi][yi]) / recalc_pf[i][fi][yi] )

        """ zero to 1E-5 """
        temp = np.where(temp==0,1E-5,temp)
        """ log sum """
        temp = np.log(temp)
        n = np.count_nonzero(temp,axis=1)
        n = np.where(n == 0, 1, n)
        calc_od[i+1][:,fi] = np.exp((np.sum(temp,axis=1)*refl_wgt[fi])/n)

    calc_od[i+1] = calc_od[i].weights * np.power(np.product(calc_od[i+1],axis=1),(1/numPoles))

    #place into OD object
    calc_od[i+1] = bunge(od.res, od.cs, od.ss, weights=calc_od[i+1])
    calc_od[i+1].normalize() 
        
recalc_pf_full[iterations-1].plot(pfs=3)
# %%

### FIBER PLOT ###

# pf_num = 0

# import mayavi.mlab as mlab
# # import matplotlib.pyplot as plt
# # fig = plt.figure()

# mlab.figure(bgcolor=(1,1,1))

# ## grid ##
# gd = mlab.points3d(bungeAngs[:,0],bungeAngs[:,1],bungeAngs[:,2],scale_factor=1,mode='point',color=(0,0,0))
# gd.actor.property.render_points_as_spheres = True
# gd.actor.property.point_size = 3
  
# ## lit point ##
# gd = mlab.points3d(0,0,0,scale_factor=1,mode='point',color=(1,0,0))
# gd.actor.property.render_points_as_spheres = True
# gd.actor.property.point_size = 5

# ## manual fibre ##
# gd2 = mlab.points3d(0,0,0,scale_factor=1,mode='point',color=(0,1,0))
# gd2.actor.property.render_points_as_spheres = True
# gd2.actor.property.point_size = 5   

# plt_list = list(fibre_e_full[pf_num].keys())
# plt_list.sort()

# @mlab.animate(delay=50)
# def anim():
#     while True:
        
#         for yi in plt_list:
                
# #            gd2.mlab_source.reset( x = fibre_wimv[pf_num][yi][:,0],
# #                                   y = fibre_wimv[pf_num][yi][:,1],
# #                                   z = fibre_wimv[pf_num][yi][:,2])
            
#             gd.mlab_source.reset( x = fibre_e_full[pf_num][yi][:,0],
#                                   y = fibre_e_full[pf_num][yi][:,1],
#                                   z = fibre_e_full[pf_num][yi][:,2])
            
#             gd2.mlab_source.reset( x = egrid_trun[pf_num][yi][:,0],
#                                     y = egrid_trun[pf_num][yi][:,1],
#                                     z = egrid_trun[pf_num][yi][:,2])
            
#             # tubePts = nn_gridPts_full[pf_num][yi]
            
#             # gd2.mlab_source.reset( x = bungeAngs[tubePts.astype(int),0],
#             #                       y = bungeAngs[tubePts.astype(int),1],
#             #                       z = bungeAngs[tubePts.astype(int),2])
        
#             yield
            
# anim()

# #for yi in range(len(pf.y[pf_num])):
# #    
# #        gd = mlab.points3d(fibre[pf_num][yi][:,0],fibre[pf_num][yi][:,1],fibre[pf_num][yi][:,2],scale_factor=1,mode='point',color=(1,0,0))
# #        gd.actor.property.render_points_as_spheres = True
# #        gd.actor.property.point_size = 5    

# mlab.show(stop=True)