#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 21:43:16 2019

@author: nate
"""

"""

E-WIMV validation with XRD pole figs

"""

import os

import numpy as np
from tqdm import trange
import rowan as quat

from pymatgen.core import Lattice, Structure

from pyTex import poleFigure, bunge
from pyTex.orientation import eu2quat, quat2eu 
from pyTex.utils import normalize, genSym
from pyTex.diffrac import calc_XRDreflWeights

dir_path = os.path.dirname(os.path.realpath('__file__'))
data_path = os.path.join(dir_path, 'Data')

P = 1

crystalSym = 'm-3m'
sampleSym = '1'
cellSize = np.deg2rad(5)
hkls = [(1,1,1),(2,0,0),(2,2,0)]

bkgd111path = os.path.join(data_path, '32.5_ bkgd.xrdml')
bkgd200path = os.path.join(data_path, '55_bkgd.xrdml')
bkgd220path = os.path.join(data_path, '55_bkgd.xrdml')

bkgds = [bkgd111path,bkgd200path,bkgd220path]
bkgd = poleFigure(bkgds, hkls, crystalSym, 'xrdml',subtype='bkgd')

def111path = os.path.join(data_path, 'defocus_38.xrdml')
def200path = os.path.join(data_path, 'defocus_45.xrdml')
def220path = os.path.join(data_path, 'defocus_65.xrdml')

defs = [def111path,def200path,def220path]
defocus = poleFigure(defs, hkls, crystalSym, 'xrdml',subtype='defocus')

pf111path = os.path.join(data_path, '111pf_2T=38.xrdml')
pf200path = os.path.join(data_path, '200pf_2T=45.xrdml')
pf220path = os.path.join(data_path, '220pf_2theta=65.xrdml')

pfs = [pf111path,pf200path,pf220path]
pf = poleFigure(pfs, hkls, crystalSym, 'xrdml')

pf.correct(bkgd=bkgd,defocus=defocus)
pf.normalize()

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

od = bunge(cellSize, crystalSym, sampleSym)

""" refl weights """

a = 4.046 #Angstrom
latt = Lattice.cubic(a)
structure = Structure(latt, ["Al", "Al", "Al", "Al"], [[0, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]])
rad = 'CuKa'

refl_wgt = calc_XRDreflWeights(structure, hkls, rad='CuKa')
refl_wgt = {0:1.0, 1:1.0, 2:1.0}

omega = np.radians(np.arange(0,365,5))
    
""" symmetry after """

hkls = normalize(np.array(hkls))

symOps = genSym(crystalSym)
symOps = np.unique(np.swapaxes(symOps,2,0),axis=0)

""" only use proper rotations """
""" complicated, simplify? """

proper = np.where( np.linalg.det(symOps) == 1 ) #proper orthogonal
quatSymOps = quat.from_matrix(symOps[proper])
quatSymOps = np.tile(quatSymOps[:,:,np.newaxis],(1,1,len(omega)))
quatSymOps = quatSymOps.transpose((2,0,1))

sliceIndex = range(omega.shape[0])

""" gen quats from bunge grid """

bungeAngs = np.zeros(( np.product(od.phi1cen.shape), 3 ))

for ii,i in enumerate(np.ndindex(od.phi1cen.shape)):
    
    bungeAngs[ii,:] = np.array((od.phi1cen[i],od.Phicen[i],od.phi2cen[i]))

qgrid = eu2quat(bungeAngs).T

# %%

""" start pointer loop """

from sklearn.neighbors import BallTree
from numba import jit

@jit(nopython=True)
def quatMetricNumba(a, b):
    
    """ from DOI 10.1007/s10851-009-0161-2, #4 """
    
    dist = np.zeros((len(a),len(b)))
    
    for bi in range(len(b)):
        
        dist[:,bi] = 1 - np.abs(np.dot(a,b[bi]))
    
    return dist

@jit(nopython=True)
def quatMetricNumba2(a,b):
    
    """ from DOI 10.1007/s10851-009-0161-2, #4 """
    
    return 1 - np.abs(np.dot(a,b))

""" distance =~ ( 1 - cos(θ) ) / 2 """

tree = BallTree(qgrid,metric=quatMetricNumba2)
theta = np.deg2rad(7.5)
rad = ( 1 - np.cos(theta) ) / 2
# rad = np.max(tree.query(qgrid,k=4)[0])
# print(rad)

""" start loop """

fibre_e = {}
fibre_q = {}

nn_gridPts = {}
nn_gridDist = {}

for fi,fam in enumerate(pf.symHKL):
    
    fibre_e[fi] = {}
    fibre_q[fi] = {}
    
    nn_gridPts[fi] = {}
    nn_gridDist[fi] = {}

    for yi in trange(len(xyz_pf)):
        
        y = xyz_pf[yi]
        
    #    for fi,fam in enumerate(pf.symHKL):
            
        axis = np.cross(fam,y)
        angle = np.arccos(np.dot(fam,y))
        
        q0 = quat.from_axis_angle(axis, angle)
        q0_n = quat.normalize(q0) 
        
        q1_n = [quat.normalize(quat.from_axis_angle(h, omega)) for h in fam]
        
         # eu = np.zeros((len(omega),3,fam.shape[0]))
        eu2 = []
        
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
        
#        fibre_q[fi][yi] = qfib[fib_idx].reshape((len(fz_idx[0]),4))
        fibre_q[fi][yi] = qfib[fib_idx]
        
        """ distance calc """
        temp = quatMetricNumba(qgrid,qfib[fib_idx])
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
        
        nn_gridPts[fi][yi] = uni_pts[0]
        nn_gridDist[fi][yi] = temp[uni_pts[1],1]
        
#fibre_e = {}
#fibre_q = {}
#
#nn_gridPts = {}
#nn_gridDist = {}
#    
#for hi in range(len(hkls)):
#    
#    h = hkls[hi]
#
#    fibre_e[hi] = {}
#    fibre_q[hi] = {}
#    
#    nn_gridPts[hi] = {}
#    nn_gridDist[hi] = {}
#    
##    qh_y = quat.vector_vector_rotation(h,xyz_pf)
##    qh_om = quat.from_axis_angle(h,omega)    
#    
#    h_y = np.cross(h, xyz_pf)
#    h_yAng = np.arccos(np.dot(xyz_pf, h))
#    
##    qh_y = np.column_stack((np.cos(h_yAng/2),np.sin(h_yAng/2)[:,None]*h_y))
##    qh_y = quat.normalize(qh_y)
##    
##    qy_om = np.column_stack((np.cos(omega/2), np.sin(omega/2)))
#
##    q0 = np.column_stack((np.cos(h_yAng/2),np.sin(h_yAng/2)[:,None]*h_y))
#    q0 = quat.from_axis_angle(h_y, h_yAng)
#    q0 = quat.normalize(q0) 
#    
##    q1 = np.column_stack((np.cos(omega/2), np.sin(omega/2)[:,None]*np.tile(h,(len(omega),1))))
#    q1 = quat.from_axis_angle(h, omega)
#    q1 = quat.normalize(q1) 
#    
#    for yi in trange(len(xyz_pf)):
#        
#        y = xyz_pf[yi]
#
#        """first attempt"""
##        qfib = quat.multiply(qh_y[yi,:],qh_om)
#        """second attempt"""
##        qy_om = np.column_stack((np.cos(omega/2), np.sin(omega/2)[:,None]*np.tile(y,(len(omega),1))))
##        qy_om = quat.normalize(qy_om)
##        qfib = quat.multiply(qy_om,qh_y[yi,:])
#        """third attempt"""
#        qfib = quat.multiply(q0[yi,:],q1)
#        
#        """  reshape for tiling, simplify?? """
#        
#        qfib = qfib.T.reshape((1,4,len(omega)),order='F')
#        qfib = np.tile(qfib,(quatSymOps.shape[1],1,1))
#        qfib = qfib.transpose((2,0,1))
#    
#        """ apply symmetry """
#    
#        qfib = quat.multiply(quatSymOps,qfib)
#        
#        """ symmetry ops, then filter by given maximum bunge angles """
#        
#        phi1, Phi, phi2 = quat2eu(qfib)
#            
#        phi1 = np.where(phi1 < 0, phi1 + 2*np.pi, phi1) #brnng back to 0 - 2pi
#        Phi = np.where(Phi < 0, Phi + np.pi, Phi) #brnng back to 0 - pi
#        phi2 = np.where(phi2 < 0, phi2 + 2*np.pi, phi2) #brnng back to 0 - 2pi
#        
#        eu_fib = np.stack( (phi1, Phi, phi2), axis=2 )
#        eu_fib = np.reshape( eu_fib, (eu_fib.shape[0]*eu_fib.shape[1], eu_fib.shape[2]) )
#        
#        fibre_e[hi][yi] = eu_fib
#        
#        fz = (eu_fib[:,0] < od._phi1max) & (eu_fib[:,1] < od._Phimax) & (eu_fib[:,2] < od._phi2max)
#        fz_idx = np.nonzero(fz)
#        
#        fibre_e[hi][yi] = eu_fib[fz]
#
#        fib_idx = np.unravel_index(fz_idx[0], (qfib.shape[0],qfib.shape[1]))
#        
#        fibre_q[hi][yi] = qfib[fib_idx].reshape((len(fz_idx[0]),4))
#        
#        """ distance calc """
#        temp = quatMetricNumba(qgrid,qfib[fib_idx])
#        """ find tube """
#        tube = (temp <= rad)
#        temp = np.column_stack((np.argwhere(tube)[:,0],temp[tube]))
#        """ sort by min distance """
#        temp = temp[np.argsort(temp[:,1],axis=0)]
#        """ return unique pts (first in list) """
#        uni_pts = np.unique(temp[:,0],return_index=True)
#
#        nn_gridPts[hi][yi] = uni_pts[0]
#        nn_gridDist[hi][yi] = temp[uni_pts[1],1]

    
# %%

""" extract distances -> weights, construct pointer 
    pf_od: pf --> pf to od
    od_pf: od --> od to pf
    each entry stored as dict; ['cell'] is cell #s ['weights'] is weights """
    
tube_exp = 0.5

pf_od = {}
odwgts_tot = np.zeros( ( len(hkls), od.bungeList.shape[0]*od.bungeList.shape[1]*od.bungeList.shape[2] ) )

test = []

for hi, h in enumerate(hkls):
    
    pf_od[hi] = {}
    
    for yi in range(len(nn_gridPts[hi].keys())):
        
        od_cells = nn_gridPts[hi][yi]
        #TODO: work on scaling towards max distance
        scaled_dist = nn_gridDist[hi][yi]
        weights = 1 / ( ( scaled_dist )**tube_exp )
        test.append(np.min(nn_gridDist[hi][yi]))
        
        if np.any(weights < 0): raise ValueError('neg weight')
            
        if np.any(weights == 0): raise ValueError('zero weight')
        
        pf_od[hi][yi] = {'cell': od_cells, 'weight': weights}
        
        odwgts_tot[hi,od_cells.astype(int)] += weights
        
#            for oc,wgt in zip(od_cells,weights):
#                
#                if oc in od_pf[hi]:
#                    
#                    od_pf[hi][oc]['cell'].append(yi)
#                    od_pf[hi][oc]['weight'].append(wgt)
#                
#                else:
#                    
#                    od_pf[hi][oc] = {'cell':[], 'weight':[]}
#                    od_pf[hi][oc]['cell'].append(yi)
#                    od_pf[hi][oc]['weight'].append(wgt)
        
#        """ build pointer for 1/wgt """
#        all_cells = list(od_pf[hi].keys())  
        
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

iterations = 12

#tq.write('starting e-wimv')

for i in trange(iterations):
    
    """ first iteration, skip recalc of PF """
    
    if i == 0: #first iteration is direct from PFs
        
        od_data = np.ones( od.bungeList.shape[0]*od.bungeList.shape[1]*od.bungeList.shape[2] )
        calc_od[0] = np.ones( (od_data.shape[0], numPoles) )        
        
        for fi in range(numPoles): 
            
            temp = np.ones(( od.bungeList.shape[0]*od.bungeList.shape[1]*od.bungeList.shape[2], int(np.max(pf_grid)) ))
            
            for pf_cell in np.ravel(pf_grid):
                
                test = []
                    
                try:
                    od_cells = pf_od[fi][pf_cell]['cell']
                    wgts = pf_od[fi][pf_cell]['weight']
                    
                    ai, bi = np.divmod(pf_cell, pf_grid.shape[1])
                    
                except:
                    continue
                
                if pf_cell < pf.data[fi].shape[0]*pf.data[fi].shape[1]: #inside of measured PF range
        
                    test.append(pf.data[fi][int(ai),int(bi)])
                    temp[od_cells.astype(int),int(pf_cell)] *= pf.data[fi][int(ai),int(bi)] #multiplied by weights (distance)
                    # od_data[od_cells.astype(int)] *= wgts*pf.data[fi][int(ai),int(bi)] #multiplied by weights (distance)
                    # od_data[od_cells.astype(int)] *= pf.data[fi][int(ai),int(bi)]*1000
            
            """ log before sum instead of product """
            temp = np.log(temp)
            n = np.count_nonzero(temp,axis=1)
            calc_od[0][:,fi] = np.exp(np.sum(temp,axis=1)/n)
            
# #            od_data = np.product(temp, axis=1)
#             # calc_od[0][:,fi] = od_data
            
#             # """ multiply by 1/wgt for each cell """
            
#             # od_cell_idx = np.nonzero(odwgts_tot[fi,:])
#             # od_data[od_cell_idx] *= np.ravel(odwgts_tot[fi,od_cell_idx])
            
#             """ exponent (geometric avg) """
            
# #            calc_od[0][:,fi] = np.power(od_data,(1/numHKLs[fi]))
#             # calc_od[0][:,fi] = np.power(od_data,(refl_wgt[fi]/len(omega)))


        calc_od[0] = np.product(calc_od[0],axis=1)**(1/numPoles)
        #place into OD object
        calc_od[0] = bunge(od.res, od.cs, od.ss, weights=calc_od[0])
        calc_od[0].normalize()
   
        """ recalculate full pole figures """
    recalc_pf[i] = np.zeros((pf_grid.shape[0],pf_grid.shape[1],numPoles))
    
    for fi in range(numPoles):
        
        for pf_cell in np.ravel(pf_grid):
            
            if pf_cell in pf_od[fi]: #pf_cell is defined
                
                od_cells = np.array(pf_od[fi][pf_cell]['cell'])
                ai, bi = np.divmod(pf_cell, pf_grid.shape[1])
                recalc_pf[i][int(ai),int(bi),fi] = ( 1 / np.sum(pf_od[fi][pf_cell]['weight']) ) * np.sum( pf_od[fi][pf_cell]['weight'] * calc_od[i].weights[od_cells.astype(int)] )
#                recalc_pf[i][int(ai),int(bi),fi] = ( 1 / len(od_cells) ) * np.sum( calc_od[i].weights[od_cells.astype(int)] )
        
    recalc_pf[i] = poleFigure(recalc_pf[i], pf.hkl, od.cs, 'recalc', resolution=5)
    recalc_pf[i].normalize()    
        
    """ (i+1)th inversion """

    od_data = np.ones( od.bungeList.shape[0]*od.bungeList.shape[1]*od.bungeList.shape[2] )
    calc_od[i+1] = np.zeros( (od_data.shape[0], numPoles) )        
    
    for fi in range(numPoles):

        temp = np.ones(( od.bungeList.shape[0]*od.bungeList.shape[1]*od.bungeList.shape[2], int(np.max(pf_grid)) ))
        
        for pf_cell in np.ravel(pf_grid):
            
            try:
                od_cells = pf_od[fi][pf_cell]['cell']
                wgts = pf_od[fi][pf_cell]['weight']
                
                ai, bi = np.divmod(pf_cell, pf_grid.shape[1])

            except: continue

            if pf_cell < pf.data[fi].shape[0]*pf.data[fi].shape[1]: #inside of measured PF range
                
                if recalc_pf[i].data[fi][int(ai),int(bi)] == 0: continue
                else: temp[od_cells.astype(int), int(pf_cell)] = ( pf.data[fi][int(ai),int(bi)] / recalc_pf[i].data[fi][int(ai),int(bi)] )
                
        temp = np.log(temp)
        n = np.count_nonzero(temp,axis=1)
        calc_od[i+1][:,fi] = np.exp(np.sum(temp,axis=1)/n)

        # od_data = np.product(temp, axis=1)
        
        # calc_od[i+1][:,fi] = np.power(od_data,(refl_wgt[fi]/len(omega)))
#        calc_od[i+1][:,fi] = np.power(od_data,(1/numHKLs[fi]))

    calc_od[i+1] = calc_od[i].weights * np.power(np.product(calc_od[i+1],axis=1),(1/numPoles))

    #place into OD object
    calc_od[i+1] = bunge(od.res, od.cs, od.ss, weights=calc_od[i+1])
    calc_od[i+1].normalize() 
    
recalc_pf[11].plot(cmap='viridis_r')


# %%

### load in WIMV pointer ###

#import pickle
#f = open("file.pkl","rb")
#fibre_wimv = pickle.load(f)
#f.close()


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

# plt_list = list(fibre_e[pf_num].keys())
# plt_list.sort()

# @mlab.animate(delay=100)
# def anim():
#     while True:
        
#         for yi in plt_list:
                
# #            gd2.mlab_source.reset( x = fibre_wimv[pf_num][yi][:,0],
# #                                   y = fibre_wimv[pf_num][yi][:,1],
# #                                   z = fibre_wimv[pf_num][yi][:,2])
            
#             gd.mlab_source.reset( x = fibre_e[pf_num][yi][:,0],
#                                   y = fibre_e[pf_num][yi][:,1],
#                                   z = fibre_e[pf_num][yi][:,2])
            
#             tubePts = nn_gridPts[pf_num][yi]
            
#             gd2.mlab_source.reset( x = bungeAngs[tubePts.astype(int),0],
#                                   y = bungeAngs[tubePts.astype(int),1],
#                                   z = bungeAngs[tubePts.astype(int),2])
        
#             yield
            
# anim()

# #for yi in range(len(pf.y[pf_num])):
# #    
# #        gd = mlab.points3d(fibre[pf_num][yi][:,0],fibre[pf_num][yi][:,1],fibre[pf_num][yi][:,2],scale_factor=1,mode='point',color=(1,0,0))
# #        gd.actor.property.render_points_as_spheres = True
# #        gd.actor.property.point_size = 5    

# mlab.show(stop=True)
        