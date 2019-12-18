#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 09:46:56 2019

@author: nate
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
cellSize = np.deg2rad(10)

hkls = []
files = []

datadir = os.path.join(dir_path,'Data','NOMAD Aluminum - no abs','pole figures','combined')

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
    
""" rotate """

rot = R.from_euler('XZY',(13,-88,90), degrees=True).as_dcm()

pf = poleFigure(files,hkls,crystalSym,'nd')
pf.rotate(rot)

od = bunge(cellSize, crystalSym, sampleSym)
hkls = np.array(hkls)

symHKL = symmetrise(crystalSym, hkls)
symHKL = normalize(symHKL)

omega = np.radians(np.arange(0,360+5,5))
    
""" symmetry after """

fibre_e = {}
fibre_q = {}
weights = {}
hkls = normalize(hkls)

""" search for unique hkls to save time """
hkls_loop, hkls_loop_idx = np.unique(hkls,axis=0,return_inverse=True)

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

""" start loop """

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

""" determine radius for k=3 sep """

tree = BallTree(qgrid,metric=quatMetricNumba2)
rad = np.max(tree.query(qgrid,k=2)[0])

# %%

""" calculate pf grid XYZ for fibre """

pf_grid, alp, bet = pf.grid(full=True, ret_ab=True)

x_pf = np.sin(alp) * np.cos(bet)
y_pf = np.sin(alp) * np.sin(bet)
z_pf = np.cos(alp)

xyz_pf = np.stack((x_pf,y_pf,z_pf),axis=2)
xyz_pf = xyz_pf.reshape((alp.shape[0]*alp.shape[1],3))

fibre_full_e = {}
fibre_full_q = {}

nn_gridPts_full = {}
nn_gridDist_full = {}

""" start loop """

def calcFibre(h, pf_y, omega):
    
    fibre_e = {}
    fibre_q = {}
    
    nn_gridPts = {}
    nn_gridDist = {}
    
    h_y = np.cross(h, pf_y)
    h_yAng = np.arccos(np.dot(pf_y, h))
    qh_y = np.column_stack((np.cos(h_yAng/2),np.sin(h_yAng/2)[:,None]*h_y))
    qh_y = quat.normalize(qh_y)
    
    qy_om = np.column_stack((np.cos(omega/2), np.sin(omega/2)))
    
#    qh_y = quat.vector_vector_rotation(h,pf_y)
#    qh_om = quat.from_axis_angle(h,omega)
    
    for yi,y in enumerate(pf_y):
        
        qy_om = np.column_stack((np.cos(omega/2), np.sin(omega/2)[:,None]*np.tile(y,(len(omega),1))))
        qy_om = quat.normalize(qy_om)
    
        qfib = quat.multiply(qy_om,qh_y[yi,:])
        
        """  reshape for tiling, simplify?? """
        
        qfib = qfib.T.reshape((1,4,len(omega)),order='F')
        qfib = np.tile(qfib,(quatSymOps.shape[1],1,1))
        qfib = qfib.transpose((2,0,1))
    
        """ apply symmetry """
    
        qfib = quat.multiply(quatSymOps,qfib)
        
        """ symmetry ops, then filter by given maximum bunge angles """
        
        phi1, Phi, phi2 = quat2eu(qfib)
            
        phi1 = np.where(phi1 < 0, phi1 + 2*np.pi, phi1) #brnng back to 0 - 2pi
        Phi = np.where(Phi < 0, Phi + np.pi, Phi) #brnng back to 0 - pi
        phi2 = np.where(phi2 < 0, phi2 + 2*np.pi, phi2) #brnng back to 0 - 2pi
        
        eu_fib = np.stack( (phi1, Phi, phi2), axis=2 )
        eu_fib = np.reshape( eu_fib, (eu_fib.shape[0]*eu_fib.shape[1], eu_fib.shape[2]) )
        
        fz = (eu_fib[:,0] < od._phi1max) & (eu_fib[:,1] < od._Phimax) & (eu_fib[:,2] < od._phi2max)
        fz_idx = np.nonzero(fz)
        
        fibre_e[yi] = eu_fib[fz]

        fib_idx = np.unravel_index(fz_idx[0], (qfib.shape[0],qfib.shape[1]))
        
        fibre_q[yi] = qfib[fib_idx].reshape((len(fz_idx[0]),4))
        
        """ distance calc """
        temp = quatMetricNumba(qgrid,qfib[fib_idx])
        """ find tube """
        tube = (temp <= rad)
        temp = np.column_stack((np.argwhere(tube)[:,0],temp[tube]))
        """ sort by min distance """
        temp = temp[np.argsort(temp[:,1],axis=0)]
        """ return unique pts (first in list) """
        uni_pts = np.unique(temp[:,0],return_index=True)

        nn_gridPts[yi] = uni_pts[0]
        nn_gridDist[yi] = temp[uni_pts[1],1]
        
    return nn_gridPts, nn_gridDist, fibre_e

""" multiprocessing using parmap """

import parmap as pm

inputs = []
inputs_full = []

for hi, h in enumerate(hkls):
    
    inputs.append((h,pf.y[hi]))
    
for hi, h in enumerate(hkls_loop):
    
    inputs_full.append((h,xyz_pf))

outputs = pm.starmap(calcFibre, inputs, omega, pm_pbar=True, pm_processes=7)
outputs_full = pm.starmap(calcFibre, inputs_full, omega, pm_pbar=True, pm_processes=7)

nn_gridPts = {}
nn_gridDist = {}

nn_gridPts_full = {}
nn_gridDist_full = {}

fibre_e = {}

for hi, h in enumerate(hkls):
    
    nn_gridPts[hi] = outputs[hi][0]
    nn_gridDist[hi] = outputs[hi][1]
    fibre_e[hi] = outputs[hi][2]

for i,hi in enumerate(hkls_loop_idx):
    
    nn_gridPts_full[i] = outputs_full[hi][0]
    nn_gridDist_full[i] = outputs_full[hi][1]

# %%

""" extract distances -> weights, construct pointer 
    pf_od: pf --> pf to od
    od_pf: od --> od to pf
    each entry stored as dict; ['cell'] is cell #s ['weights'] is weights """
    
def calcPointers(gridPts, gridDist, hkls, tube_exp=2):

    pf_od = {}
    odwgts_tot = np.zeros( ( len(hkls), od.bungeList.shape[0]*od.bungeList.shape[1]*od.bungeList.shape[2] ) )
    
    for hi, h in enumerate(hkls):
        
        pf_od[hi] = {}
        
        for yi in range(len(gridPts[hi].keys())):
            
            od_cells = gridPts[hi][yi]
#            weights = 1 / ( (nn_gridDist[hi][yi])**tube_exp )
            weights = 1 / ( gridDist[hi][yi]**0.75 )
            
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
    
    return pf_od, odwgts_tot

""" not using pfwgts_full """
""" need to simplify the two sets of pointers, maybe combine both?? """

pf_od, odwgts = calcPointers(nn_gridPts, nn_gridDist, hkls)
pf_odfull, odwgts_full = calcPointers(nn_gridPts_full, nn_gridDist_full, hkls)
        
# %%
    
""" e-wimv iteration start """
        
od_data = np.ones( od.bungeList.shape[0]*od.bungeList.shape[1]*od.bungeList.shape[2] )
calc_od = {}
recalc_pf = {}

recalc_pf_full = {}
    
numPoles = pf._numHKL
numHKLs = [len(fam) for fam in pf.symHKL]

iterations = 10

print('starting ewimv')

for i in tqdm(range(iterations)):
    
    """ first iteration, skip recalc of PF """
    
    if i == 0: #first iteration is direct from PFs
        
        od_data = np.ones( od.bungeList.shape[0]*od.bungeList.shape[1]*od.bungeList.shape[2] )
        calc_od[0] = np.ones( (od_data.shape[0], numPoles) )        
        
        for fi in range(numPoles): 
            
            temp = np.ones(( od.bungeList.shape[0]*od.bungeList.shape[1]*od.bungeList.shape[2], len(pf.y[fi]) ))
                
            for yi in range(len(pf.y[fi])): #looping through discrete positions
                
                od_cells = pf_od[fi][yi]['cell']
                wgts = pf_od[fi][yi]['weight']
                
#                od_data[od_cells.astype(int)] *= wgts*abs(pf.data[fi][yi]) #multiplied by weights (distance)
#                temp[od_cells.astype(int),yi] *= wgts*abs(pf.data[fi][yi]) #multiplied by weights (distance)
                temp[od_cells.astype(int),yi] *= wgts*abs(pf.data[fi][yi]) #multiplied by weights (distance)
            
            od_data = np.product(temp, axis=1)
            
#            """ multiply by 1/wgt for each cell """
#            
#            od_cell_idx = np.nonzero(odwgts[fi,:])
#            od_data[od_cell_idx] *= np.ravel(odwgts[fi,od_cell_idx])
            
            """ exponent (geometric avg) """
            
#            calc_od[0][:,fi] = np.power(od_data,(refl_wgt[fi]/numHKLs[fi]))
            calc_od[0][:,fi] = np.power(od_data,(refl_wgt[fi]/len(omega)))


        calc_od[0] = np.product(calc_od[0],axis=1)**(1/numPoles)
        #place into OD object
        calc_od[0] = bunge(od.res, od.cs, od.ss, weights=calc_od[0])
        calc_od[0].normalize()

    """ recalculate pole figures """
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
            
            if pf_cell in pf_odfull[fi]: #pf_cell is defined
                
                od_cells = np.array(pf_odfull[fi][pf_cell]['cell'])
                ai, bi = np.divmod(pf_cell, pf_grid.shape[1])
                recalc_pf_full[i][int(ai),int(bi),fi] = ( 1 / np.sum(pf_odfull[fi][pf_cell]['weight']) ) * np.sum( pf_odfull[fi][pf_cell]['weight'] * calc_od[i].weights[od_cells.astype(int)] )
        
    recalc_pf_full[i] = poleFigure(recalc_pf_full[i], pf.hkl, od.cs, 'recalc', resolution=5)
    recalc_pf_full[i].normalize()    
    
    """ compare recalculated to experimental """
    
#        RP_err = {}
#        prnt_str = None
#        
#        for fi in range(numPoles):
#            
#            expLim = pf.data[fi].shape
#            RP_err[fi] = ( pf.data[fi] - recalc_pf[i].data[fi][:expLim[0],:expLim[1]] ) / pf.data[fi]
#            RP_err[fi] = np.sum(RP_err[fi])
#            
#            if prnt_str is None: prnt_str = 'RP Error: {}'.format(round(RP_err[fi],ndigits=2))
#            else: prnt_str += ' | {}'.format(round(RP_err[fi],ndigits=2))
#            
#        print(prnt_str)
        
    """ (i+1)th inversion """

    od_data = np.ones( od.bungeList.shape[0]*od.bungeList.shape[1]*od.bungeList.shape[2] )
    calc_od[i+1] = np.zeros( (od_data.shape[0], numPoles) )        
    
    for fi in range(numPoles):
        
        temp = np.ones(( od.bungeList.shape[0]*od.bungeList.shape[1]*od.bungeList.shape[2], len(pf.y[fi]) ))
            
        for yi in range(len(pf.y[fi])): #looping through discrete positions
            
            od_cells = pf_od[fi][yi]['cell']
            wgts = pf_od[fi][yi]['weight']
            
#            try:
#                od_cells = np.array(pf_od[fi][pf_cell])
#                ai, bi = np.divmod(pf_cell, pf_grid.shape[1])
#            except: continue

#            if pf_cell < pf.data[fi].shape[0]*pf.data[fi].shape[1]: #inside of measured PF range
                
            if recalc_pf[i][fi][yi] == 0: continue
            else: temp[od_cells.astype(int),yi] *= ( abs(pf.data[fi][yi]) / recalc_pf[i][fi][yi] )
            
#            temp[od_cells.astype(int),yi] *= ( abs(pf.data[fi][yi]) / recalc_pf[i][fi][yi] )
        
        od_data = np.product(temp, axis=1)
        
        """ loop over od_cells (alternative) """
    #    for od_cell in tqdm(np.ravel(od.bungeList)):
           
    #        pf_cells = od_pf[fi][od_cell]
           
    #        pf_cellMax = pf.data[fi].shape[0]*pf.data[fi].shape[1]
    #        pf_cells = pf_cells[pf_cells < pf_cellMax]
           
    #        ai, bi = np.divmod(pf_cells, pf_grid.shape[1])
    #        od_data[int(od_cell)] = np.product( pf.data[fi][ai.astype(int),bi.astype(int)] / recalc_pf[i].data[fi][ai.astype(int), bi.astype(int)] )
                
#        calc_od[i+1][:,fi] = np.power(od_data,(refl_wgt[fi]/numHKLs[fi]))
        calc_od[i+1][:,fi] = np.power(od_data,(refl_wgt[fi]/len(omega)))
    
    calc_od[i+1] = calc_od[i].weights * np.power(np.product(calc_od[i+1],axis=1),(1/numPoles))
    
    #place into OD object
    calc_od[i+1] = bunge(od.res, od.cs, od.ss, weights=calc_od[i+1])
    calc_od[i+1].normalize()    

        
# %%

### FIBER PLOT ###
    
#pt = pointer[:,:,pf_num]

#pf_num = 0
#
#import mayavi.mlab as mlab
# # import matplotlib.pyplot as plt
# # fig = plt.figure()
#
#mlab.figure(bgcolor=(1,1,1))
#
## ## grid ##
#gd = mlab.points3d(bungeAngs[:,0],bungeAngs[:,1],bungeAngs[:,2],scale_factor=1,mode='point',color=(0,0,0))
#gd.actor.property.render_points_as_spheres = True
#gd.actor.property.point_size = 3
#  
# ## lit point ##
#gd = mlab.points3d(0,0,0,scale_factor=1,mode='point',color=(1,0,0))
#gd.actor.property.render_points_as_spheres = True
#gd.actor.property.point_size = 8
#
# ## manual fibre ##
#gd2 = mlab.points3d(0,0,0,scale_factor=1,mode='point',color=(0,1,0))
#gd2.actor.property.render_points_as_spheres = True
#gd2.actor.property.point_size = 5   
#
#@mlab.animate(delay=1000)
#def anim():
#    while True:
#        
#        for yi in range(len(pf.y[pf_num])):
#                      
#            gd.mlab_source.reset( x = fibre_e[pf_num][yi][:,0],
#                                  y = fibre_e[pf_num][yi][:,1],
#                                  z = fibre_e[pf_num][yi][:,2])
#            
#            tubePts = nn_gridPts[pf_num][yi]
#            
#            gd2.mlab_source.reset( x = bungeAngs[tubePts.astype(int),0],
#                                   y = bungeAngs[tubePts.astype(int),1],
#                                   z = bungeAngs[tubePts.astype(int),2])
#        
#            yield
#            
#anim()
# 
##for yi in range(len(pf.y[pf_num])):
##    
##        gd = mlab.points3d(fibre[pf_num][yi][:,0],fibre[pf_num][yi][:,1],fibre[pf_num][yi][:,2],scale_factor=1,mode='point',color=(1,0,0))
##        gd.actor.property.render_points_as_spheres = True
##        gd.actor.property.point_size = 5    
# 
#mlab.show(stop=True)
#        

    






        
        
        