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

import os,sys
from math import pi

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import rowan as quat

sys.path.append('/home/nate/projects/pyTex/')
from pyTex import poleFigure, bunge
from pyTex.orientation import eu2quat, quat2eu 
from pyTex.utils import symmetrise, normalize, genSymOps
from pyTex.diffrac import calc_NDreflWeights

dir_path = os.path.dirname(os.path.realpath('__file__'))

P = 1

crystalSym = 'm-3m'
sampleSym = '1'
cellSize = np.deg2rad(5)
theta = np.deg2rad(7)
sampleName = 'Al_peakInt_5x7'

""" NRSF2 .jul """

data_path = os.path.join(dir_path, 'Data', 'HB2B - Aluminum')
hkls = np.array([(2,2,2), (3,1,1), (4,0,0)])

pf222path = os.path.join(data_path, 'HB2B_exp129_3Chi_222.jul')
pf311path = os.path.join(data_path, 'HB2B_exp129_3Chi_311.jul')
pf400path = os.path.join(data_path, 'HB2B_exp129_3Chi_400.jul')

pfs = [pf222path,pf311path,pf400path]
rot = R.from_euler('XZX', (90,90,90), degrees=True).as_dcm()
pf = poleFigure(pfs, hkls, crystalSym, 'jul')

""" peak-fitted pole figures """

# hkls = []
# files = []

# # datadir = os.path.join(dir_path,'Data','NOMAD Aluminum - no abs','combined')
# # datadir = os.path.join(dir_path,'Data','NOMAD Nickel - full abs - peak int','pole figures','combined')
# datadir = os.path.join(dir_path,'Data','NOMAD Aluminum - no abs - peak int','combined')
# # datadir = '/media/nate/2E7481AA7481757D/Users/Nate/Dropbox/ORNL/Texture/NRSF2/mtex_export'

# for file in os.listdir(datadir):
    
#     pfName = file.split(')')[0].split('(')[1]
    
#     try:
#         hkls.append(tuple([int(c) for c in pfName]))
#         files.append(os.path.join(datadir,file))
#     except: #not hkls
#         continue
    
#     sortby = [sum([c**2 for c in h]) for h in hkls]
#     hkls = [x for _, x in sorted(zip(sortby,hkls), key=lambda pair: pair[0])]
#     files = [x for _, x in sorted(zip(sortby,files), key=lambda pair: pair[0])]
    
# rot = R.from_euler('XZY',(13,-88,90), degrees=True).as_dcm()
# pf = poleFigure(files,hkls,crystalSym,'nd')

""" rotate """

pf.rotate(rot)

od = bunge(cellSize, crystalSym, sampleSym)
hkls = np.array(hkls)

phi = np.linspace(0,2*pi,73)
    
""" symmetry after """

fibre_e = {}
fibre_q = {}
weights = {}
refls = symmetrise(crystalSym,hkls)
hkls = normalize(hkls)
symHKL = symmetrise(crystalSym, hkls)

""" search for unique hkls to save time """

hkls_loop, uni_hkls_idx, hkls_loop_idx = np.unique(hkls,axis=0,return_inverse=True,return_index=True)
symHKL_loop = symmetrise(crystalSym, hkls_loop)
symHKL_loop = normalize(symHKL_loop)

symOps = genSymOps(crystalSym)
symOps = np.unique(np.swapaxes(symOps,2,0),axis=0)

""" only use proper rotations """
""" complicated, simplify? """

proper = np.where( np.linalg.det(symOps) == 1 ) #proper orthogonal
quatSymOps = quat.from_matrix(symOps[proper])
quatSymOps = np.tile(quatSymOps[:,:,np.newaxis],(1,1,len(phi)))
quatSymOps = quatSymOps.transpose((2,0,1))

""" gen quats from bunge grid """

bungeAngs = np.zeros(( np.product(od.phi1cen.shape), 3 ))

for ii,i in enumerate(np.ndindex(od.phi1cen.shape)):
    
    bungeAngs[ii,:] = np.array((od.phi1cen[i],od.Phicen[i],od.phi2cen[i]))

qgrid = eu2quat(bungeAngs).T

""" refl weights """

def_al = {'name': 'Al',
          'composition': [dict(ion='Al', pos=[0, 0, 0]),
                          dict(ion='Al', pos=[0.5, 0, 0.5]),
                          dict(ion='Al', pos=[0.5, 0.5, 0]),
                          dict(ion='Al', pos=[0, 0.5, 0.5])],
          'lattice': dict(abc=[4.0465, 4.0465, 4.0465], abg=[90, 90, 90]),
          'debye-waller': False,
          'massNorm': False}

refl_wgt = calc_NDreflWeights(def_al, refls)

# """ ones for refl_wgt """

# refl_wgt = {}

# for hi,h in enumerate(hkls):
#     refl_wgt[hi] = 1

# hkl_str = [''.join(tuple(map(str,h))) for h in hkls]

""" calculate pf grid XYZ for fibre """

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
euc_rad = 4*np.sin(theta)**2

fibre_marc = {}
    
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
                
                q0[fi][yi][hi] = quat.normalize(np.hstack( [ np.cos(omega[fi][yi][hi]/2), np.sin(omega[fi][yi][hi]/2) * HxY ] ))
                q[fi][yi][hi]  = quat.normalize(np.hstack( [ cphi[:, np.newaxis], np.tile( y, (len(cphi),1) ) * sphi[:, np.newaxis] ] ))
                
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
            
            fibre_e[fi][yi] = eu_fib
    
            fib_idx = np.unravel_index(fz_idx[0], (qfib.shape[0],qfib.shape[1]))
            
            fibre_q[fi][yi] = qfib
            
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
            
            # egrid_trun[fi][yi] = bungeAngs[query_uni]
            
    return nn_gridPts, nn_gridDist, fibre_e, axis, omega

nn_gridPts, nn_gridDist, fibre_e, axis, omega = calcFibre(pf.symHKL,pf.y,qgrid,phi,rad,tree,euc_rad)
tempPts_full, tempDist_full, tempFibre_e, dump, dump2  = calcFibre(symHKL_loop,xyz_pf,qgrid,phi,rad,tree,euc_rad)

for i,hi in enumerate(hkls_loop_idx):

    nn_gridPts_full[i] = tempPts_full[hi]
    nn_gridDist_full[i] = tempDist_full[hi]
    fibre_full_e[i] = tempFibre_e[hi]
    
# %%        

""" extract distances -> weights, construct pointer 
    pf_od: pf --> pf to od
    od_pf: od --> od to pf
    each entry stored as dict; ['cell'] is cell #s ['weights'] is weights """
    
tube_exp = 1

pf_od = {}
pf_od_full = {}
odwgts_tot = np.zeros( ( len(hkls), od.bungeList.shape[0]*od.bungeList.shape[1]*od.bungeList.shape[2] ) )

test = []

for hi, h in enumerate(hkls):
    
    pf_od[hi] = {}
    pf_od_full[hi] = {}
    
    for yi in range(len(nn_gridPts[hi].keys())):
        
        od_cells = nn_gridPts[hi][yi]

        #handle no od_cells
        if len(od_cells) == 0: continue
        else:

            scaled_dist = nn_gridDist[hi][yi]
            weights = 1 / ( ( abs(scaled_dist) )**tube_exp )
            
            if np.any(weights < 0): raise ValueError('neg weight')
            if np.any(weights == 0): raise ValueError('zero weight')
            
            pf_od[hi][yi] = {'cell': od_cells, 'weight': weights}
            
            odwgts_tot[hi,od_cells.astype(int)] += weights
        
    for yi in range(len(nn_gridPts_full[hi].keys())):
        
        od_cells = nn_gridPts_full[hi][yi]

        #handle no od_cells
        if len(od_cells) == 0: continue
        else:

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
rel_err = {}

eps = 2
recalc_pf_full = {}
    
numPoles = pf._numHKL
numHKLs = [len(fam) for fam in pf.symHKL]

iterations = 9

for i in tqdm(range(iterations),position=0):
    
    """ first iteration, skip recalc of PF """
    
    if i == 0: #first iteration is direct from PFs
        
        od_data = np.ones( od.bungeList.shape[0]*od.bungeList.shape[1]*od.bungeList.shape[2] )
        calc_od[0] = np.ones( (od_data.shape[0], numPoles) )                                
        
        for fi in range(numPoles): 
            
            temp = np.ones(( od.bungeList.shape[0]*od.bungeList.shape[1]*od.bungeList.shape[2], len(pf.y[fi]) ))
            
            for yi in range(len(pf.y[fi])):

                #check for zero OD cells that correspond to the specified pole figure direction 
                # if len(nn_gridPts[fi][yi]) > 0:
                if yi in pf_od[fi]:
                    
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
   
    # """ recalculate full pole figures """
    # recalc_pf_full[i] = np.zeros((pf_grid.shape[0],pf_grid.shape[1],numPoles))
    
    # for fi in range(numPoles):
        
    #     for pf_cell in np.ravel(pf_grid):
            
    #         if pf_cell in pf_od_full[fi]: #pf_cell is defined
                
    #             od_cells = np.array(pf_od_full[fi][pf_cell]['cell'])
    #             ai, bi = np.divmod(pf_cell, pf_grid.shape[1])
    #             recalc_pf_full[i][int(ai),int(bi),fi] = ( 1 / np.sum(pf_od_full[fi][pf_cell]['weight']) ) * np.sum( pf_od_full[fi][pf_cell]['weight'] * calc_od[i].weights[od_cells.astype(int)] )
        
    # recalc_pf_full[i] = poleFigure(recalc_pf_full[i], pf.hkl, od.cs, 'recalc', resolution=5)
    # recalc_pf_full[i].normalize()

    """ compare recalculated to experimental """
        
    RP_err = {}
    prnt_str = None
    
    np.seterr(divide='ignore')

    for fi in range(3):
        
        RP_err[fi] = np.abs( recalc_pf[i][fi] - pf.data[fi] ) / recalc_pf[i][fi]
        RP_err[fi][np.isinf(RP_err[fi])] = 0
        RP_err[fi] = np.sqrt(np.mean(RP_err[fi]**2))
        
        if prnt_str is None: prnt_str = 'RP Error: {:.4f}'.format(np.round(RP_err[fi],decimals=4))
        else: prnt_str += ' | {:.4f}'.format(np.round(RP_err[fi],decimals=4))
        
    tqdm.write(prnt_str)

    """ recalculate full pole figures """
    recalc_pf_full[i] = {}
    
    for fi in range(numPoles):
        
        recalc_pf_full[i][fi] = np.zeros(len(xyz_pf))

        for yi in range(len(xyz_pf)):
            
            if yi in pf_od_full[fi]: #pf_cell is defined
                
                od_cells = np.array(pf_od_full[fi][yi]['cell'])

                recalc_pf_full[i][fi][yi] = ( 1 / np.sum(pf_od_full[fi][yi]['weight']) ) * np.sum( pf_od_full[fi][yi]['weight'] * calc_od[i].weights[od_cells.astype(int)] )
        
    recalc_pf_full[i] = poleFigure(recalc_pf_full[i], pf.hkl, od.cs, 'recalc', resolution=5, arb_y=xyz_pf)
    recalc_pf_full[i].normalize()    
        
    """ (i+1)th inversion """

    od_data = np.ones( od.bungeList.shape[0]*od.bungeList.shape[1]*od.bungeList.shape[2] )
    calc_od[i+1] = np.zeros( (od_data.shape[0], numPoles) )        
    
    for fi in range(numPoles):

        temp = np.ones(( od.bungeList.shape[0]*od.bungeList.shape[1]*od.bungeList.shape[2], len(pf.y[fi]) ))
        
        for yi in range(len(pf.y[fi])):
            
            #check for zero OD cells that correspond to the specified pole figure direction 
            # if len(nn_gridPts_full[fi][yi]) > 0:     
            if yi in pf_od[fi]:              
            
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
     
cl = np.arange(0,7.5,0.5)       
recalc_pf_full[iterations-1].plot(pfs=3,contourlevels=cl,cmap='magma_r',proj='none')
# calc_od[iterations-1].sectionPlot('phi2',np.deg2rad(90))
print(calc_od[iterations-1].index())
# calc_od[iterations-1].export('/mnt/c/Users/Nate/Dropbox/ORNL/EWIMVvsMTEX/EWIMV exports/'+sampleName+'.odf')
# recalc_pf_full[iterations-1].export('/mnt/c/Users/Nate/Dropbox/ORNL/EWIMVvsMTEX/EWIMV exports',sampleName=sampleName)


# %%

# test_wgts = calc_od[iterations-1].weights
# test_wgts = test_wgts.reshape(calc_od[iterations-1].Phicen.shape)
# test_wgts = np.tile(test_wgts,(1,1,4))
# od_noSS = bunge(od.res,'m-3m','1',weights=np.ravel(test_wgts))

# """ recalculate full pole figures """
# recalc_pf_test = {}

# for fi in range(numPoles):
    
#     recalc_pf_test[fi] = np.zeros(len(xyz_pf))

#     for yi in range(len(xyz_pf)):
        
#         if yi in pf_od_full[fi]: #pf_cell is defined
            
#             od_cells = np.array(pf_od_full[fi][yi]['cell'])

#             recalc_pf_test[fi][yi] = ( 1 / np.sum(pf_od_full[fi][yi]['weight']) ) * np.sum( pf_od_full[fi][yi]['weight'] * od_noSS.weights[od_cells.astype(int)] )
    
# recalc_pf_test = poleFigure(recalc_pf_test, pf.hkl, od.cs, 'recalc', resolution=5, arb_y=xyz_pf)
# recalc_pf_test.normalize()  
# recalc_pf_test.plot(pfs=3,contourlevels=cl,cmap='magma',proj='none')

# %%

### 3D ODF plot ###

import mayavi.mlab as mlab
from tvtk.util import ctf
from matplotlib.pyplot import cm

mlab.figure(figure='1',bgcolor=(0.75,0.75,0.75))

#reshape pts
data = calc_od[iterations-1].weights.reshape(calc_od[iterations-1].phi1cen.shape)
#round small values (<1E-5)
data[data < 1E-5] = 0

# calc_od[iterations-1].phi1cen,calc_od[iterations-1].Phicen,calc_od[iterations-1].phi2cen,

#needs work
# vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(data), vmin=0, vmax=0.8)
# vol.volume_mapper_type = 'FixedPointVolumeRayCastMapper'

cont = mlab.pipeline.contour_surface(mlab.pipeline.scalar_field(data),
                                     contours=list(np.linspace(0,np.max(data),30)),
                                     transparent=True)

ax = mlab.axes(color=(0,0,0),
               xlabel='phi2',
               ylabel='Phi',
               zlabel='phi1',
               ranges=[0, np.rad2deg(calc_od[iterations-1]._phi2max),
                       0, np.rad2deg(calc_od[iterations-1]._Phimax),
                       0, np.rad2deg(calc_od[iterations-1]._phi1max)])  

ax.label_text_property.font_family = 'arial'
ax.label_text_property.font_size = 1
ax.title_text_property.font_size = 1


cbar = mlab.scalarbar(cont)
cbar.shadow = True
cbar.number_of_labels = 10
#adjust label position
cbar.label_text_property.justification = 'centered'
cbar.label_text_property.font_family = 'arial'
cbar.scalar_bar.text_pad = 10
cbar.scalar_bar.unconstrained_font_size = False
cbar.label_text_property.italic = False
cbar.label_text_property.font_size = 20
#turn off parallel projection
mlab.gcf().scene.parallel_projection = False

#setup correct view
mlab.view(azimuth=50,elevation=None)
mlab.show(stop=True)


# %%

### FIBER PLOT ###

# """ import matlab """

# from scipy.io import loadmat

# mtex_fib = loadmat('/home/nate/Dropbox/ORNL/Texture/NRSF2/bunList.mat')['bun_list']

# pf_num = 0

# import mayavi.mlab as mlab
# # import matplotlib.pyplot as plt
# # fig = plt.figure()

# mlab.figure(bgcolor=(1,1,1))

# ## grid ##
# gd = mlab.points3d(bungeAngs[:,0],bungeAngs[:,1],bungeAngs[:,2],scale_factor=1,mode='point',color=(0,0,0))
# gd.actor.property.render_points_as_spheres = True
# gd.actor.property.point_size = 3
  
# # ## lit point ##
# # gd2 = mlab.points3d(0,0,0,scale_factor=1,mode='point',color=(1,0,0))
# # gd2.actor.property.render_points_as_spheres = True
# # gd2.actor.property.point_size = 5

# ## manual fibre ##
# gd2 = mlab.points3d(fibre_e[pf_num][0][:,0],
#                     fibre_e[pf_num][0][:,1],
#                     fibre_e[pf_num][0][:,2],
#                     scale_factor=1,
#                     mode='point',
#                     color=(0,1,0))

# gd2.actor.property.render_points_as_spheres = True
# gd2.actor.property.point_size = 5


# ## mtex fibre ##
# gd3 = mlab.points3d(mtex_fib[:,0],
#                     mtex_fib[:,1],
#                     mtex_fib[:,2],
#                     scale_factor=1,
#                     mode='point',
#                     color=(0,0,1))

# gd3.actor.property.render_points_as_spheres = True
# gd3.actor.property.point_size = 5

# # ## trun grid ##
# # gd3 = mlab.points3d(0,0,0,scale_factor=1,mode='point',color=(0,0,1))
# # gd3.actor.property.render_points_as_spheres = True
# # gd3.actor.property.point_size = 5   

# # plt_list = list(fibre_full_e[pf_num].keys())
# # plt_list.sort()

# # """ cube (111) pts """

# # from scipy.spatial.distance import cdist

# # azi = np.deg2rad(np.array((45,135,225,315)))
# # pol = np.deg2rad(np.array((35,35,35,35)))
# # x = np.sin(pol) * np.cos(azi)
# # y = np.sin(pol) * np.sin(azi)
# # z = np.cos(pol)
# # pts = np.array((x,y,z)).T

# # dist_mat = cdist(xyz_pf,pts)
# # plt_list = np.argmin(dist_mat,axis=0)

# # @mlab.animate(delay=100)
# # def anim():
# #     while True:
        
# #         for yi in plt_list:
                
# # #            gd2.mlab_source.reset( x = fibre_wimv[pf_num][yi][:,0],
# # #                                   y = fibre_wimv[pf_num][yi][:,1],
# # #                                   z = fibre_wimv[pf_num][yi][:,2])
            
# #             gd2.mlab_source.reset( x = fibre_full_e[pf_num][yi][:,0],
# #                                   y = fibre_full_e[pf_num][yi][:,1],
# #                                   z = fibre_full_e[pf_num][yi][:,2])
            
# #             # gd2.mlab_source.reset( x = egrid_trun[pf_num][yi][:,0],
# #             #                         y = egrid_trun[pf_num][yi][:,1],
# #             #                         z = egrid_trun[pf_num][yi][:,2])
            
# #             tubePts = nn_gridPts_full[pf_num][yi]
            
# #             gd3.mlab_source.reset( x = bungeAngs[tubePts.astype(int),0],
# #                                     y = bungeAngs[tubePts.astype(int),1],
# #                                     z = bungeAngs[tubePts.astype(int),2])
        
# #             yield
            
# # anim()

# #for yi in range(len(pf.y[pf_num])):
# #    
# #        gd = mlab.points3d(fibre[pf_num][yi][:,0],fibre[pf_num][yi][:,1],fibre[pf_num][yi][:,2],scale_factor=1,mode='point',color=(1,0,0))
# #        gd.actor.property.render_points_as_spheres = True
# #        gd.actor.property.point_size = 5    

# mlab.show(stop=True)

# %%

        # q1_n = [quat.from_axis_angle(h, omega) for h in fam]
    
        # for yi,y in enumerate(it):
            
        #     axis = np.cross(fam,y)
        #     angle = np.arccos(np.dot(fam,y))
            
        #     q0_n = quat.from_axis_angle(axis, angle)
        #     # q0_n = quat.normalize(q0)
            
        #     qfib = np.zeros((len(q1_n[0]),len(q0_n),4))
            
        #     for sym_eq,(qA,qB) in enumerate(zip(q0_n,q1_n)):
                
        #         temp = quat.multiply(qA, qB)
                
        #         qfib[:,sym_eq,:] = temp