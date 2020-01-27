#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:16:57 2019

@author: nate
"""

import sys,os
import numpy as np

# from tqdm import tqdm as _tqdm

filepath = os.path.dirname(os.path.abspath(__file__))

from pyTex import poleFigure as _poleFigure
from pyTex import bunge as _bunge
from pyTex.inversion import wimv
from pyTex.inversion import _wimv_test

from pyTex.utils import XYZtoSPH as _XYZtoSPH
from pyTex.orientation import quat2eu as _quat2eu

crystalSym = 'm-3m'
sampleSym = '1'
cellSize = np.deg2rad(5)
hkls = [(1,1,1),(2,0,0),(2,2,0)]

bkgd111path = os.path.join(filepath,'Data','XRD Aluminum','32.5_ bkgd.xrdml')
bkgd200path = os.path.join(filepath,'Data','XRD Aluminum','55_bkgd.xrdml')
bkgd220path = os.path.join(filepath,'Data','XRD Aluminum','55_bkgd.xrdml')

bkgds = [bkgd111path,bkgd200path,bkgd220path]
bkgd = _poleFigure(bkgds, hkls, crystalSym, 'xrdml',subtype='bkgd')

def111path = os.path.join(filepath,'Data','XRD Aluminum','defocus_38.xrdml')
def200path = os.path.join(filepath,'Data','XRD Aluminum','defocus_45.xrdml')
def220path = os.path.join(filepath,'Data','XRD Aluminum','defocus_65.xrdml')

defs = [def111path,def200path,def220path]
defocus = _poleFigure(defs, hkls, crystalSym, 'xrdml',subtype='defocus')

pf111path = os.path.join(filepath,'Data','XRD Aluminum','111pf_2T=38.xrdml')
pf200path = os.path.join(filepath,'Data','XRD Aluminum','200pf_2T=45.xrdml')
pf220path = os.path.join(filepath,'Data','XRD Aluminum','220pf_2theta=65.xrdml')

pfs = [pf111path,pf200path,pf220path]
pfs = _poleFigure(pfs, hkls, crystalSym, 'xrdml')

pfs.correct(bkgd=bkgd,defocus=defocus)
pfs.normalize()

orient_dist = _bunge(cellSize, crystalSym, sampleSym)

# pf_grid = pf.grid(full=True)

# recalc_pf, calc_od, pf_od, od_pf, prnt_str = wimv(pfs, orient_dist, ret_pointer=True )
recalc_pf, calc_od = _wimv_test(pfs, orient_dist)

cl = np.arange(0,9.5,0.5)
recalc_pf[11].plot(contourlevels=cl)

# %%

# """
# perform WIMV inversion
# fixed grid in PF space requiredpointer

# input:
#     exp_pfs    : poleFigure object
#     orient_dist: orientDist object
#     iterations : number of iterations
# """

# fullPFgrid, alp, bet = pfs.grid(pfs.res,
#                              radians=True,
#                              cen=True,
#                              ret_ab=True)

# xyz = {}   
# sph = {}

# a_bins = np.histogram_bin_edges(np.arange(0,np.pi/2+pfs.res,pfs.res),18)
# b_bins = np.histogram_bin_edges(np.arange(0,2*np.pi+pfs.res,pfs.res),72)

# #dict | key:od cell# value:pf cell#s
# od_pf = {}

# #dict | key:pf_cell# value:od cell#s
# pf_od = {}

# numPoles = pfs._numHKL
# numHKLs = [len(fam) for fam in pfs.symHKL]

# """ generate pointer matrix od->pf / pf->od """

# for fi,fam in enumerate(pfs.symHKL):
    
#     # Mx3xN array | M - hkl multi. N - # of unique g
#     xyz[fi] = np.dot(fam,orient_dist.g)

#     sph[fi] = _XYZtoSPH(xyz[fi],proj='none')
    
#     od_pf[fi] = {}
#     pf_od[fi] = {}
    
#     for od_cell in np.ravel(orient_dist.bungeList):
        
#         ai = np.searchsorted(a_bins, sph[fi][:,1,int(od_cell)], side='left')
#         bi = np.searchsorted(b_bins, sph[fi][:,0,int(od_cell)], side='left')
        
#         #value directly at pi/2
#         ai = np.where(ai==18, 17, ai)
        
#         #value directly at 2pi
#         bi = np.where(bi==72, 0, bi)
    
#         pfi = fullPFgrid[ai.astype(int),bi.astype(int)] #pole figure index
        
#         od_pf[fi][od_cell] = pfi
            
#         for p in pfi:
               
#             try:
#                 pf_od[fi][p].append(od_cell)
                
#             except:
#                 pf_od[fi][p] = []
#                 pf_od[fi][p].append(od_cell)

# """ done with pointer generation """

# od_data = np.ones( orient_dist.bungeList.shape[0]*orient_dist.bungeList.shape[1]*orient_dist.bungeList.shape[2] )
# calc_od = {}
# recalc_pf = {}

# for i in _tqdm(range(iterations),position=0):
    
#     """ first iteration, skip recalc of PF """
    
#     if i == 0: #first iteration is direct from PFs
        
#         od_data = np.ones( orient_dist.bungeList.shape[0]*orient_dist.bungeList.shape[1]*orient_dist.bungeList.shape[2] )
#         calc_od[0] = np.zeros( (od_data.shape[0], numPoles) )        
        
#         for fi in range(numPoles): 
                
#             for pf_cell in np.ravel(fullPFgrid):
                
#                 try:
#                     od_cells = np.array(pf_od[fi][pf_cell])
#                     ai, bi = np.divmod(pf_cell, fullPFgrid.shape[1])
#                 except:
#                     continue
                
#                 if pf_cell < pfs.data[fi].shape[0]*pfs.data[fi].shape[1]: #inside of measured PF range
                    
#                     od_data[od_cells.astype(int)] *= pfs.data[fi][int(ai),int(bi)]
            
#             """ loop over od_cells (alternative) """
#         #    for od_cell in np.ravel(orient_dist.bungeList):
               
#         #        pf_cells = od_pf[fi][od_cell]
               
#         #        pf_cellMax = pf.data[fi].shape[0]*pf.data[fi].shape[1]
#         #        pf_cells = pf_cells[pf_cells < pf_cellMax]
               
#         #        ai, bi = np.divmod(pf_cells, fullPFgrid.shape[1])
#         #        od_data[int(od_cell)] = np.product( pf.data[fi][ai.astype(int),bi.astype(int)] )            
                        
#             calc_od[0][:,fi] = np.power(od_data,(1/numHKLs[fi]))
#             # calc_od[0][:,fi] = np.power(od_data,1)

            
#         calc_od[0] = np.product(calc_od[0],axis=1)**(1/numPoles)
#         #place into OD object
#         calc_od[0] = _bunge(orient_dist.res, orient_dist.cs, orient_dist.ss, weights=calc_od[0])
#         calc_od[0].normalize()

#     """ recalculate pole figures """
#     recalc_pf[i] = np.zeros((fullPFgrid.shape[0],fullPFgrid.shape[1],numPoles))
    
#     for fi in range(numPoles):
        
#         for pf_cell in np.ravel(fullPFgrid):
            
#             if pf_cell in pf_od[fi]: #pf_cell is defined
                
#                 od_cells = np.array(pf_od[fi][pf_cell])
#                 ai, bi = np.divmod(pf_cell, fullPFgrid.shape[1])
#                 recalc_pf[i][int(ai),int(bi),fi] = ( 1 / len(od_cells) ) * np.sum( calc_od[i].weights[od_cells.astype(int)] )
        
#     recalc_pf[i] = _poleFigure(recalc_pf[i], pfs.hkl, orient_dist.cs, 'recalc', resolution=5)
#     recalc_pf[i].normalize()
    
#     """ compare recalculated to experimental """
        
#     RP_err = {}
#     prnt_str = None
    
#     np.seterr(divide='ignore')

#     for fi in range(numPoles):
        
#         expLim = pfs.data[fi].shape
#         RP_err[fi] = np.abs( recalc_pf[i].data[fi][:expLim[0],:expLim[1]] - pfs.data[fi] ) / recalc_pf[i].data[fi][:expLim[0],:expLim[1]]
#         RP_err[fi][np.isinf(RP_err[fi])] = 0
#         RP_err[fi] = np.sqrt(np.mean(RP_err[fi]**2))
        
#         if prnt_str is None: prnt_str = 'RP Error: {:.4f}'.format(np.round(RP_err[fi],decimals=4))
#         else: prnt_str += ' | {:.4f}'.format(np.round(RP_err[fi],decimals=4))
        
#     _tqdm.write(prnt_str)
        
#     """ (i+1)th inversion """

#     od_data = np.ones( orient_dist.bungeList.shape[0]*orient_dist.bungeList.shape[1]*orient_dist.bungeList.shape[2] )
#     calc_od[i+1] = np.zeros( (od_data.shape[0], numPoles) )        
    
#     for fi in range(numPoles):
            
#         for pf_cell in np.ravel(fullPFgrid):
            
#             if pf_cell in pf_od[fi]:
                
#                 od_cells = np.array(pf_od[fi][pf_cell])
#                 ai, bi = np.divmod(pf_cell, fullPFgrid.shape[1])

#                 if pf_cell < pfs.data[fi].shape[0]*pfs.data[fi].shape[1]: #inside of measured PF range
                    
#                     if recalc_pf[i].data[fi][int(ai),int(bi)] == 0: continue
#                     else: od_data[od_cells.astype(int)] *= ( pfs.data[fi][int(ai),int(bi)] / recalc_pf[i].data[fi][int(ai),int(bi)] )
        
#         """ loop over od_cells (alternative) """
#     #    for od_cell in _tqdm(np.ravel(orient_dist.bungeList)):
           
#     #        pf_cells = od_pf[fi][od_cell]
           
#     #        pf_cellMax = pf.data[fi].shape[0]*pf.data[fi].shape[1]
#     #        pf_cells = pf_cells[pf_cells < pf_cellMax]
           
#     #        ai, bi = np.divmod(pf_cells, fullPFgrid.shape[1])
#     #        od_data[int(od_cell)] = np.product( pf.data[fi][ai.astype(int),bi.astype(int)] / recalc_pf[i].data[fi][ai.astype(int), bi.astype(int)] )
                
#         calc_od[i+1][:,fi] = np.power(od_data,(1/numHKLs[fi]))
    
#     calc_od[i+1] = calc_od[i].weights * np.power(np.product(calc_od[i+1],axis=1),(0.8/numPoles))
    
#     #place into OD object
#     calc_od[i+1] = _bunge(orient_dist.res, orient_dist.cs, orient_dist.ss, weights=calc_od[i+1])
#     calc_od[i+1].normalize()   
    



# %%

#fibre_out = {}
#
#for hi, h in enumerate(hkls):
#    
#    fibre_out[hi] = {}
#    
#    for pf_cell in pf_od[hi].keys():
#        
#        od_cells = np.array(pf_od[hi][pf_cell])
#        od_idx = np.unravel_index(od_cells.astype(int),od.bungeList.shape)
#        
#        temp = np.column_stack((od.phi1cen[od_idx], od.Phicen[od_idx], od.phi2cen[od_idx]))
#        
#        fibre_out[hi][pf_cell] = temp
#        
#import pickle
#f = open("file.pkl","wb")
#pickle.dump(fibre_out,f)
#f.close()


# %%

#""" plotting test """
#
#import mayavi.mlab as mlab
#mlab.figure(bgcolor=(1,1,1))
#
# ## grid cen ##
#gd = mlab.points3d(od.phi1cen,od.Phicen,od.phi2cen,scale_factor=1,mode='point',color=(1,0,0))
#gd.actor.property.render_points_as_spheres = True
#gd.actor.property.point_size = 5
#
# ## grid ##
#gd = mlab.points3d(od.phi1,od.Phi,od.phi2,scale_factor=1,mode='point',color=(0,0,0))
#gd.actor.property.render_points_as_spheres = True
#gd.actor.property.point_size = 3
