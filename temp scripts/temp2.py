#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:05:59 2019

@author: nate
"""

import sys

from tqdm import tqdm

import numpy as np
#import pandas as pd
#from tqdm import tqdm

sys.path.insert(0,'/home/nate/wimv')

from classes import poleFigure, bunge
from utils.orientation import symmetrise, normalize, XYZtoSPH

crystalSym = 'm-3m'
sampleSym = '1'
cellSize = np.deg2rad(5)
hkls = [(1,1,1),(2,0,0),(2,2,0)]

bkgd111path = '/home/nate/wimv/Data/32.5_ bkgd.xrdml'
bkgd200path = '/home/nate/wimv/Data/55_bkgd.xrdml'
bkgd220path = '/home/nate/wimv/Data/55_bkgd.xrdml'

bkgds = [bkgd111path,bkgd200path,bkgd220path]
bkgd = poleFigure(bkgds, hkls, crystalSym, 'xrdml',subtype='bkgd')

def111path = '/home/nate/wimv/Data/defocus_38.xrdml'
def200path = '/home/nate/wimv/Data/defocus_45.xrdml'
def220path = '/home/nate/wimv/Data/defocus_65.xrdml'

defs = [def111path,def200path,def220path]
defocus = poleFigure(defs, hkls, crystalSym, 'xrdml',subtype='defocus')

pf111path = '/home/nate/wimv/Data/111pf_2T=38.xrdml'
pf200path = '/home/nate/wimv/Data/200pf_2T=45.xrdml'
pf220path = '/home/nate/wimv/Data/220pf_2theta=65.xrdml'

#bkgd111path = '/home/nate/wimv/Data/Bkgd_42_Al_sample_MC14.xrdml'
#bkgd200path = '/home/nate/wimv/Data/Bkgd_42_Al_sample_MC14.xrdml'
#bkgd220path = '/home/nate/wimv/Data/Bkgd_60_Al_sample_MC14.xrdml'
#
#bkgds = [bkgd111path,bkgd200path,bkgd220path]
#bkgd = poleFigure(bkgds, hkls, crystalSym, 'xrdml',subtype='bkgd')
#
#def111path = '/home/nate/wimv/Data/defocus_38_70Chi.xrdml'
#def200path = '/home/nate/wimv/Data/defocus_45_70Chi.xrdml'
#def220path = '/home/nate/wimv/Data/defocus_65_70Chi.xrdml'
#
#defs = [def111path,def200path,def220path]
#defocus = poleFigure(defs, hkls, crystalSym, 'xrdml',subtype='defocus')
#
#pf111path = '/home/nate/wimv/Data/111_pf_Al_sample_MC14.xrdml'
#pf200path = '/home/nate/wimv/Data/200_pf_Al_sample_MC14.xrdml'
#pf220path = '/home/nate/wimv/Data/220_pf_Al_sample_MC14.xrdml'

#bkgd111path = '/home/nate/wimv/ARL/Microbeam Collimator/Ti18_asrec/Bkgd_45_Ti18_asrec.xrdml'
#bkgd200path = '/home/nate/wimv/ARL/Microbeam Collimator/Ti18_asrec/Bkgd_60_Ti18_asrec.xrdml'
#bkgd220path = '/home/nate/wimv/ARL/Microbeam Collimator/Ti18_asrec/Bkgd_80_Ti18_asrec.xrdml'
#
#bkgds = [bkgd111path,bkgd200path,bkgd220path]
#bkgd = poleFigure(bkgds, hkls, crystalSym,'xrdml',subtype='bkgd')
#
#def111path = '/home/nate/wimv/ARL/Microbeam Collimator/Defocus_111_Si.xrdml'
#def200path = '/home/nate/wimv/ARL/Microbeam Collimator/Defocus_220_Si.xrdml'
#def220path = '/home/nate/wimv/ARL/Microbeam Collimator/Defocus_311_Si.xrdml'
#
#defs = [def111path,def200path,def220path]
#defocus = poleFigure(defs, hkls, crystalSym, 'xrdml',subtype='defocus')
#
#pf111path = '/home/nate/wimv/ARL/Microbeam Collimator/Ti18_asrec/beta_110_Ti18_asrec.xrdml'
#pf200path = '/home/nate/wimv/ARL/Microbeam Collimator/Ti18_asrec/beta_200_Ti18_asrec.xrdml'
#pf220path = '/home/nate/wimv/ARL/Microbeam Collimator/Ti18_asrec/beta_211_Ti18_asrec.xrdml'

pfs = [pf111path,pf200path,pf220path]
pf = poleFigure(pfs, hkls, crystalSym, 'xrdml')

pf.correct(bkgd=bkgd,defocus=defocus)
pf.normalize()

od = bunge(cellSize, crystalSym, sampleSym)

hkls = np.array(hkls)

symHKL = symmetrise(crystalSym, hkls)
symHKL = normalize(symHKL)

pf_grid, alp, bet = pf.grid(full=True, ret_ab=True)

xyz = {}   
pol = {}

a_bins = np.histogram_bin_edges(np.arange(0,np.pi/2+pf.res,pf.res),18)
b_bins = np.histogram_bin_edges(np.arange(0,2*np.pi+pf.res,pf.res),72)

#dict | key:od cell# value:pf cell#s
od_pf = {}

#dict | key:pf_cell# value:od cell#s
pf_od = {}

#test = []

for fi,fam in tqdm(enumerate(symHKL)):
    
    # Mx3xN array | M - hkl multi. N - # of unique g
    xyz[fi] = np.dot(fam,od.g)

    # stereographic projection
    pol[fi] = XYZtoSPH(xyz[fi])
    
    od_pf[fi] = {}
    pf_od[fi] = {}
    
    for od_cell in np.ravel(od.bungeList):
        
        """
        use histogram_bin_edges for bins
        then searchsorted to find indices
        """
        
        ai = np.searchsorted(a_bins, pol[fi][:,1,int(od_cell)], side='left')
        bi = np.searchsorted(b_bins, pol[fi][:,0,int(od_cell)], side='left')
        
        #value directly at 2pi
        bi = np.where(bi==72, 0, bi)
    
        pfi = pf_grid[ai.astype(int),bi.astype(int)] #pole figure index
        
        od_pf[fi][od_cell] = pfi
            
        for p in pfi:
               
            try:
                pf_od[fi][p].append(od_cell)
                
            except:
                pf_od[fi][p] = []
                pf_od[fi][p].append(od_cell)
            
# %%
                
calc_od = {}
recalc_pf = {}
test = {}
                
iterations = range(20)

for i in iterations:
    
    """ first iteration, skip recalc of PF """
    
    if i == 0: #first iteration is direct from PFs
        
        od_data = np.ones( od.bungeList.shape[0]*od.bungeList.shape[1]*od.bungeList.shape[2] )
        calc_od[0] = np.zeros( (od_data.shape[0], 3) )        
        
        for fi in range(len(symHKL)): 
                
            for pf_cell in np.ravel(pf_grid):
                
                try:
                    od_cells = np.array(pf_od[fi][pf_cell])
                    ai, bi = np.divmod(pf_cell, pf_grid.shape[1])
                except:
                    continue
                
                if pf_cell < pf.data[fi].shape[0]*pf.data[fi].shape[1]: #inside of measured PF range
                    
                    od_data[od_cells.astype(int)] *= pf.data[fi][int(ai),int(bi)]
            
#            for od_cell in np.ravel(od.bungeList):
#                
#                pf_cells = od_pf[fi][od_cell]
#                
#                pf_cellMax = pf.data[fi].shape[0]*pf.data[fi].shape[1]
#                pf_cells = pf_cells[pf_cells < pf_cellMax]
#                
#                ai, bi = np.divmod(pf_cells, pf_grid.shape[1])
#                od_data[int(od_cell)] = np.product( pf.data[fi][ai.astype(int),bi.astype(int)] )            
                        
            calc_od[0][:,fi] = np.power(od_data,(1/len(symHKL[fi])))
            
        calc_od[0] = np.product(calc_od[0],axis=1)**(1/len(symHKL))
        #place into OD object
        calc_od[0] = bunge(cellSize, crystalSym, sampleSym, weights=calc_od[0])
        calc_od[0].normalize()

    ### recalculate pole figure ###
    recalc_pf[i] = np.zeros((pf_grid.shape[0],pf_grid.shape[1],len(symHKL)))
    
    for fi in range(len(symHKL)):
        
        for pf_cell in np.ravel(pf_grid):
            
            if pf_cell in pf_od[fi]: #pf_cell is defined
                
                od_cells = np.array(pf_od[fi][pf_cell])
                ai, bi = np.divmod(pf_cell, pf_grid.shape[1])
                recalc_pf[i][int(ai),int(bi),fi] = ( 1 / len(od_cells) ) * np.sum( calc_od[i].weights[od_cells.astype(int)] )
        
            #temp test
#            if recalc_pf[i][int(ai),int(bi),fi] == 0: recalc_pf[i][int(ai),int(bi),fi] = 0.0005
        
    recalc_pf[i] = poleFigure(recalc_pf[i], hkls, crystalSym, 'recalc', resolution=5)
    recalc_pf[i].normalize()
    
    ### compare recalculated to experimental ###
    
    RMS_err = {}
    prnt_str = None
    
    for fi in range(len(symHKL)):
        
        expLim = pf.data[fi].shape
        RMS_err[fi] = ( recalc_pf[i].data[fi][:expLim[0],:expLim[1]] - pf.data[fi] )**2
        RMS_err[fi] = np.sqrt( np.sum(RMS_err[fi]) / len(RMS_err[fi]) )
        
        if prnt_str is None: prnt_str = 'RMS Error: {}'.format(round(RMS_err[fi],ndigits=3))
        else: prnt_str += ' | {}'.format(round(RMS_err[fi],ndigits=3))
        
    print(prnt_str)
        
    ### (i+1)th inversion ###
    od_data = np.ones( od.bungeList.shape[0]*od.bungeList.shape[1]*od.bungeList.shape[2] )
    calc_od[i+1] = np.zeros( (od_data.shape[0], 3) )        
    
    for fi in range(len(symHKL)):
            
        for pf_cell in np.ravel(pf_grid):
            
            try:
                od_cells = np.array(pf_od[fi][pf_cell])
                ai, bi = np.divmod(pf_cell, pf_grid.shape[1])
            except:
                continue
            
            if pf_cell < pf.data[fi].shape[0]*pf.data[fi].shape[1]: #inside of measured PF range
                
                if recalc_pf[i].data[fi][int(ai),int(bi)] == 0: continue
                else: od_data[od_cells.astype(int)] *= ( pf.data[fi][int(ai),int(bi)] / recalc_pf[i].data[fi][int(ai),int(bi)] )
        
#        for od_cell in tqdm(np.ravel(od.bungeList)):
#            
#            pf_cells = od_pf[fi][od_cell]
#            
#            pf_cellMax = pf.data[fi].shape[0]*pf.data[fi].shape[1]
#            pf_cells = pf_cells[pf_cells < pf_cellMax]
#            
#            ai, bi = np.divmod(pf_cells, pf_grid.shape[1])
#            od_data[int(od_cell)] = np.product( pf.data[fi][ai.astype(int),bi.astype(int)] / recalc_pf[i].data[fi][ai.astype(int), bi.astype(int)] )
                
        calc_od[i+1][:,fi] = np.power(od_data,(1/len(symHKL[fi])))
    
    test[i+1] = np.power(np.product(calc_od[i+1],axis=1),(1/len(symHKL)))
    calc_od[i+1] = calc_od[i].weights * np.power(np.product(calc_od[i+1],axis=1),(1/len(symHKL)))
    
    #place into OD object
    calc_od[i+1] = bunge(cellSize, crystalSym, sampleSym, weights=calc_od[i+1])
    calc_od[i+1].normalize()    
        
        
# %%
        
#recalc_pf = poleFigure(recalc_data,hkls,'recalc',resolution=5)
#recalc_pf.normalize()

#recalc_pf.plot()


# %%

#test = copy.deepcopy(xyz[0])
#
#sep = np.linspace(0,test.shape[2]*test.shape[0],test.shape[2])
#
##new = np.zeros((test.shape[2]*test.shape[0],test.shape[1]))
#new = []
#
#for i,s in enumerate(sep):
#    
#    if s == 0: continue
#    else:
##        new[sep[i-1]:s-1,:] = test[:,:,i]
#        new.append((sep[i-1],s-1))
        
        
# %%

""" old code snippets """

#        #normalize to volume
#        od_idx = np.unravel_index(np.ravel(od.bungeList).astype(int),od.bungeList.shape)
#            
#        #calculate cell volume for distortion
#        Phi = od.Phi[od_idx]
#        od_cellVol = od.res*od.res*( np.cos( Phi - (od.res/2) ) - np.cos( Phi + (od.res/2) ) )
#        
#        # f(g)dg
#        calc_od_dg = calc_od * od_cellVol 
#        
#        # N
#        od_norm = sum(od_cellVol) / sum(calc_od_dg)
#        
#        calc_od_N = od_norm * calc_od

# %%

#pf_num = 1
#
#import matplotlib.pyplot as plt
#fig = plt.figure()
#
# ## PF Cell Loop ###
#
#for od_cell in np.ravel(od.bungeList):
#    
#    # binned cell
#    pf_cells = od_pf[pf_num][od_cell]
#    ai, bi = np.divmod(pf_cells, pf_grid.shape[1])    
#    pltcells = np.zeros_like(pf_grid)
#    pltcells[ai.astype(int),bi.astype(int)] += 1
#    
#    pltcells = pltcells.T
#    theta, r = np.mgrid[0:2*np.pi:73j, 0:np.pi/2:19j]
#    
#    ax = fig.add_subplot(111,projection='polar')
#    ax.set_axisbelow(True)
#     #r
#    ax.set_ylim([0,np.pi/2+0.05])
#    ax.set_yticks(np.linspace(0,np.pi/2,19))
#    ax.set_yticklabels([])
#     #theta
#    ax.set_xticks(np.linspace(0,2*np.pi,73))
#    ax.set_xticklabels([])
#    
#    ax.pcolormesh(theta,r,pltcells,cmap='Reds')
#    ax.scatter(pol[pf_num][:,0,int(od_cell)],pol[pf_num][:,1,int(od_cell)],s=10,c='b')
#    plt.pause(0.05)
#    plt.cla()                

# %%
        
# pole figure normalization
#dg = pf.res * ( np.cos( alp - (pf.res/2) ) - np.cos( alp + (pf.res/2) ) )
#
#recalc_pf_dg = np.zeros_like(recalc_data)
#pg_n = np.zeros_like(recalc_data)
#
#for fi in range(len(symHKL)):
#    
#    pg_dg = recalc_data[:,:,fi] * dg
#    
#    norm = np.sum( np.ravel(pg_dg) ) / np.sum( np.ravel(dg) )
#    
#    pg_n[:,:,fi] = ( 1 / norm ) * recalc_data[:,:,fi] 
###
###    recalc_pf_dg[:,:,fi] = recalc_data[:,:,fi] * pf_cellVol  
###
###    temp = np.ravel(recalc_data[:,:,fi])
###    pf_norm = ( sum(np.ravel(pf_cellVol)) / sum(np.ravel(recalc_pf_dg)) )
###    
###    print(pf_norm)
###    
###    recalc_pf_N[:,:,fi] = pf_norm * recalc_data[:,:,fi]     
##        
#pf2 = poleFigure(pg_n,hkls,'recalc',resolution=5)
