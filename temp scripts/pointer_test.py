#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 10:32:25 2019

@author: nate
"""

## analyze pointers



import pickle
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

from classes import poleFigure

pf111path = '/home/nate/wimv/Data/111pf_2T=38.xrdml'
pf200path = '/home/nate/wimv/Data/200pf_2T=45.xrdml'
pf220path = '/home/nate/wimv/Data/220pf_2theta=65.xrdml'

crystalSym = 'm-3m'
sampleSym = '1'
cellSize = np.deg2rad(5)
hkls = [(1,1,1),(2,0,0),(2,2,0)]

pfs = [pf111path,pf200path,pf220path]
pf = poleFigure(pfs, hkls, crystalSym, 'xrdml')

pf_grid, alp, bet = pf.grid(full=True, ret_ab=True)

with open('pf_od.out','rb') as f:
    
    pf_od = pickle.load(f)
    
with open('od_pf.out','rb') as f:
    
    od_pf = pickle.load(f)
    
temp = scio.loadmat('pointer2.mat')['pointer']

poles = range(3)

mtlb_pt = {}

for fi in poles:
    
    mtlb_pt[fi] = {}
    
    for od_cell in range(len(temp[0,fi][0])):
        
        mtlb_pt[fi][od_cell] = temp[0,fi][0,od_cell]
        
pf_num = 1

fig = plt.figure()

 ## PF Cell Loop ###

for od_cell in range(len(temp[0,fi][0])):
    
    # binned cell
    pf_cells = od_pf[pf_num][od_cell]
#    ai, bi = np.divmod(pf_cells, 72)    
#    pltcells = np.zeros_like(pf_grid)
#    pltcells[ai.astype(int),bi.astype(int)] += 1
#    pltcells = pltcells.T
    
    pf_cells2 = mtlb_pt[pf_num][od_cell] - 1
    
    pf_cells.sort()
    pf_cells2.sort()
    
    print(np.equal(pf_cells,pf_cells2))
    
#    ai, bi = np.divmod(pf_cells2, 72)    
#    pltcells2 = np.zeros_like(pf_grid)
#    pltcells2[ai.astype(int),bi.astype(int)] += 1
#    pltcells2 = pltcells2.T    
#    
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
#    ax.pcolormesh(theta,r,pltcells2,cmap='Blues')
#    
#    plt.pause(0.05)
#    plt.cla()   