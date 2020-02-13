#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 21:35:33 2020

@author: nate
"""

"""
MAUD ODF import 
"""

from pyTex import bunge
import numpy as np

crystalSym = 'm-3m'
sampleSym = 'mmm'
cellSize = np.deg2rad(5)

od_file = '/mnt/c/Users/Nate/Dropbox/ORNL/EWIMVvsMTEX/MAUD EWIMV exports/'
sampleName = 'NOMAD_Al_4Datasets_SSabs_MAUD_5res.odf'

maud_od = bunge.loadMAUD(od_file+sampleName,
                         np.deg2rad(5),
                         'm-3m',
                         '1')

cV = maud_od.res * maud_od.res * ( np.cos( maud_od.Phi - ( maud_od.res/2 ) ) - np.cos( maud_od.Phi + ( maud_od.res/2 ) ) )

print(maud_od.index())
print(maud_od.entropy())
# maud_od.plot3d()

tube_rad = np.deg2rad(8)
tube_exp = 1
hkls = [(1,1,1),(3,1,1),(2,2,0)]

test = maud_od.export(od_file+'NOMAD_Al_4Datasets_SSabs_MTEX_5res.odf',vol_norm=True)

# recalc_pf = maud_od._calcPF( hkls, tube_rad, tube_exp, tube_proj=True )

# cl = np.arange(0,8.5,0.5)
# recalc_pf.plot(contourlevels=cl,proj='earea')
# recalc_pf._interpolate(np.deg2rad(5))
# recalc_pf.export('/mnt/c/Users/Nate/Desktop',sampleName='maud_recalc')

# %% 

## sort angles into 3-fold sectors

def bounds(path):
    
    """
    0 -> 90 (phi2) boundary: one
    90 -> 0 (phi2) boundary: two
    
    returns cond1, cond2, cond3
    """

    #bound1
    func1_val = np.arccos( np.cos( path[:,2] ) / np.sqrt( 1 + np.cos( path[:,2] )**2 ) )
    
    #bound2
    func2_val = np.arccos( np.cos( path[:,2] - np.pi/2 ) / np.sqrt( 1 + np.cos( path[:,2] - np.pi/2 )**2 ) )
        
    #condition1 - region I (Randle & Engler)
    #Phi <= func #1 and Phi <= func #2
    cond1 = ( ((func1_val - path[:,1]) > 0) * ((func2_val - path[:,1]) > 0) )
    
    #condition2 - region III (Randle & Engler)
    #Phi > func #1 and Phi > func #2
    cond2 = ( ((func1_val - path[:,1]) <= 0) * ((func2_val - path[:,1]) <= 0) )
    
    #condition3 - region II
    cond3 = ~(cond1 + cond2)
    
    return cond1, cond2, cond3
    
## plotting
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist

pf_cell = 300
fi = 1

#calculate y's for given component
brass = R.from_euler('ZXZ',(35,45,0),degrees=True)
brass_y = brass.apply(recalc_pf._symHKL[fi])
brass_y[brass_y[:,2] < 0] *= -1

close_y = {}

#calcuating along path
fullPFgrid, xyz_pf = recalc_pf.genGrid(recalc_pf.res,
                                        radians=True,
                                        centered=False,
                                        ret_xyz=True)

# single point
close_y[0] = pf_cell

# #component
# close_yList = []

# for yi,y in enumerate(brass_y):
    
#     close_y[yi] = np.argmin(cdist(xyz_pf,y[None,:]))
#     close_yList.append(np.argmin(cdist(xyz_pf,y[None,:])))

bungeAngs = np.zeros(( np.product(maud_od.phi1.shape), 3 ))

for ii,i in enumerate(np.ndindex(maud_od.phi1.shape)):
    
    bungeAngs[ii,:] = np.array((maud_od.phi1[i],maud_od.Phi[i],maud_od.phi2[i]))
    
bungeAngs = bungeAngs

#sym groups for grid
cond1_grid, cond2_grid, cond3_grid = bounds(bungeAngs)

## mayavi
# import mayavi.mlab as mlab

# # fig = maud_od.plot3d()
# fig = mlab.figure(bgcolor=(1,1,1))

# ## grid ##
# gd = mlab.points3d(bungeAngs[cond2_grid,0],
#                    bungeAngs[cond2_grid,2],
#                    -bungeAngs[cond2_grid,1],
#                    scale_factor=1,
#                    mode='point',c
#                    olor=(0,0,0),
#                    figure=fig)

# gd.actor.property.render_points_as_spheres = True
# gd.actor.property.point_size = 2
  
## lit point ##
# gd2 = mlab.points3d(0,0,0,scale_factor=1,mode='point',color=(1,0,0))
# gd2.actor.property.render_points_as_spheres = True
# gd2.actor.property.point_size = 5

cond_val = np.zeros((len(xyz_pf),3))

comp = np.zeros((len(xyz_pf),2))

# for yi, cell_num in close_y.items():
for cell_num, y in enumerate(xyz_pf):
    
    path = maud_od.paths['full']['euler path'][fi][cell_num]
    od_cells = maud_od.pointer['full']['pf to od'][fi][cell_num]['cell']
    od_weights = maud_od.pointer['full']['pf to od'][fi][cell_num]['weight']
    
    cond1_grid, cond2_grid, cond3_grid = bounds(bungeAngs[od_cells.astype(int)])
    # cond1, cond2, cond3 = bounds(path)
    
    cond_val[cell_num,0] = ( 1 / np.sum(od_weights[cond1_grid]) ) * np.sum( od_weights[cond1_grid] * maud_od.weights[od_cells[cond1_grid].astype(int)] ) 
    cond_val[cell_num,1] = ( 1 / np.sum(od_weights[cond2_grid]) ) * np.sum( od_weights[cond2_grid] * maud_od.weights[od_cells[cond2_grid].astype(int)] ) 
    cond_val[cell_num,2] = ( 1 / np.sum(od_weights[cond3_grid]) ) * np.sum( od_weights[cond3_grid] * maud_od.weights[od_cells[cond3_grid].astype(int)] ) 
    
    comp[cell_num,0] = np.mean(cond_val[cell_num])
    comp[cell_num,1] = np.std(cond_val[cell_num])
    

    # ## manual fibre ##
    # path = maud_od.paths['full']['euler path'][fi][cell_num]
    # tube = bungeAngs[od_cells]
    
    # gd = mlab.points3d(path[cond2,0],
    #                    path[cond2,2],
    #                    -path[cond2,1],
    #                    scale_factor=1,
    #                    color=(0,1,0),
    #                    mode='point',
    #                    figure=fig)
    
    # gd.actor.property.render_points_as_spheres = True
    # gd.actor.property.point_size = 8
    
    # ## tube ##
    # gd2 = mlab.points3d(tube[:,2],
    #                     tube[:,1],
    #                     tube[:,0],
    #                     scale_factor=1,
    #                     mode='point',
    #                     color=(0,0,1),
    #                     figure=fig)
    
    # gd2.actor.property.render_points_as_spheres = True
    # gd2.actor.property.point_size = 5
    

import matplotlib.pyplot as plt
plt.plot(cond_val[:,0])
plt.plot(cond_val[:,1])
plt.plot(cond_val[:,2])

# %%

# from pyTex import poleFigure

# fullPFgrid, alp, bet = poleFigure.grid(res=np.deg2rad(5),
#                                        radians=True,
#                                        cen=False,
#                                        ret_ab=True)

# #calculate pole figure y's
# sph = np.array((np.ravel(alp),np.ravel(bet))).T
# #offset (001) direction to prevent errors during path calculation
# sph[:,0] = np.where(sph[:,0] == 0, np.deg2rad(0.25), sph[:,0])

# #convert to xyz
# xyz_pf = np.zeros((sph.shape[0],3))
# xyz_pf[:,0] = np.sin( sph[:,0] ) * np.cos( sph[:,1] )
# xyz_pf[:,1] = np.sin( sph[:,0] ) * np.sin( sph[:,1] )
# xyz_pf[:,2] = np.cos( sph[:,0] )
