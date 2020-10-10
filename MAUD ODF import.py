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
from pyTex.utils import tensor2voigt, voigt2tensor
from pyTex.orientation import eu2om
import numpy as np
import copy

crystalSym = 'm-3m'
sampleSym  = '1'
cellSize = np.deg2rad(5)

# # NRSF2
# od_file = '/mnt/c/Users/np7ut/Dropbox/ORNL/Texture/NRSF2/maud/exports/'
# sampleName = 'NRSF2_MAUD_detGroup_trun_eta_mmm_odf'

## NOMAD
od_file = '/mnt/c/Users/np7ut/Dropbox/ORNL/Texture/NOMAD/MAUD_new/exports/'
sampleName = 'NOMAD_Al_new_groupByTTH_mmm_odf'

MAUD_od = bunge._loadMAUD(od_file+sampleName, cellSize, crystalSym, sampleSym)

test = MAUD_od.export(od_file+'{}.odf'.format(sampleName.split('odf')[0]),vol_norm=True)

# %%

## volume fractions

# from tqdm import tqdm
# from scipy.spatial.transform import Rotation as R

# betaFiber =np.vstack([[35.3,45.0,0.0],
#             [33.6,47.7,5.0],
#             [32.1,51.0,10.0],
#             [31.1,54.7,15.0],
#             [31.3,59.1,20.0],
#             [35.2,64.2,25.0],
#             [46.1,69.9,30.0],
#             [49.5,76.2,35.0],          
#             [51.8,83.0,40.0],
#             [54.7,90.0,45.0],
#             [90.0,35.3,45.0],
#             [80.2,35.4,50.0],
#             [73.0,35.7,55.0],
#             [66.9,36.2,60.0],
#             [61.2,37.0,65.0],
#             [55.9,38.0,70.0],
#             [50.7,39.2,75.0],
#             [45.6,40.8,80.0],
#             [40.5,42.7,85.0],
#             [35.3,45.0,90.0]])

# g_betaFiber = R.from_euler('ZXZ', betaFiber,degrees=True).as_matrix()

# vf = []

# for g in tqdm(g_betaFiber):
    
#     vf.append(MAUD_od.compVolume(g,10))

# print(vf)

# %%

# print(maud_od.index())
# print(maud_od.entropy())
# # maud_od.plot3d()

# tube_rad = np.deg2rad(8)
# tube_exp = 1
# hkls = [(1,1,1),(3,1,1),(2,2,0)]

# test = maud_od.export(od_file+'NOMAD_Al_4Datasets_SSabs_MTEX_5res.odf',vol_norm=True)

# recalc_pf = maud_od._calcPF( hkls, tube_rad, tube_exp, tube_proj=True )

# cl = np.arange(0,8.5,0.5)
# recalc_pf.plot(contourlevels=cl,proj='earea')
# recalc_pf._interpolate(np.deg2rad(5))
# recalc_pf.export('/mnt/c/Users/Nate/Desktop',sampleName='maud_recalc')

# %%

# from tqdm import tqdm

# def transform(T,g):
    
#     Tprime = np.zeros([3,3,3,3])
    
#     for i in range(3):
#         for j in range(3):
#             for k in range(3):
#                 for l in range(3):
                    
#                     for m in range(3):
#                         for n in range(3):
#                             for o in range(3):
#                                 for p in range(3):
#                                     Tprime[i,j,k,l] += g[i,m] * g[j,n] * g[k,o] * g[l,p] * T[m,n,o,p]

#     return Tprime

# ## stiffness tensors

# # C11 = 106.8
# # C22 = C11
# # C33 = C11
# # C12 = 60.74
# # C13 = C12
# # C23 = C12
# # C44 = 28.21
# # C55 = C44
# # C66 = C44

# C11 = 52
# C12 = 34
# C13 = C12
# C23 = C12
# C22 = 52
# C33 = C22
# C44 = 173
# C55 = C44
# C66 = C44

# Cvoigt =  np.array([[C11, C12, C13, 0.0, 0.0, 0.0],
#                     [C12, C22, C23, 0.0, 0.0, 0.0],
#                     [C13, C23, C33, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, C44, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, C55, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 0.0, C66]]);  

# Svoigt = np.linalg.inv(Cvoigt)

# Ctensor = voigt2tensor(Cvoigt)
# Stensor = voigt2tensor(Svoigt,compliance=True)

# scale = (weights.flatten() * cellVolume.flatten()) / sum((weights.flatten() * cellVolume.flatten()))

# Ctemp = np.zeros([3,3,3,3])
# Stemp = np.zeros([3,3,3,3])

# for n,wgt in tqdm(enumerate(scale)):
    
#     g_temp = np.copy(g[:,:,n])
    
#     Sten_Tr = np.einsum('im,jn,ko,lp,mnop',g_temp,g_temp,g_temp,g_temp,Stensor)
#     Cten_Tr = np.einsum('im,jn,ko,lp,mnop',g_temp,g_temp,g_temp,g_temp,Ctensor)
#     Ctemp += Cten_Tr*wgt
#     Stemp += Sten_Tr*wgt

# Voigt = tensor2voigt(Ctemp)
# Reuss = np.linalg.inv(tensor2voigt(Stemp,compliance=True))

# Hill  = 0.5*( Voigt + Reuss )

# %% 

## sort angles into 2-fold sectors

# def bounds(path):
    
#     """
#     0 -> 90 (phi2) boundary: one
#     90 -> 0 (phi2) boundary: two
    
#     returns cond1, cond2, cond2
#     """

#     #bound1
#     func1_val = np.arccos( np.cos( path[:,2] ) / np.sqrt( 1 + np.cos( path[:,2] )**2 ) )
    
#     #bound2
#     func2_val = np.arccos( np.cos( path[:,2] - np.pi/2 ) / np.sqrt( 1 + np.cos( path[:,2] - np.pi/2 )**2 ) )
        
#     #condition1 - region I (Randle & Engler)
#     #Phi <= func #1 and Phi <= func #2
#     cond1 = ( ((func1_val - path[:,1]) > 0) * ((func2_val - path[:,1]) > 0) )
    
#     #condition2 - region III (Randle & Engler)
#     #Phi > func #1 and Phi > func #2
#     cond2 = ( ((func1_val - path[:,1]) <= 0) * ((func2_val - path[:,1]) <= 0) )
    
#     #condition2 - region II
#     cond2 = ~(cond1 + cond2)
    
#     return cond1, cond2, cond2
    
# ## plotting
# from scipy.spatial.transform import Rotation as R
# from scipy.spatial.distance import cdist

# pf_cell = 200
# fi = 1

# #calculate y's for given component
# brass = R.from_euler('ZXZ',(35,45,0),degrees=True)
# brass_y = brass.apply(recalc_pf._symHKL[fi])
# brass_y[brass_y[:,2] < 0] *= -1

# close_y = {}

# #calcuating along path
# fullPFgrid, xyz_pf = recalc_pf.genGrid(recalc_pf.res,
#                                         radians=True,
#                                         centered=False,
#                                         ret_xyz=True)

# # single point
# close_y[0] = pf_cell

# # #component
# # close_yList = []

# # for yi,y in enumerate(brass_y):
    
# #     close_y[yi] = np.argmin(cdist(xyz_pf,y[None,:]))
# #     close_yList.append(np.argmin(cdist(xyz_pf,y[None,:])))

# bungeAngs = np.zeros(( np.product(maud_od.phi1.shape), 3 ))

# for ii,i in enumerate(np.ndindex(maud_od.phi1.shape)):
    
#     bungeAngs[ii,:] = np.array((maud_od.phi1[i],maud_od.Phi[i],maud_od.phi2[i]))
    
# bungeAngs = bungeAngs

# #sym groups for grid
# cond1_grid, cond2_grid, cond3_grid = bounds(bungeAngs)

# ## mayavi
# # import mayavi.mlab as mlab

# # # fig = maud_od.plot3d()
# # fig = mlab.figure(bgcolor=(1,1,1))

# # ## grid ##
# # gd = mlab.points3d(bungeAngs[cond2_grid,0],
# #                    bungeAngs[cond2_grid,2],
# #                    -bungeAngs[cond2_grid,1],
# #                    scale_factor=1,
# #                    mode='point',c
# #                    olor=(0,0,0),
# #                    figure=fig)

# # gd.actor.property.render_points_as_spheres = True
# # gd.actor.property.point_size = 2
  
# ## lit point ##
# # gd2 = mlab.points3d(0,0,0,scale_factor=1,mode='point',color=(1,0,0))
# # gd2.actor.property.render_points_as_spheres = True
# # gd2.actor.property.point_size = 5

# cond_val = np.zeros((len(xyz_pf),3))

# comp = np.zeros((len(xyz_pf),2))

# # for yi, cell_num in close_y.items():
# for cell_num, y in enumerate(xyz_pf):
    
#     path = maud_od.paths['full']['euler path'][fi][cell_num]
#     od_cells = maud_od.pointer['full']['pf to od'][fi][cell_num]['cell']
#     od_weights = maud_od.pointer['full']['pf to od'][fi][cell_num]['weight']
    
#     cond1_grid, cond2_grid, cond3_grid = bounds(bungeAngs[od_cells.astype(int)])
#     # cond1, cond2, cond3 = bounds(path)
    
#     cond_val[cell_num,0] = ( 1 / np.sum(od_weights[cond1_grid]) ) * np.sum( od_weights[cond1_grid] * maud_od.weights[od_cells[cond1_grid].astype(int)] ) 
#     cond_val[cell_num,1] = ( 1 / np.sum(od_weights[cond2_grid]) ) * np.sum( od_weights[cond2_grid] * maud_od.weights[od_cells[cond2_grid].astype(int)] ) 
#     cond_val[cell_num,2] = ( 1 / np.sum(od_weights[cond3_grid]) ) * np.sum( od_weights[cond3_grid] * maud_od.weights[od_cells[cond3_grid].astype(int)] ) 
    
#     comp[cell_num,0] = np.mean(cond_val[cell_num])
#     comp[cell_num,1] = np.std(cond_val[cell_num])
    

#     # ## manual fibre ##
#     # path = maud_od.paths['full']['euler path'][fi][cell_num]
#     # tube = bungeAngs[od_cells]
    
#     # gd = mlab.points3d(path[cond2,0],
#     #                    path[cond2,2],
#     #                    -path[cond2,1],
#     #                    scale_factor=1,
#     #                    color=(0,1,0),
#     #                    mode='point',
#     #                    figure=fig)
    
#     # gd.actor.property.render_points_as_spheres = True
#     # gd.actor.property.point_size = 8
    
#     # ## tube ##
#     # gd2 = mlab.points3d(tube[:,2],
#     #                     tube[:,1],
#     #                     tube[:,0],
#     #                     scale_factor=1,
#     #                     mode='point',
#     #                     color=(0,0,1),
#     #                     figure=fig)
    
#     # gd2.actor.property.render_points_as_spheres = True
#     # gd2.actor.property.point_size = 5
    

# import matplotlib.pyplot as plt
# plt.plot(cond_val[:,0])
# plt.plot(cond_val[:,1])
# plt.plot(cond_val[:,2])

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

# %%

# import numpy as np
# from scipy.spatial.transform import Rotation as R
# from pyTex.orientation import symmetrise as symOri
# from pyTex.orientation import om2eu
# from pyTex.utils import genSymOps
# from pyTex import bunge

# ## volume fraction 

# cs = 'm-3m'
# ss = 'mmm'

# od = bunge(np.deg2rad(5), cs, ss)

# hkl = np.array([2, 1, 3])
# hkl = np.divide(hkl, np.linalg.norm(hkl))

# test = R.from_euler('ZXZ', betaFiber[2], degrees=True)
# testSym = symOri(test.as_matrix(),cs, ss)

# #try transpose
# testSym = testSym.transpose((0,2,1))

# # convert to euler
# testSym_eu = om2eu(testSym)

# # pick fundamental zone
# fz = (testSym_eu[:,0] <= 2*np.pi) & (testSym_eu[:,1] <= np.pi/2) & (testSym_eu[:,2] <= np.pi/2)
# fz_idx = np.nonzero(fz)
# g_fz = testSym[fz_idx[0],:,:]

# # generate crystal sym ops
# crysSymOps = genSymOps(cs)
# smplSymOps = genSymOps(ss)

# # create Nx3 array of grid points
# eu_grid = np.array([od.phi1.flatten(),od.Phi.flatten(),od.phi2.flatten()]).T
# g_grid  = eu2om(eu_grid,out='mdarray_2')
# g_grid  = g_grid.transpose((2,0,1))

# trace = {}
# misori = {}
# mo_cell = []

# for gi,g in enumerate(g_fz):    
    
#     trace[gi] = []
#     misori[gi] = []
#     k = 0
    
#     for crys_op in crysSymOps:
        
#         # for smpl_op in smplSymOps:
    
#         temp = g @ g_grid
#         test = crys_op @ temp 
#         trace[gi].append( np.trace( test,axis1=1,axis2=2 ) )
        
#         #calculate misorientation
#         mo = np.arccos( np.clip( (trace[gi][k] - 1)/2, -1, 1) )
        
#         #criteria
#         crit = np.where(mo <= np.deg2rad(10))
#         # crit = np.argmin(mo)
        
#         #store cell id, misorientation angle for each sym equiv.
#         misori[gi].append( np.array( [crit[0], mo[crit]] ).T )
#         k+=1
            
#     # concatenate, pull true min from sym equiv.
#     misori[gi] = np.vstack(misori[gi])
#     # mo_cell.append( misori[gi][ np.argmin(misori[gi][:,1]), 0 ].astype(int) )
#     mo_cell.append(np.unique(misori[gi],axis=0)[:,0].T)    
        
#     # misori.append(np.argmin(np.arccos((np.vstack(trace)-1)/2)))
#     # k+=1

# mo_cell = np.unique(np.hstack(mo_cell).astype(int))

# import mayavi.mlab as mlab
# from pyTex import bunge

# od = bunge(np.deg2rad(5), cs, ss)

# mlab.figure(bgcolor=(1,1,1))

# gd = mlab.points3d(od.phi2/od.res + 1,od.Phi/od.res + 1,od.phi1/od.res + 1,mode='point',scale_factor=1,color=(0.25,0.25,0.25))
# gd.actor.property.render_points_as_spheres = True
# gd.actor.property.point_size = 2

# pts = mlab.points3d(testSym_eu[fz_idx,2]/od.res + 1,testSym_eu[fz_idx,1]/od.res + 1,testSym_eu[fz_idx,0]/od.res + 1,mode='point',scale_factor=1,color=(0,1,0))
# pts.actor.property.render_points_as_spheres = True
# pts.actor.property.point_size = 10

# pts = mlab.points3d(eu_grid[mo_cell,2]/od.res + 1,eu_grid[mo_cell,1]/od.res + 1,eu_grid[mo_cell,0]/od.res + 1,mode='point',scale_factor=1,color=(1,0,0))
# pts.actor.property.render_points_as_spheres = True
# pts.actor.property.point_size = 6

# MAUD_od.plot3d()

# %%

## testing import routine

# with open(od_file+sampleName,'r') as f:

#     odf_data = []
    
#     #read in odf data
#     odf_str = f.readlines()
#     counter = [0,0]
#     for n,line in enumerate(odf_str):
#         if n < 3: 
#             counter[0] += 1
#             counter[1] += 1
#         else:
#             if line in ['\n']: #check for blank line
#                 odf_data.append(np.genfromtxt(odf_str[counter[0]:counter[1]]))
#                 # print(counter)
#                 counter[0] = n+1
#                 counter[1] += 1
#             else: counter[1] += 1 
                
#     print('loaded ODF')
#     print('header: "'+odf_str[0].strip('\n')+'"')
#     file_sym = int(odf_str[1].split(' ')[0])

# sym_beartex = {11: ['622'],
#                 10: ['6'],
#                 9: ['32'],
#                 8: ['3'],
#                 7: ['432','m-3m'],
#                 6: ['23','m-3'],
#                 5: ['422'],
#                 4: ['4'],
#                 3: ['222'],
#                 2: ['2'],
#                 1: ['1']}

# file_sym = sym_beartex.get(file_sym, lambda: 'Unknown Laue group')
# if any([crystalSym in sym for sym in file_sym]): pass
# else: print('Supplied crystal sym does not match file sym')

# # set boundary in Bunge space (not rigorous for cubic)
# if sampleSym == '1': alphaMax = np.deg2rad(360)
# elif sampleSym == 'm': alphaMax = np.deg2rad(180)
# elif sampleSym == 'mmm': alphaMax = np.deg2rad(90)
# else: raise ValueError('invalid sampleSym')

# if crystalSym == 'm-3m' or crystalSym == '432': 
#     betaMax = np.deg2rad(90)
#     gammaMax = np.deg2rad(90)
# elif crystalSym == 'm-3' or crystalSym == '23': raise NotImplementedError('coming soon..')
# else: raise ValueError('invalid crystalSym, only cubic so far..')

# gammaRange = np.arange(0,gammaMax+cellSize,cellSize)
# betaRange  = np.arange(0,betaMax+cellSize,cellSize)
# alphaRange = np.arange(0,alphaMax,cellSize)

# gam, bet, alp = np.meshgrid(gammaRange,betaRange,alphaRange,indexing='ij')

# weights = np.zeros_like(gam)

# for gi,g in enumerate(gammaRange):
#     for bi,b in enumerate(betaRange):
#         for ai,a in enumerate(alphaRange):
            
#             weights[gi,bi,ai] = odf_data[gi][bi,ai]
            
# out = np.array([gam.flatten(),bet.flatten(),alp.flatten(),weights.flatten()]).T

# ## shift back to phi1, Phi, phi2
# new_phi1 = alp + np.pi/2
# new_Phi  = np.copy(bet)
# new_phi2 = -gam + np.pi/2
# new_phi1 = np.where(new_phi1 > alphaMax, new_phi1 - alphaMax, new_phi1) #brnng back to 0 - 2pi
# new_phi2 = np.where(new_phi2 > gammaMax, new_phi2 - gammaMax, new_phi2) #brnng back to 0 - 2pi

# phi1_0 = np.argmax(new_phi1[0,0,:])

# ## add duplicate slice (phi1=360) at phi1 = 0
# new_phi1 = np.insert(new_phi1,0,np.zeros_like(new_phi1[:,:,phi1_0]),axis=2)
# new_Phi  = np.insert(new_Phi,0,new_Phi[:,:,0],axis=2)
# new_phi2 = np.insert(new_phi2,0,new_phi2[:,:,0],axis=2)
# weights  = np.insert(weights,0,weights[:,:,phi1_0],axis=2)

# out = np.array([new_phi1.flatten(),new_Phi.flatten(),new_phi2.flatten(),weights.flatten()]).T
# out_sort = out[np.lexsort((out[:,0],out[:,1],out[:,2]))]

# phi1 =  out_sort[:,0].reshape(new_phi1.shape)
# Phi  =  out_sort[:,1].reshape(new_Phi.shape)
# phi2 =  out_sort[:,2].reshape(new_phi2.shape)
# weights = np.copy(out_sort[:,3])

# g, bungeList = eu2om((phi1,Phi,phi2),out='mdarray')

# # no edge correction
# cellVolume = cellSize * cellSize * ( np.cos( Phi - ( cellSize/2 ) ) - np.cos( Phi + ( cellSize/2 ) ) )