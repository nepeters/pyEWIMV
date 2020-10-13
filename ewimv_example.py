#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 22:10:29 2020
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

import numpy as np
from scipy.spatial.transform import Rotation as R

from pyEWIMV import poleFigure, euler
from pyEWIMV.inversion import e_wimv

dir_path = os.path.dirname(os.path.realpath('__file__'))

P = 1

#sample information
sampleName = 'Al_NOMAD_10it_5x7_mmm_vn'
rad_type = 'nd'
def_al = {'name': 'Al',
          'composition': [dict(ion='Al', pos=[0, 0, 0]),
                          dict(ion='Al', pos=[0.5, 0, 0.5]),
                          dict(ion='Al', pos=[0.5, 0.5, 0]),
                          dict(ion='Al', pos=[0, 0.5, 0.5])],
          'lattice': dict(abc=[4.0465, 4.0465, 4.0465], abg=[90, 90, 90]),
          'debye-waller': False,
          'massNorm': False}

crystalSym = 'm-3m'
sampleSym = '1'
cellSize = np.deg2rad(5)
od = euler(cellSize, crystalSym, sampleSym)

#tube radius
theta = np.deg2rad(7)
#tube exponent
tube_exp = 1

""" NRSF2 .jul """

#define pole figures
data_path = os.path.join(dir_path, 'Data', 'HB2B - Aluminum')
hkls = np.array([(2,2,2), (3,1,1), (4,0,0)])
pf222path = os.path.join(data_path, 'HB2B_exp129_3Chi_222.jul')
pf311path = os.path.join(data_path, 'HB2B_exp129_3Chi_311.jul')
pf400path = os.path.join(data_path, 'HB2B_exp129_3Chi_400.jul')

#load pole figures
pf = poleFigure([pf222path,pf311path,pf400path], hkls, crystalSym, 'jul')
# pf = poleFigure([pf222path], hkls, crystalSym, 'jul')

rot = R.from_euler('ZYX', (90,90,90), degrees=True).as_matrix()

""" peak-fitted pole figures """

# hkls = []
# files = []

# # datadir = os.path.join(dir_path,'Data','NOMAD Aluminum - no abs','combined')
# # datadir = os.path.join(dir_path,'Data','NOMAD Nickel - full abs - peak int','pole figures','combined')
# # datadir = os.path.join(dir_path,'Data','NOMAD Aluminum - no abs - peak int','combined')
# # datadir = '/media/nate/2E7481AA7481757D/Users/Nate/Dropbox/ORNL/Texture/NRSF2/mtex_export'
# # datadir = '/mnt/c/Users/Nate/pyReducePF/pole figures/pole figures peak int Al absCorr/combined'
# datadir = '/home/nate/projects/pyReducePF/pole figures/pole figures integ int Al absCorr 2ndFit/combined'

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
    

# pf = poleFigure(files,hkls,crystalSym,'sparse')

# rot = R.from_euler('XZY',(13,-88,90), degrees=True).as_matrix()

#rotate pole figures
pf.rotate(rot)

#perform E-WIMV iterations
recalc_pf, calc_od = e_wimv( pf, od, theta, tube_exp, rad_type, def_al, iterations=10 )
final_iter = max(list(calc_od.keys()))

# #recalculate (111), (200), (220)
# hkls = [(1,1,1),(2,0,0),(2,2,0)]
# recalc_pf_new = calc_od[final_iter-1].calcPF( hkls, theta, tube_exp, tube_proj=True )

# plot recalculated pole figures
recalc_pf[final_iter-1].plot(pfs=3,cmap='viridis_r',proj='none')

# #plot ODF section
# calc_od[final_iter-1].sectionPlot('phi2',np.deg2rad(65))

# copper = R.from_euler('ZXZ', [90, 35, 45], degrees=True)

#calculate texture index & entropy
print(sampleName)
print('iterations: '+str(final_iter))
print(calc_od[final_iter-1].index())
print(calc_od[final_iter-1].entropy())

recalc_pf[final_iter-1].export_beartex('test',sampleName=sampleName)

# %%

## volume fractions

# from tqdm import tqdm

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
    
#     vf.append(calc_od[final_iter-1].compVolume(g,15))

# print(vf)

# print(calc_od[final_iter-1]._volume(copper.as_matrix(),10))

# C11 = 52
# C12 = 34
# C13 = C12
# C23 = C12
# C22 = 52
# C33 = C22
# C44 = 173
# C55 = C44
# C66 = C44

# elastic =  np.array([[C11, C12, C13, 0.0, 0.0, 0.0],
#                      [C12, C22, C23, 0.0, 0.0, 0.0],
#                      [C13, C23, C33, 0.0, 0.0, 0.0],
#                      [0.0, 0.0, 0.0, C44, 0.0, 0.0],
#                      [0.0, 0.0, 0.0, 0.0, C55, 0.0],
#                      [0.0, 0.0, 0.0, 0.0, 0.0, C66]]);  

# # voigt = calc_od[final_iter-1].voigt(elastic)
# # reuss = calc_od[final_iter-1].reuss(np.linalg.inv(elastic))
# hill = calc_od[final_iter-1].hill(elastic)

# export data
# calc_od[final_iter-1].export('/mnt/c/Users/np7ut/Dropbox/ORNL/Manuscript/github/Results/directEWIMV/'+sampleName+'.odf',vol_norm=True)
# recalc_pf[final_iter-1].export('/mnt/c/Users/np7ut/Dropbox/ORNL/Manuscript/github/Results/directEWIMV/',sampleName=sampleName)
# recalc_pf_new.export('/mnt/c/Users/Nate/Dropbox/ORNL/EWIMVvsMTEX/EWIMV exports (abs corr)/',sampleName=sampleName)

# %%

# ## plot fiber tube

# pf_num = 0
# yi     = 700

# bungeAngs = np.zeros(( np.product(od.phi1cen.shape), 3 ))
# for ii,i in enumerate(np.ndindex(od.phi1cen.shape)):    
#     bungeAngs[ii,:] = np.array((od.phi1cen[i],od.Phicen[i],od.phi2cen[i]))

# fibers = od.paths['arb']['euler path'][pf_num][yi]

# import mayavi.mlab as mlab

# mlab.figure(bgcolor=(1,1,1))

# gd = mlab.points3d(od.phi1,od.Phi,od.phi2,mode='point',scale_factor=1,color=(0.25,0.25,0.25))
# gd.actor.property.render_points_as_spheres = True
# gd.actor.property.point_size = 2

# mlab.axes(color=(0,0,0),ranges=[0,2*np.pi,0,np.pi/2,0,np.pi/2])

# od_cells = od.paths['arb']['grid points'][pf_num][yi].astype(int)

# pts = mlab.points3d(bungeAngs[od_cells,0],bungeAngs[od_cells,1],bungeAngs[od_cells,2],mode='point',scale_factor=1,color=(0,1,0))
# pts.actor.property.render_points_as_spheres = True
# pts.actor.property.point_size = 5

# # pts2 = mlab.points3d(bungeAngs[od_cells_new,0],bungeAngs[od_cells_new,1],bungeAngs[od_cells_new,2],mode='point',scale_factor=1,color=(0,1,1))
# # pts2.actor.property.render_points_as_spheres = True
# # pts2.actor.property.point_size = 6

# centerTube = mlab.plot3d(fibers[:,0],fibers[:,1],fibers[:,2],tube_sides=36,opacity=0.25,tube_radius=theta,color=(0,0,1))

# fiber = mlab.points3d(fibers[:,0],fibers[:,1],fibers[:,2],mode='point',scale_factor=1,color=(1,0,0)) 
# fiber.actor.property.render_points_as_spheres = True
# fiber.actor.property.point_size = 5

