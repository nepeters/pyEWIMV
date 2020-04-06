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

from pyTex import poleFigure, bunge
from pyTex.inversion import e_wimv

dir_path = os.path.dirname(os.path.realpath('__file__'))

P = 1

#sample information
sampleName = 'Al_NRSF2_5x8'
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
sampleSym = 'mmm'
cellSize = np.deg2rad(5)
od = bunge(cellSize, crystalSym, sampleSym)

#tube radius
theta = np.deg2rad(8)
#tube exponent
tube_exp = 1

""" NRSF2 .jul """

# #define pole figures
# data_path = os.path.join(dir_path, 'Data', 'HB2B - Aluminum')
# hkls = np.array([(2,2,2), (3,1,1), (4,0,0)])
# pf222path = os.path.join(data_path, 'HB2B_exp129_3Chi_222.jul')
# pf311path = os.path.join(data_path, 'HB2B_exp129_3Chi_311.jul')
# pf400path = os.path.join(data_path, 'HB2B_exp129_3Chi_400.jul')

# #load pole figures
# pf = poleFigure([pf222path,pf311path,pf400path], hkls, crystalSym, 'jul')

# rot = R.from_euler('XZX', (90,90,90), degrees=True).as_matrix()

""" peak-fitted pole figures """

hkls = []
files = []

# datadir = os.path.join(dir_path,'Data','NOMAD Aluminum - no abs','combined')
# datadir = os.path.join(dir_path,'Data','NOMAD Nickel - full abs - peak int','pole figures','combined')
# datadir = os.path.join(dir_path,'Data','NOMAD Aluminum - no abs - peak int','combined')
# datadir = '/media/nate/2E7481AA7481757D/Users/Nate/Dropbox/ORNL/Texture/NRSF2/mtex_export'
# datadir = '/mnt/c/Users/Nate/pyReducePF/pole figures/pole figures peak int Al absCorr/combined'
datadir = '/home/nate/projects/pyReducePF/pole figures/pole figures integ int Al absCorr 2ndFit/combined'

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
    

pf = poleFigure(files,hkls,crystalSym,'sparse')

rot = R.from_euler('XZY',(13,-88,90), degrees=True).as_matrix()

#rotate pole figures
pf.rotate(rot)

#perform E-WIMV iterations
recalc_pf, calc_od = e_wimv( pf, od, theta, tube_exp, rad_type, def_al, iterations=8 )
final_iter = max(list(calc_od.keys()))

# #recalculate (111), (200), (220)
# hkls = [(1,1,1),(2,0,0),(2,2,0)]
# recalc_pf_new = calc_od[final_iter-1].calcPF( hkls, theta, tube_exp, tube_proj=True )

# # plot recalculated pole figures
# cl = np.arange(0,10.5,0.5)
# recalc_pf[final_iter-1].plot(pfs=3,contourlevels=cl,cmap='viridis_r',proj='none')
# # recalc_pf[7].plot(contourlevels=cl,cmap='viridis',proj='stereo')

# #plot ODF section
# calc_od[final_iter-1].sectionPlot('phi2',np.deg2rad(65))

# copper = R.from_euler('ZXZ', [90, 35, 45], degrees=True)

#calculate texture index & entropy
print(sampleName)
print('iterations: '+str(final_iter-1))
print(calc_od[final_iter-1].index())
print(calc_od[final_iter-1].entropy())

# %%

## volume fractions

from tqdm import tqdm

betaFiber =np.vstack([[35.3,45.0,0.0],
            [33.6,47.7,5.0],
            [32.1,51.0,10.0],
            [31.1,54.7,15.0],
            [31.3,59.1,20.0],
            [35.2,64.2,25.0],
            [46.1,69.9,30.0],
            [49.5,76.2,35.0],          
            [51.8,83.0,40.0],
            [54.7,90.0,45.0],
            [90.0,35.3,45.0],
            [80.2,35.4,50.0],
            [73.0,35.7,55.0],
            [66.9,36.2,60.0],
            [61.2,37.0,65.0],
            [55.9,38.0,70.0],
            [50.7,39.2,75.0],
            [45.6,40.8,80.0],
            [40.5,42.7,85.0],
            [35.3,45.0,90.0]])

g_betaFiber = R.from_euler('ZXZ', betaFiber,degrees=True).as_matrix()

vf = []

for g in tqdm(g_betaFiber):
    
    vf.append(calc_od[final_iter-1]._volume(g,10))


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

## export data
# calc_od[final_iter-1].export('/mnt/c/Users/Nate/Dropbox/ORNL/EWIMVvsMTEX/EWIMV exports (abs corr)/'+sampleName+'.odf',vol_norm=True)
# recalc_pf[final_iter-1].export('/mnt/c/Users/Nate/Dropbox/ORNL/EWIMVvsMTEX/EWIMV exports (abs corr)/',sampleName=sampleName)
# recalc_pf_new.export('/mnt/c/Users/Nate/Dropbox/ORNL/EWIMVvsMTEX/EWIMV exports (abs corr)/',sampleName=sampleName)

# %% attempt with symmetry

# #take OD and copy it to create full 90x90x360
# new_od = bunge(np.deg2rad(5), 'm-3m', 'mmm')

# weights = np.copy(calc_od[7].weights).reshape(calc_od[7].phi1cen.shape)
# weights_tld = np.tile(weights,(1,1,4))

# new_od.weights=weights_tld.flatten()

# hkls = np.array([(2,2,2), (3,1,1), (4,0,0)])
# recalc_pf_new2 = new_od.calcPF( hkls, theta, tube_exp, tube_proj=True )

# %%

# from pyTex.utils import genSymOps
# from pyTex.orientation import quat2eu, om2eu, eu2om
# from scipy.spatial.transform import Rotation as R

# mmm_ops = genSymOps('mmm')
# mmm_ops = np.swapaxes(mmm_ops,2,0)
# mmm_ops = mmm_ops[np.where( np.linalg.det(mmm_ops) == 1 )]
# mmm_ops_R = R.from_matrix(mmm_ops)

# bunge = eu2om([np.deg2rad(65),np.deg2rad(65),np.deg2rad(65)])
# bunge_R = R.from_matrix(bunge)

# test = np.zeros((4,3,3))

# for i,smpl_op in enumerate(mmm_ops_R):
   
#     temp = bunge_R * smpl_op
#     test[i,:,:] = temp.as_matrix()
    
# test = om2eu(test)    

# import mayavi.mlab as mlab

# xAx=mlab.quiver3d(0,0,0,2*np.pi,0,0,line_width=0.5,scale_factor=1,mode='arrow',color=(1,0,0))
# xAx.glyph.glyph_source.glyph_source.tip_radius = 0.02
# xAx.glyph.glyph_source.glyph_source.tip_length = 0.1
# xAx.glyph.glyph_source.glyph_source.shaft_radius = 0.005
# yAx=mlab.quiver3d(0,0,0,0,np.pi,0,line_width=0.5,scale_factor=1,mode='arrow',color=(0,1,0))
# yAx.glyph.glyph_source.glyph_source.tip_radius = 0.02
# yAx.glyph.glyph_source.glyph_source.tip_length = 0.1
# yAx.glyph.glyph_source.glyph_source.shaft_radius = 0.005
# zAx=mlab.quiver3d(0,0,0,0,0,2*np.pi,line_width=0.5,scale_factor=1,mode='arrow',color=(0,0,1))
# zAx.glyph.glyph_source.glyph_source.tip_radius = 0.02
# zAx.glyph.glyph_source.glyph_source.tip_length = 0.1
# zAx.glyph.glyph_source.glyph_source.shaft_radius = 0.005

# pts = mlab.points3d(test[:,0],test[:,1],test[:,2],mode='point')
# pts.actor.property.render_points_as_spheres = True
# pts.actor.property.point_size = 8
    
# phi1_new, Phi_new, phi2_new = quat2eu(test)

# phi1_new = np.where(phi1_new < 0, phi1_new + 2*np.pi, phi1_new) #brnng back to 0 - 2pi
# Phi_new = np.where(Phi_new < 0, Phi_new + np.pi, Phi_new) #brnng back to 0 - pi
# phi2_new = np.where(phi2_new < 0, phi2_new + 2*np.pi, phi2_new) #brnng back to 0 - 2pi

# #fundamental zone calc (not true!)
# all_new = np.stack( (phi1_new, Phi_new, phi2_new), axis=2 )
# all_new = np.reshape( all_new, (all_new.shape[0]*all_new.shape[1],all_new.shape[2]) ) #new method 

# fz = (all_new[:,0] <= 2*np.pi) & (all_new[:,1] <= np.pi/2) & (all_new[:,2] <= np.pi/2)
# fz_idx = np.nonzero(fz)

# eu_fz = all_new[fz_idx]

# all_new2 = np.vstack(test_eu)

# phi1_new = np.where(all_new2[:,0] < 0, all_new2[:,0] + 2*np.pi, all_new2[:,0]) #brnng back to 0 - 2pi
# Phi_new  = np.where(all_new2[:,1] < 0, all_new2[:,1] + np.pi, all_new2[:,1]) #brnng back to 0 - pi
# phi2_new = np.where(all_new2[:,2] < 0, all_new2[:,2] + 2*np.pi, all_new2[:,2]) #brnng back to 0 - 2pi


 # %%
 
 #rotations in od

# phi1, Phi, phi2 = bunge._genGrid(np.deg2rad(5), np.deg2rad(360), np.deg2rad(90), np.deg2rad(90))

# bungeList = np.array([phi1.flatten(),Phi.flatten(),phi2.flatten()]).T
# bunge_R = R.from_euler('ZXZ', bungeList)

# # test = np.zeros((len(bungeList),4,4))
# test_eu = np.zeros((len(bungeList)*4,3,3))
# k = 0

# for i,rot in enumerate(bunge_R):
    
    
#     sym_R = mmm_ops_R * rot
#     # test[i,:,:] = sym_R.as_quat()
#     # test_eu.append(sym_R.as_euler('ZXZ'))
#     for sr in sym_R:
#         test_eu[k,:,:] = sr.as_matrix()
#         k += 1

# test_eu = np.clip(test_eu,-1,1)
# all_new2 = om2eu(test_eu)

# import mayavi.mlab as mlab

# pts = mlab.points3d(all_new2[:,0],all_new2[:,1],all_new2[:,2],mode='point',color=(0,0,0))
# pts.actor.property.render_points_as_spheres = True
# pts.actor.property.point_size = 8

# pts2 = mlab.points3d(bungeList[:,0],bungeList[:,1],bungeList[:,2],mode='point',color=(0,0,1))
# pts2.actor.property.render_points_as_spheres = True
# pts2.actor.property.point_size = 4

# orig = mlab.points3d(0,0,0,mode='point',color=(1,0,0))
# orig.actor.property.render_points_as_spheres = True
# orig.actor.property.point_size = 10