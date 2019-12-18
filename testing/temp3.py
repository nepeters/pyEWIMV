#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 09:24:20 2019

@author: nate
"""

import sys

import numpy as np

sys.path.insert(0,'/home/nate/wimv')

from pyTex import poleFigure, bunge
from pyTex.inversion import wimv

crystalSym = 'm-3m'
sampleSym = '1'
cellSize = np.deg2rad(5)
hkls = [(1,1,1),(2,0,0),(2,2,0)]

#bkgd111path = '/home/nate/wimv/Data/316L_AM/Horizontal_BDup/Cr Tube/Bkgd_73_316L_horiz.xrdml'
#bkgd200path = '/home/nate/wimv/Data/316L_AM/Horizontal_BDup/Cr Tube/Bkgd_100_316L_horiz.xrdml'
#bkgd220path = '/home/nate/wimv/Data/316L_AM/Horizontal_BDup/Cr Tube/Bkgd_100_316L_horiz.xrdml'
#
#bkgds = [bkgd111path,bkgd200path,bkgd220path]
#bkgd = poleFigure(bkgds, hkls, crystalSym, 'xrdml',subtype='bkgd')
#
#pf111path = '/home/nate/wimv/Data/316L_AM/Horizontal_BDup/Cr Tube/110_pf_316L_horiz.xrdml'
#pf200path = '/home/nate/wimv/Data/316L_AM/Horizontal_BDup/Cr Tube/200_pf_316L_horiz.xrdml'
#pf220path = '/home/nate/wimv/Data/316L_AM/Horizontal_BDup/Cr Tube/211_pf_316L_horiz.xrdml'

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

pf_grid, alp, bet = pf.grid(full=True,ret_ab=True)

recalc_pf, calc_od, pf_od, od_pf = wimv(pf, od, ret_pointer=True)

# %%
            
### FIBER CALC ###

import rowan
from tqdm import tqdm

from pyTex.orientation import ro2ax, ax2om, om2eu

#calculate pole figure y's

sph = np.array(np.divmod(np.ravel(pf_grid),pf_grid.shape[1])).T
sph = sph * pf.res

#convert to xyz
pf_xyz = np.zeros((sph.shape[0],3))

pf_xyz[:,0] = np.sin( sph[:,0] ) * np.cos( sph[:,1] )
pf_xyz[:,1] = np.sin( sph[:,0] ) * np.sin( sph[:,1] )
pf_xyz[:,2] = np.cos( sph[:,0] )

pf_num = 0

omega = np.radians(np.arange(0,365,5))

ro_y = {}
yi = 0

for v in tqdm(pf_xyz):
    
    ro_h = {}
    
    for fi,fam in enumerate(pf.symHKL):
        
        axis = np.cross(fam,v)
        angle = np.arccos(np.dot(fam,v))
        
        q0 = rowan.from_axis_angle(axis, angle)
        q0_n = rowan.normalize(q0) 
        
        q1_n = [rowan.normalize(rowan.from_axis_angle(h, omega)) for h in fam]
        
         # eu = np.zeros((len(omega),3,fam.shape[0]))
        eu = []
        
        for i,(qA,qB) in enumerate(zip(q0_n,q1_n)):
            
            qF = rowan.multiply(qA, qB)
            
            with np.errstate(divide='ignore'):

                temp = np.array((qF[:,1]/qF[:,0],qF[:,2]/qF[:,0],qF[:,3]/qF[:,0])).T
                ax = ro2ax(temp)
                om = ax2om(ax)
                eu.append(om2eu(om))
                
        ro_h[fi] = np.vstack(eu)
        
    ro_y[yi] = ro_h
    yi += 1
    
# %% 
            
#### FIBER PLOT ###

pf_num = 0

import mayavi.mlab as mlab
 # import matplotlib.pyplot as plt
 # fig = plt.figure()

mlab.figure(bgcolor=(1,1,1))

 ## grid ##
gd = mlab.points3d(od.phi1,od.Phi,od.phi2,scale_factor=1,mode='point',color=(0,0,0))
gd.actor.property.render_points_as_spheres = True
gd.actor.property.point_size = 5
  
 ## lit point ##
gd = mlab.points3d(0,0,0,scale_factor=1,mode='point',color=(1,0,0))
gd.actor.property.render_points_as_spheres = True
gd.actor.property.point_size = 5

 ## manual fibre ##

gd2 = mlab.points3d(0,0,0,scale_factor=1,mode='point',color=(0,1,0))
gd2.actor.property.render_points_as_spheres = True
gd2.actor.property.point_size = 5
    
plt_list = list(pf_od[pf_num].keys())
plt_list.sort()

@mlab.animate(delay=200)
def anim():
    while True:
        
        for pf_cell in plt_list:
            
             # plt.pause(0.05)
             # plt.cla()
            
            od_cells = np.array(pf_od[pf_num][pf_cell])      
            od_idx = np.unravel_index(od_cells.astype(int),od.bungeList.shape)
            
            gd.mlab_source.reset( x = od.phi1cen[od_idx],
                                  y = od.Phicen[od_idx],
                                  z = od.phi2cen[od_idx] )
            
             #limit to fund zone
            temp = ro_y[pf_cell][pf_num]
            temp = temp[temp[:,1] <= np.pi/2]
            temp = temp[temp[:,2] <= np.pi/2]
            
            gd2.mlab_source.reset( x = temp[:,0],
                                    y = temp[:,1],
                                    z = temp[:,2])
            
             # # generate PF central grid point to plot
             # pf_pts = np.divmod(pf_cell,pf_grid.shape[1])
             # pf_pts = np.multiply(pf_pts,np.deg2rad(5))
            
             # # determine corresponding OD cells to the given pf-cell
             # p_od = np.nonzero(pt[:,int(pf_cell)])
             # plt_od = np.vstack([pol[pf_num][:,:,v] for v in p_od[0]])
            
             # ax = fig.add_subplot(111,projection='polar')
             # ax.set_axisbelow(True)
             # #r
             # ax.set_ylim([0,np.pi/2+0.05])
             # ax.set_yticks(np.linspace(0,np.pi/2,19))
             # ax.set_yticklabels([])
             # #theta
             # ax.set_xticks(np.linspace(0,2*np.pi,73))
             # ax.set_xticklabels([])
            
             # ax.scatter(plt_od[:,0],plt_od[:,1],s=10,c='r')
             # ax.scatter(pf_pts[1]+np.deg2rad(2.5),pf_pts[0]+np.deg2rad(2.5),s=10,c='k')
   
            yield
            
anim()
mlab.show(stop=True)