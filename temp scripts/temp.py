# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0,'/home/nate/wimv')

from classes import poleFigure, bunge
from utils.orientation import symmetrise, normalize, XYZtoSPH

crystalSym = 'm-3m'
sampleSym = '1'
cellSize = np.deg2rad(5)

pf111path = '/home/nate/wimv/111pf_2T=38.xrdml'
pf200path = '/home/nate/wimv/200pf_2T=45.xrdml'
pf220path = '/home/nate/wimv/220pf_2theta=65.xrdml'

pfs = [pf111path,pf200path,pf220path]

hkls = [(1,1,1),(2,0,0),(2,2,0)]

pf = poleFigure(pfs, hkls, crystalSym, 'xrdml')
od = bunge(cellSize, crystalSym, sampleSym)

## ##

hkls = np.array(hkls)

symHKL = symmetrise(crystalSym, hkls)
symHKL = normalize(symHKL)

pf_grid, alp, bet = pf.grid(full=True, ret_ab=True)

xyz = {}   
pol = {}
pointer = np.zeros((od.bungeList.shape[0]*od.bungeList.shape[1]*od.bungeList.shape[2],
                    pf_grid.shape[0]*pf_grid.shape[1],
                    len(symHKL)))

a_bins = np.histogram_bin_edges(np.arange(0,np.pi/2+pf.resolution,pf.resolution),18)
b_bins = np.histogram_bin_edges(np.arange(0,2*np.pi+pf.resolution,pf.resolution),72)

for fi,fam in enumerate(symHKL):
    
    # Mx3xN array | M - hkl multi. N - # of unique g
    xyz[fi] = np.dot(fam,od.g)

    # stereographic projection
    pol[fi] = XYZtoSPH(xyz[fi])
    
    for b_cell in np.ravel(od.bungeList):
            
#        ai = pol[fi][:,1,int(b_cell)] // pf.resolution #alpha index
#        bi = pol[fi][:,0,int(b_cell)] // pf.resolution #beta index
        
        """
        use histogram_bin_edges for bins
        then searchsorted to find indices
        """
        
        ai = np.digitize(pol[fi][:,1,int(b_cell)],a_bins) - 1
        bi = np.digitize(pol[fi][:,0,int(b_cell)],b_bins) - 1
        
#        if len(np.where(ai<0)[0]) > 0: print(ai)
#        #check for over 2pi
#        bi = np.where(bi>=72, 0, bi)
        
#        if len(np.where(bi>=72)[0]) > 0: 
#            print('over')
#            print(np.rad2deg(np.max(pol[fi][:,0,int(b_cell)])))
        
        pi = pf_grid[ai.astype(int),bi.astype(int)] #pole figure index
        
        if len(pi) == 0:
            print('oh no!')
        
        pointer[int(b_cell),pi.astype(int),fi] += 1
        
#od_data = np.ones( od.bungeList.shape[0]*od.bungeList.shape[1]*od.bungeList.shape[2] )
#        
#for pf_cell in np.ravel(pf_grid):
#    
#    ai, bi = np.divmod(pf_cell, pf_grid.shape[1])
#        
#    for fi in range(len(symHKL)):
#        
#        od_cells = np.nonzero(pointer[:,int(pf_cell),fi])
#        
#        try:
#            od_data[od_cells] *= pf.data[fi][int(ai),int(bi)]**(1/(len(symHKL)*len(symHKL[fi])))
#        except:
#            pass    
#            
##normalize to volume
#od_idx = np.unravel_index(np.ravel(od.bungeList).astype(int),od.bungeList.shape)
#    
##calculate cell volume for distortion
#Phi = od.Phi[od_idx]
#od_cellVol = od.res*od.res*( np.cos( Phi - (od.res/2) ) - np.cos( Phi + (od.res/2) ) )
#
## f(g)dg
#od_data = od_data * od_cellVol 
#
## N
#od_norm = sum(od_cellVol) / sum(od_data*od_cellVol)
#
#od_dataN = od_norm * od_data


# %%
### load matlab pointer ###

import scipy.io as scio

pt_mat = scio.loadmat('pointer2.mat')['pointer']
pfCell_mat = list(scio.loadmat('pfCell.mat')['pfCell'])
pfCell_mat = np.vstack(pfCell_mat).T

pfAngle = scio.loadmat('pfAngle.mat')['pfAngle']

import matplotlib.pyplot as plt

fig = plt.figure()

for fi in range(len(symHKL)):
    
    for od_cell in np.ravel(od.bungeList):
        
        mat_cells = pt_mat[0][fi][0][int(od_cell)][0]   
        py_cells = np.nonzero(pointer[int(od_cell),:,fi])[0]
        
        py_idx = np.unravel_index(py_cells.astype(int),pf_grid.shape)
        
        mat_cells.sort()
        py_cells.sort()
#        
#        mat_bunge = pfCell_mat[mat_cells]
#        py_bunge = np.array((alp[py_idx],np.deg2rad(bet[py_idx]))).T
#        
#        #check if diff
#        if np.array_equal(mat_bunge,py_bunge) is False:
##            
##            mat_plt = np.vstack(list(pfAngle[int(od_cell)][fi][0]))
##            
##            mat_plt[np.where(mat_bunge != py_bunge)]
#
#            ax = fig.add_subplot(111, projection='polar')
#            ax.set_axisbelow(True)
#            
#            ax.set_ylim([0,np.pi/2+0.05])
#            ax.set_yticks(np.linspace(0,np.pi/2,19))
#            ax.set_yticklabels([])
#            #theta
#            ax.set_xticks(np.linspace(0,2*np.pi,73))
#            ax.set_xticklabels([])
#    #
#            ax.scatter(mat_bunge[:,1],mat_bunge[:,0],c='r',s=8)
#            ax.scatter(py_bunge[:,1],py_bunge[:,0],c='g',s=8)
#            ax.scatter(pol[fi][:,0,int(od_cell)],pol[fi][:,1,int(od_cell)],c='k',s=8)
#    #        
#            plt.pause(0.75)
#            plt.cla()


# %%


#recalc pole figure initial
#recalc_data = {}
#recalc_pf = {}
#recalc_data[0] = np.zeros((pf_grid.shape[0],pf_grid.shape[1],len(symHKL)))

#recalc_data = np.zeros((pf_grid.shape[0],pf_grid.shape[1],len(symHKL)))
#
#for pf_cell in np.ravel(pf_grid):
#    
#    for fi in range(len(symHKL)):
#        
#        pf_idx = np.unravel_index(int(pf_cell),pf_grid.shape)
#        od_cells = np.nonzero(pointer[:,int(pf_cell),fi])
#        
#        if len(od_cells[0]) > 0: recalc_data[pf_idx[0],pf_idx[1],fi] = 1/len(od_data[od_cells]) * np.sum(od_data[od_cells])
#        else: recalc_data[pf_idx[0],pf_idx[1],fi] = 0
#        
## pole figure normalization
#pf_idx = np.unravel_index(np.ravel(pf_grid).astype(int),pf_grid.shape)
#Alpha = alp[pf_idx]
#pf_cellVol = pf.resolution * pf.resolution * ( np.cos( Alpha - (pf.resolution/2) ) - np.cos( Alpha + (pf.resolution/2) ) )
#
#recalc_pf = np.zeros_like(recalc_data)
##
#for fi in range(len(symHKL)):
##    
#    temp = np.ravel(recalc_data[:,:,fi])
#    pf_norm = ( sum(pf_cellVol) / sum(temp*pf_cellVol) ) 
#    print(pf_norm)
#    pf_cellN = pf_norm * temp * pf_cellVol
#    
#    recalc_pf[pf_idx[0],pf_idx[1],fi] = pf_cellN
#    

"""
compare values with Mtex

"""


#test = poleFigure(recalc_data,hkls,'recalc',resolution=5)
#test.plot()

#for ai in range(pf_grid.shape[0]):
#    
#    for bi in range(pf_grid.shape[1]):
#        
#        for fi in range(len(symHKL)):
#        
#            pf_cell = pf_grid[ai,bi]
#            
#            od_cells = np.nonzero(pointer[:,int(pf_cell),fi])
#            
#            recalc_data[0][ai,bi,fi] += np.sum(od_data[od_cells])
#            
#recalc_pf[0] = poleFigure(recalc_data[0],hkls,'recalc',resolution=5)
#
#iterations = 5
#
#od_data = np.ones( (od.bungeList.shape[0]*od.bungeList.shape[1]*od.bungeList.shape[2],iterations) )

### start iterations ####

#for i in range(iterations):
#    
#    ### calculate OD_cell ###
#    for ai in range(pf.alpha.shape[0]):
#        
#        for bi in range(pf.alpha.shape[1]):
#            
#            for fi in range(len(symHKL)):
#            
#                pf_cell = pf_grid[ai,bi]
#                
#                od_cells = np.nonzero(pointer[:,int(pf_cell),fi])
#                
#                od_data[od_cells,i] *= (pf.data[fi][ai,bi]/recalc_pf[i].data[fi][ai,bi])**(1/(len(symHKL)*len(symHKL[fi])))
#                
#    #normalize to volume
#    for p1i in range( od.bungeList.shape[0] ):
#        
#        for pi in range( od.bungeList.shape[1] ):
#            
#            for p2i in range( od.bungeList.shape[2] ):
#                
#                od_cell = od.bungeList[p1i,pi,p2i]
#                
#                #calculate cell volume for distortion
#                Phi = pi*od.res
#                cellVolume = od.res*od.res*( np.cos( pi*od.res - (od.res/2) ) - np.cos( pi*od.res + (od.res/2) ) )
#                
#                od_data[int(od_cell),i] *= cellVolume
#                
#    #recalc pole figure initial
#    recalc_data[i+1] = np.zeros((pf_grid.shape[0],pf_grid.shape[1],len(symHKL)))
#    
#    for ai in range(pf_grid.shape[0]):
#        
#        for bi in range(pf_grid.shape[1]):
#            
#            for fi in range(len(symHKL)):
#            
#                pf_cell = pf_grid[ai,bi]
#                
#                od_cells = np.nonzero(pointer[:,int(pf_cell),fi])
#                
#                recalc_data[i+1][ai,bi,fi] += np.sum(od_data[od_cells,i])
#                
#    recalc_pf[i+1] = poleFigure(recalc_data[i+1],hkls,'recalc',resolution=5)
##    recalc_pf.plot()
#    print('done with iteration')
#            
    
            

# %%

#pf_num = 0
#pt = pointer[:,:,pf_num]
#
#import matplotlib.pyplot as plt
#fig = plt.figure()

 ### PF Cell Loop ###

#for pf_cell_num in range(pf_grid.shape[0]*pf_grid.shape[1]):
#
#     # generate PF central grid point to plot
#    pf_pts = np.divmod(pf_cell_num,pf_grid.shape[1])
#    pf_pts = np.multiply(pf_pts,np.deg2rad(5))
#    
#     # determine corresponding OD cells to the given pf-cell
#    p_od = np.nonzero(pt[:,pf_cell_num])
#    plt_od = np.vstack([pol[pf_num][:,:,v] for v in p_od[0]])
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
#    ax.scatter(plt_od[:,0],plt_od[:,1],s=10,c='r')
#    ax.scatter(pf_pts[1]+np.deg2rad(2.5),pf_pts[0]+np.deg2rad(2.5),s=10,c='k')
#    plt.pause(0.05)
#    plt.cla()

##### OD Cell Loop ####

#for b_cell_num in range(od.bungeList.shape[0]*od.bungeList.shape[1]*od.bungeList.shape[2]):
#
#    # determine corresponding PF cells to given OD-cell
#    p_pf = np.nonzero(pt[b_cell_num,:])
#    plt_pf_alpha, plt_pf_beta = np.divmod(p_pf,pf_grid.shape[1])
#    
##    # generate PF central grid point to plot
##    pf_pts = np.divmod(pf_cell_num,pf_grid.shape[1])
##    pf_pts = np.multiply(pf_pts,np.deg2rad(5))
##    
##    # determine corresponding OD cells to the given pf-cell
##    p_od = np.nonzero(pt[:,pf_cell_num])
##    plt_od = np.vstack([pol[0][v] for v in p_od[0]])
#    
#    ax = fig.add_subplot(111,projection='polar')
#    ax.set_axisbelow(True)
#    #r
#    ax.set_ylim([0,np.pi/2+0.05])
#    ax.set_yticks(np.linspace(0,np.pi/2,19))
#    ax.set_yticklabels([])
#    #theta
#    ax.set_xticks(np.linspace(0,2*np.pi,73))
#    ax.set_xticklabels([])
#    
#    ax.scatter(pol[pf_num][:,0,int(b_cell_num)],pol[pf_num][:,1,int(b_cell_num)],c='k',s=8)
#    ax.scatter(plt_pf_beta*np.deg2rad(5)+np.deg2rad(2.5),plt_pf_alpha*np.deg2rad(5)+np.deg2rad(2.5),s=10,c='g')
#    plt.pause(0.15)
#    plt.cla()

# %%
            
### FIBER CALC ###

import rowan
from tqdm import tqdm

from utils.orientation import ro2ax, ax2om, om2eu

 #calculate pole figure y's

sph = np.array(np.divmod(np.ravel(pf_grid),pf_grid.shape[1])).T
sph = sph * pf.resolution

 #convert to xyz
pf_xyz = np.zeros((sph.shape[0],3))

pf_xyz[:,0] = np.sin( sph[:,0] ) * np.cos( sph[:,1] )
pf_xyz[:,1] = np.sin( sph[:,0] ) * np.sin( sph[:,1] )
pf_xyz[:,2] = np.cos( sph[:,0] )

pf_num = 0

omega = np.radians(np.arange(-180,180,5))
ro_y = {}
yi = 0

for v in tqdm(pf_xyz):
    
    ro_h = {}
    
    for fi,fam in enumerate(symHKL):
        
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
            
### FIBER PLOT ###

pf_num = 0
pt = pointer[:,:,pf_num]

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

@mlab.animate(delay=200)
def anim():
    while True:
        
        for pf_cell in np.ravel(pf_grid):
            
             # plt.pause(0.05)
             # plt.cla()
                
            od_cells = np.nonzero(pointer[:,int(pf_cell),pf_num])       
            od_idx = np.unravel_index(od_cells,od.bungeList.shape)
            
            gd.mlab_source.reset( x = od.phi1[od_idx],
                                   y = od.Phi[od_idx],
                                   z = od.phi2[od_idx] )
            
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




# %%

# 3D
#reshape
    
#od_plt = np.zeros_like(od.bungeList)
#    
#for od_cell in np.ravel(od.bungeList):
#    
#    od_idx = np.unravel_index(int(od_cell),od.bungeList.shape)
#    od_plt[od_idx] = od_dataN[int(od_cell)]
#
#
#import mayavi.mlab as mlab
#
#mlab.figure(bgcolor=(1,1,1))
#
#l = mlab.points3d(od.phi1,od.Phi,od.phi2,od_plt, colormap="copper")
##l.actor.property.render_points_as_spheres = True
##l.actor.property.point_size = 10
#
#mlab.show(stop=True)
