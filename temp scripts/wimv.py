#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:09:43 2019

@author: nate
"""

"""
WIMV
"""

import os,sys

import numpy as np 
import pandas as pd 

import xrdtools

### user functions ###

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def pfgrid( alpha, beta ):

    """
    Return list of grid points on pole figure
    """
    
    k = 0
    
    pf = np.zeros((alpha.shape[0], alpha.shape[1]))
    
    for a in range(alpha.shape[0]):
        
        for b in range(alpha.shape[1]):
    
            pf[a,b] = k
            k += 1

    return pf 

def genSymHKL(hkl,symList):
    
    """
    generate symmetric equivalents
    
    only valid for cubic currently  
    """

    symList=[np.array([[1,0,0],[0,1,0],[0,0,1]]), #list of 24 matrix trans.
        np.array([[0,0,1],[1,0,0],[0,1,0]]),
        np.array([[0,1,0],[0,0,1],[1,0,0]]),
        np.array([[0,-1,0],[0,0,1],[-1,0,0]]),
        np.array([[0,-1,0],[0,0,-1],[1,0,0]]),
        np.array([[0,1,0],[0,0,-1],[-1,0,0]]),
        np.array([[0,0,-1],[1,0,0],[0,-1,0]]),
        np.array([[0,0,-1],[-1,0,0],[0,1,0]]),
        np.array([[0,0,1],[-1,0,0],[0,-1,0]]),
        np.array([[-1,0,0],[0,1,0],[0,0,-1]]),
        np.array([[-1,0,0],[0,-1,0],[0,0,1]]),
        np.array([[1,0,0],[0,-1,0],[0,0,-1]]),
        np.array([[0,0,-1],[0,-1,0],[-1,0,0]]),
        np.array([[0,0,1],[0,-1,0],[1,0,0]]),
        np.array([[0,0,1],[0,1,0],[-1,0,0]]),
        np.array([[0,0,-1],[0,1,0],[1,0,0]]),
        np.array([[-1,0,0],[0,0,-1],[0,-1,0]]),
        np.array([[1,0,0],[0,0,-1],[0,1,0]]),
        np.array([[1,0,0],[0,0,1],[0,-1,0]]),
        np.array([[-1,0,0],[0,0,1],[0,1,0]]),
        np.array([[0,-1,0],[-1,0,0],[0,0,-1]]),
        np.array([[0,1,0],[-1,0,0],[0,0,1]]),
        np.array([[0,1,0],[1,0,0],[0,0,-1]]),
        np.array([[0,-1,0],[1,0,0],[0,0,1]])]

    symHKL=np.asarray([np.dot(symList[symIdx],hkl) for symIdx in range(len(symList))])
    symHKL=np.unique(symHKL,axis=0)

    return symHKL

def eu2om(phi1,Phi,phi2): ##calcRotMatrix
    """
    input is meshgrid phi1,Phi,phi2
    
    output Nx3 ndarray | bunge angles for Nth cell
    output 3x3xN mdarray | g matrix for Nth cell
    
    g is crystal -> sample
    
    """

    c1=np.cos(phi1)
    c2=np.cos(phi2)
    c=np.cos(Phi)
    
    s1=np.sin(phi1)
    s2=np.sin(phi2)
    s=np.sin(Phi)
    
    g11=c1*c2-s1*c*s2
    g12=s1*c2+c1*s2*c
    g13=s2*s
    
    g21=-c1*s2-s1*c2*c
    g22=-s1*s2+c1*c2*c
    g23=c2*s
    
    g31=s1*s
    g32=-c1*s
    g33=c
    
    g = np.zeros((3,3,np.prod(Phi.shape)))
    bunge = np.zeros_like(Phi)
    k = 0
    
    for p1 in range(Phi.shape[0]):
        for p in range(Phi.shape[1]):
            for p2 in range(Phi.shape[2]):
                
                bunge[p1,p,p2] = k
                
                g[0,0,k] = g11[p1,p,p2]
                g[0,1,k] = g21[p1,p,p2]
                g[0,2,k] = g31[p1,p,p2]
                
                g[1,0,k] = g12[p1,p,p2]
                g[1,1,k] = g22[p1,p,p2]
                g[1,2,k] = g32[p1,p,p2]
                
                g[2,0,k] = g13[p1,p,p2]
                g[2,1,k] = g23[p1,p,p2]
                g[2,2,k] = g33[p1,p,p2]
                k += 1

    return g,bunge

def normVector(v):
    
    """
    normalize vectors
    
    input: Mx3xN ndarray | M - hkl multi. N - # of unique g
    
    """
    
    vN = np.zeros_like(v)

    norm = np.linalg.norm(v,axis=1)
    
    for n in range(v.shape[2]):
    
        vN[:,:,n] = v[:,:,n] / norm[:, n, None]
    
    temp = {}
    
    for j in range(vN.shape[2]):
        
        temp[j] = []
        
        for i in range(vN.shape[0]):
            
            if vN[i,2,j] >= 0:
                
                temp[j].append(vN[i,:,j])
                    
        temp[j] = np.vstack(temp[j])
    
    return temp

def XYZtoSPH(xyz):
    
    sph = {}
    
    for n,v in xyz.items():
    
        sph[n]=np.zeros((v.shape[0],2))
        xy=np.power(v[:,1],2) + np.power(v[:,0],2)
    
        #azimuth,elevation (2theta,eta) eta in range of 0 <-> pi from z axis
        
        for si,s in enumerate(sph[n]):
            
            if s[0] < 0: sph[n][si,0] += 2*np.pi
        
        sph[n][:,0]=np.arctan2(np.sqrt(xy), v[:,2]) #alpha
        sph[n][:,1]=np.arctan2(v[:,1], v[:,0]) #beta

    return sph   

### inputs ###

pf111path = '/home/nate/wimv/111pf_2T=38.xrdml'
pf200path = '/home/nate/wimv/200pf_2T=45.xrdml'
pf220path = '/home/nate/wimv/220pf_2theta=65.xrdml'

phi1max = 360
PhiMax = 90
phi2max = 90
odRes = 5

alphaMax = 90
betaMax = 357.5 #panalytical specific 2.5 -> 357.5
pfRes = 5

expAlphaMax = (80/5)+1

hkl = []
hkl.append([1,1,1])
hkl.append([2,0,0])
hkl.append([2,2,0])

symHKL = [genSymHKL(h,None) for h in hkl]

### computation ###

blockPrint()
expPFs = {}
expPFs[0] = xrdtools.read_xrdml(pf111path) 
expPFs[1] = xrdtools.read_xrdml(pf200path)
expPFs[2] = xrdtools.read_xrdml(pf220path)
enablePrint()

phi1 = np.deg2rad(np.arange(0,phi1max+odRes,odRes))
Phi = np.deg2rad(np.arange(0,PhiMax+odRes,odRes))
phi2 = np.deg2rad(np.arange(0,phi2max+odRes,odRes))

alpha = np.deg2rad(np.arange(0,alphaMax+pfRes,pfRes))
beta = np.deg2rad(np.arange(2.5,betaMax+pfRes,pfRes))

pfRes = np.deg2rad(pfRes) #for later use during computation
odRes = np.deg2rad(odRes)

[phi1,Phi,phi2] = np.meshgrid(phi1,Phi,phi2,indexing='ij')
[alpha,beta] = np.meshgrid(alpha,beta,indexing='ij')

pf = pfgrid(alpha,beta)

g, bunge = eu2om(phi1,Phi,phi2)

pol = {}
xyz = {}   
pointer = np.zeros((bunge.shape[0]*bunge.shape[1]*bunge.shape[2],alpha.shape[0]*alpha.shape[1],len(symHKL)))

for fi,fam in enumerate(symHKL):
    
    # Mx3xN array | M - hkl multi. N - # of unique g
    xyz[fi] = np.dot(fam,g)
    # normalize
    xyz[fi] = normVector(xyz[fi])
    # project (stereographic)
    pol[fi] = XYZtoSPH(xyz[fi])
    
    for b_cell in range(bunge.shape[0]*bunge.shape[1]*bunge.shape[2]):
            
        ai = pol[fi][b_cell][:,0] // pfRes #alpha index
        bi = pol[fi][b_cell][:,1] // pfRes #beta index
        
        #check for over 2pi
        bi = np.where(bi>=72, 0, bi)
        
        pi = pf[ai.astype(int),bi.astype(int)] #pole figure index
        
        if len(pi) == 0:
            print('oh no!')
        
        pointer[b_cell,pi.astype(int),fi] += 1
        
# %%
        
### initial WIMV iteration ###
        
od = np.zeros((bunge.shape[0]*bunge.shape[1]*bunge.shape[2]))
cellVolume = np.zeros_like(od)
        
for b_cell in range(bunge.shape[0]*bunge.shape[1]*bunge.shape[2]):
    
    tempOD = np.zeros(3)
    
    for fi in range(3):
        
        #These are the pole figure cells that correspond to this particular Bunge cell
        pf_pts = np.nonzero(pointer[b_cell,:,fi])[0]
        
        #With these cell numbers,
        ai,bi = np.divmod(pf_pts,beta.shape[1])
        
        #check if outside of exp range
        mask = ai < expAlphaMax
        ai = ai[mask]
        bi = bi[mask]
        
        pf_int = expPFs[fi]['data'][ai.astype(int),bi.astype(int)]
        
        #inner product | all hkls in pole family for a given OD cell
        temp_int = [pi**(1/(len(symHKL)*len(symHKL[fi]))) for pi in pf_int]
        tempOD[fi] = np.product(temp_int)
        
    #calculate OD cell indices
    temp = phi1.shape[0]*phi1.shape[1] #size of phi1/Phi grid
    p2id,re = np.divmod(b_cell,temp) #remainder is in phi1/Phi grid
    p1id,Pid = np.divmod(re,Phi.shape[1]) #remainder is Phi index
    
    # dg
    cellVolume[b_cell] = odRes*odRes*( np.cos( Pid*odRes - (odRes/2) ) - np.cos( Pid*odRes + (odRes/2) ) )
    # outer product | all pole figures
    od[b_cell] = cellVolume[b_cell]*np.product( tempOD )
        
# normalize to 1
od = (1/np.sum(od)) * od

#### recalculate ###

recalc_pf = np.zeros((pf.shape[0],pf.shape[1],len(symHKL)))
pf_cell = 0

for pf_ai in range(pf.shape[0]):
    
    for pf_bi in range(pf.shape[1]):
    
        for fi in range(len(symHKL)):
            
            #return non-zero values from pointer matrix
            OD_PFPts = np.nonzero(pointer[:,pf_cell,fi])[0]
            
            if len(OD_PFPts) == 0: recalc_pf[pf_ai,pf_bi,fi] = 0
            else:
                #calculate PF intensites from OD
                pfCellVolume = pfRes*pfRes*( np.cos( pf_ai*pfRes - (pfRes/2) )- np.cos( pf_ai*pfRes + (pfRes/2) ))
                recalc_pf[pf_ai,pf_bi,fi] = (1/len(OD_PFPts))*np.sum(od[OD_PFPts.astype(int)])*pfCellVolume
            
        pf_cell += 1   

# %% 

## OD vis ##

#import mayavi.mlab as mlab
#
#k = 0
#
#bunge2 = np.zeros((bunge.shape[0]*bunge.shape[1]*bunge.shape[2],3))
#
#for p2 in range(Phi.shape[2]):
#    for p in range(Phi.shape[1]):
#        for p1 in range(Phi.shape[0]):
#            
#            bunge2[k,0] = p1*odRes
#            bunge2[k,1] = p*odRes
#            bunge2[k,2] = p2*odRes
#            k += 1
#
#mlab.figure(bgcolor=(1,1,1))
#
#pts = mlab.points3d(bunge2[:,0], bunge2[:,1], bunge2[:,2], od, scale_factor=100, colormap="copper")
# 
#mlab.show(stop=True)
    
# %%

## PF vis ##
        
test = [v for k,v in pol[0].items()]
test = np.vstack(test)
        
test = np.zeros(pf.shape[0]*pf.shape[1])

for pf_cell in range(pf.shape[0]*pf.shape[1]):
     
    test[pf_cell] = np.count_nonzero(pointer[:,pf_cell,0])

import matplotlib.pyplot as plt

k = 0

point_plt = np.zeros_like(pf)

for ai in range(pf.shape[0]):
    
    for bi in range(pf.shape[1]):
        
        point_plt[ai, bi] = test[k]
        k += 1

fig = plt.figure()

ax = fig.add_subplot(111, projection='polar')
ax.set_rlim([0,np.pi/2]) 
ax.contour(beta, alpha, recalc_pf[:,:,1]) 
#ax.scatter(test[:,0],test[:,1])   

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='polar')
ax2.set_rlim([0,np.pi/2])

ax2.contour(beta[:17,:],alpha[:17,:], expPFs[1]['data']) 
    

# %% vis
    

#import mayavi.mlab as mlab
#
#mlab.figure(bgcolor=(1, 1, 1))
#
#green = (0.039, 1, 0.054)
#red = (1, 0.039, 0.039)
#blue = (0.039, 0.160, 1)
#
#mlab.quiver3d(0,0,0,1,0,0,line_width=0.5,scale_factor=0.5,color=red)
#mlab.quiver3d(0,0,0,0,1,0,line_width=0.5,scale_factor=0.5,color=green)
#mlab.quiver3d(0,0,0,0,0,1,line_width=0.5,scale_factor=0.5,color=blue)
#
#pts = mlab.points3d(xyz[:,0,0],xyz[:,1,0],xyz[:,2,0], scale_factor=1, mode='point', color=(1, 0.039, 0.039))
#pts.actor.property.render_points_as_spheres = True
#pts.actor.property.point_size = 12
#
##e1 = mlab.quiver3d(0,0,0,g[0,0,1],g[1,0,1],g[2,0,1],line_width=0.5,scale_factor=0.5,color=(0,0,0))
##e2 = mlab.quiver3d(0,0,0,g[0,1,1],g[1,1,1],g[2,1,1],line_width=0.5,scale_factor=0.5,color=(0,0,0))
##e3 = mlab.quiver3d(0,0,0,g[0,2,1],g[1,2,1],g[2,2,1],line_width=0.5,scale_factor=0.5,color=(0,0,0))
#
##Now animate the data.
#@mlab.animate(delay=500)
#def anim():
#    while True:
#        for n in range(xyz.shape[2]):
#                        
#            pts.mlab_source.reset(x=xyz[:,0,n],y=xyz[:,1,n],z=xyz[:,2,n])
#            
##            e1.mlab_source.reset(u=g[0,0,n],v=g[1,0,n],w=g[2,0,n])
##            e2.mlab_source.reset(u=g[0,1,n],v=g[1,1,n],w=g[2,1,n])
##            e3.mlab_source.reset(u=g[0,2,n],v=g[1,2,n],w=g[2,2,n])
#            
#            yield
#
#anim()
#mlab.show(stop=True)
#
# %%

#import matplotlib.pyplot as plt
#
#fig = plt.figure()
#
##ax = fig.add_subplot(111, projection='polar')
##ax.set_rlim([0,np.pi/2])
##ax.scatter(pol[:,0,171], pol[:,1,171])
#
#for n,v in pol.items():
#
#    ax = fig.add_subplot(111, projection='polar')
#    ax.set_rlim([0,np.pi/2])
#    ax.scatter(v[:,0], v[:,1])
#    plt.pause(0.25)
#    plt.cla()