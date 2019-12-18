#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:39:46 2019

@author: nate
"""


### Pulstec texture ###

import os,sys

import numpy as np
from math import cos, sin

sys.path.insert(0, '/home/nate/pyReducePF/functions')

from crystal import Xtal

def eu2om(angs, out=None): 
    """
    
    returns g matrix
    
    crystal -> sample
    
    input: 
        tuple of single angles (ω,ψ,φ)
        3N meshgrid in tuple (ω,ψ,φ)
        Nx3 array [:,(ω,ψ,φ)]
    out: 
        None     :single matrix
        'mdarray':3x3xN mdarray, N: #of (ω,ψ,φ)
        'ndarray':Nx9 array [:,(ω,ψ,φ)]
    """

    if out is None:

        co=np.cos(angs[0])
        cps=np.cos(angs[1])
        cph=np.cos(angs[2])

        so=np.sin(angs[0])
        sps=np.sin(angs[1])
        sph=np.sin(angs[2])

        g=np.zeros((3,3))

        g[0,0]=-co*sps*cph+so*sph
        g[0,1]=so*sps*cph+co*sph
        g[0,2]=-cps*cph
        g[1,0]=-co*sps*sph-so*cph
        g[1,1]=so*sps*sph-co*cph
        g[1,2]=-cps*sph
        g[2,0]=-co*cps
        g[2,1]=so*cps
        g[2,2]=sps
        
        return g
    
    if out == 'mdarray':
    
        co=np.cos(angs[0])
        cps=np.cos(angs[1])
        cph=np.cos(angs[2])

        so=np.sin(angs[0])
        sps=np.sin(angs[1])
        sph=np.sin(angs[2])
        
        g11=-co*sps*cph+so*sph
        g12=so*sps*cph+co*sph
        g13=-cps*co
        
        g21=-co*sps*sph-so*cph
        g22=so*sps*sph-co*cph
        g23=-cps*sph
        
        g31=-co*cps
        g32=so*cps
        g33=sps
        
        g = np.zeros((3,3,np.prod(angs[1].shape)))
        b_list = np.zeros_like(angs[1])
        k = 0
        
        for p1 in range(angs[1].shape[0]):
            for p in range(angs[1].shape[1]):
                for p2 in range(angs[1].shape[2]):
                    
                    b_list[p1,p,p2] = k
                    
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
                    
        return g,b_list

    elif out == 'ndarray':

        co=np.cos(angs[0])
        cps=np.cos(angs[1])
        cph=np.cos(angs[2])

        so=np.sin(angs[0])
        sps=np.sin(angs[1])
        sph=np.sin(angs[2])

        g = np.empty((len(angs),9))

        g[:,0]=-co*sps*cph+so*sph
        g[:,1]=so*sps*cph+co*sph
        g[:,2]=-cps*co
        g[:,3]=-co*sps*sph-so*cph
        g[:,4]=so*sps*sph-co*cph
        g[:,5]=-cps*sph
        g[:,6]=-co*cps
        g[:,7]=so*cps
        g[:,8]=sps
        
        return g
    
    else:
        raise NotImplementedError('out_type not recognized')

def XYZtoSPH(xyz, proj='stereo'):
     
    sph=np.zeros((xyz.shape[0],2))
    xy=np.power(xyz[:,1],2) + np.power(xyz[:,0],2)

    #azimuth,elevation (2theta,eta) eta in range of 0 <-> pi from z axis
    
    sph[:,0] = np.arctan2( np.sqrt(xy), xyz[:,2] ) #alpha
    sph[:,1] = np.arctan2( xyz[:,1], xyz[:,0] ) #beta
    
    if proj == 'stereo': sph[:,0] = 2*np.tan(sph[:,0]/2)
    elif proj == 'earea': sph[:,0] = 2*np.sin(sph[:,0]/2)

#    for si,s in enumerate(sph):
#        
#        if s[1] < 0: sph[si,1] += 2*np.pi

    return sph 

    
### program start ###

lmbda = 2.29
alpha_steps = 100

al = Xtal.cubic(a=4.046,lattice_type='F')
hkl = al.d_spacing(dmin=1.145)

#extract d-spacing
d = np.array([float(k) for k,v in hkl.items()])

#theta calculation
theta = np.arcsin(lmbda/(2*d))

#eta
eta = (np.pi/2) - theta
#alpha
alpha = np.linspace(0,2*np.pi,alpha_steps)

"""
from Ramirez-Rico et al. (2016)

qL - diff vector in inst.
qS - diff vector in samp.

o - e1 rot (out of plane sample rotate)
s - e2 rot (tilt axis)
p - e3 rot (in-plane sample rotate)
"""

o = np.deg2rad([0])
ps = np.deg2rad([-30,-15,0,15,30])
ph = np.deg2rad([0,90])

sample_rot = []

for i in o:

    for j in ps:
        
        for k in ph:
            
            sample_rot.append((i,j,k))

qL = {}
qS = {}
qS_pol = {}

for i,R in enumerate(sample_rot):
    
    qL[i] = {}
    qS[i] = {} 
    qS_pol[i] = {}

    for e,k in zip(eta,hkl.keys()):
        
        qL[i][hkl[k][-1]] = []
        qS[i][hkl[k][-1]] = []
        
        for a in alpha:
            
            # qL #
            temp = np.array((-cos(e),-sin(e)*sin(a),-sin(e)*cos(a)))
            temp = temp / np.linalg.norm(temp)
            qL[i][hkl[k][-1]].append(temp)
            
            ps = R[1]
            ph = R[2]
            
            # qS #
            qS1 = cos(e)*sin(ps)*cos(ph) + sin(e)*cos(ps)*cos(ph)*cos(a) - sin(e)*sin(ph)*sin(a)
            qS2 = cos(e)*sin(ps)*sin(ph) + sin(e)*cos(ps)*sin(ph)*cos(a) + sin(e)*cos(ph)*sin(a)
            qS3 = cos(e)*cos(ps) - sin(e)*sin(ps)*cos(a)
            
            qS[i][hkl[k][-1]].append(np.array((qS1,qS2,qS3)))
            
        qL[i][hkl[k][-1]] = np.vstack(qL[i][hkl[k][-1]])
        qS[i][hkl[k][-1]] = np.vstack(qS[i][hkl[k][-1]])
        
        qS_pol[i][hkl[k][-1]] = XYZtoSPH(qS[i][hkl[k][-1]])
    
# %%
        
import matplotlib.pyplot as plt

colors = plt.cm.viridis(np.linspace(0,1,len(sample_rot)))

fig = plt.figure()

ax = fig.add_subplot( 111, projection = 'polar' )
ax.set_rlim([0,np.pi/2])
ax.set_yticklabels([])
ax.set_xticklabels([])

for i,R in enumerate(sample_rot):

    ax.scatter(qS_pol[i][(2,2,2)][:,1], qS_pol[i][(2,2,2)][:,0],s=2,c=colors[i,:3],label=str(i))

plt.legend()












 
