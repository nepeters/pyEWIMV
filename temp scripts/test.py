#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 22:20:45 2019

@author: nate
"""

import numpy as np


crystalSym = 'm-3m'
sampleSym = '1'
cellSize = np.deg2rad(5)

# set boundary in Bunge space (not rigorous for cubic)
if sampleSym == '1': phi1max = np.deg2rad(360)
elif sampleSym == 'm': phi1max = np.deg2rad(180)
elif sampleSym == 'mmm': phi1max = np.deg2rad(90)
else: raise ValueError('invalid sampleSym')

if crystalSym == 'm-3m': 
    Phimax = np.deg2rad(90)
    phi2max = np.deg2rad(90)
elif crystalSym == 'm-3': raise NotImplementedError('coming soon..')
else: raise ValueError('invalid crystalSym, only cubic so far..')

# setup grid
phi1range = np.arange(0,phi1max+cellSize,cellSize)
Phirange = np.arange(0,Phimax+cellSize,cellSize)
phi2range = np.arange(0,phi2max+cellSize,cellSize)

phi2, Phi, phi1 = np.meshgrid(phi2range, Phirange, phi1range, indexing = 'ij')

eps = 9

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
b_list = np.zeros_like(Phi)
k = 0

angs = np.zeros((np.prod(Phi.shape),3))

for p2 in range(phi1.shape[0]):
    for p in range(phi1.shape[1]):
        for p1 in range(phi1.shape[2]):
            
            b_list[p2,p,p1] = k
            
            angs[k,0] = phi1[p2,p,p1]
            angs[k,1] = Phi[p2,p,p1]
            angs[k,2] = phi2[p2,p,p1]
            
            g[0,0,k] = g11[p2,p,p1]
            g[0,1,k] = g21[p2,p,p1]
            g[0,2,k] = g31[p2,p,p1]
            
            g[1,0,k] = g12[p2,p,p1]
            g[1,1,k] = g22[p2,p,p1]
            g[1,2,k] = g32[p2,p,p1]
            
            g[2,0,k] = g13[p2,p,p1]
            g[2,1,k] = g23[p2,p,p1]
            g[2,2,k] = g33[p2,p,p1]
            k += 1
            
g = np.round(g, decimals=eps)

c1=np.cos(angs[:,0])
c2=np.cos(angs[:,2])
c=np.cos(angs[:,1])

s1=np.sin(angs[:,0])
s2=np.sin(angs[:,2])
s=np.sin(angs[:,1])

g2 = np.empty((len(angs),9))

g2[:,0]=c1*c2-s1*c*s2
g2[:,1]=s1*c2+c1*c*s2
g2[:,2]=s*s2
g2[:,3]=-c1*s2-s1*c*c2
g2[:,4]=-s1*s2+c1*c*c2
g2[:,5]=s*c2
g2[:,6]=s1*s
g2[:,7]=-c1*s
g2[:,8]=c

for k in range(np.prod(Phi.shape)):
    
    temp = g2[k,:].reshape(3,3)
    
    if np.allclose(temp,g[:,:,k]) is False:
        
        raise ValueError

#""" store/weights with quat sym intrinsic distance """
#""" need to calculate misorient from center of cell  """
#""" pull this out and use MP, save new fib as well """
#
#""" reshape to Nx26353x4 """
#
#def fibre_dist(qfib,qgrid,hi):
#    
#    """
#    input is qfib dict (per pole)
#        key: yi
#        value: fibre thru space; yi || h
#    """
#    
#    fibDist = np.zeros((len(qfib),len(qgrid)))
#    
#    for yi,qf in qfib.items(): #looping by y
#        
#        qf = qf.T.reshape((1,4,len(qf)),order='F')
#        qf = np.tile(qf,(qgrid.shape[0],1,1))
#        qf = qf.transpose((2,0,1))
#        
#        temp = quat.geometry.sym_distance(qf,qgrid)
#        
#        """ store smallest value (misorient) for each cell """
#        fibDist_idx = (np.argmin(temp, axis=0),range(len(qgrid))) #index for each Euler (Bunge) cell
#        fibDist[yi,:] = temp[fibDist_idx]
#        
#    return fibDist
#        
#inputs = []
#
#for hi,h in enumerate(hkls):
#    
#    inputs.append((fibre_q[hi],qgrid,hi))
#
#print('starting pool')
#
#fibre_dist 
#
#if __name__ == '__main__':
#
#    results = pm.starmap(fibre_dist,inputs, pm_pbar=True, pm_processes=7)
#    
# 
#    
#print('done!')