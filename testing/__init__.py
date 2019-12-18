#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:28:24 2019

@author: nate
"""

"""
orientation module
"""

import numpy as np

__all__ = ['eu2om',
           'om2eu',
           'ro2ho',
           'ro2ax',
           'ho2ro',
           'ax2om']

### Euler (Bunge) space ###

def eu2om(bunge, out=None): 
    """
    
    returns g matrix
    
    crystal -> sample
    
    input: 
        tuple of single angles (φ1,Φ,φ2)
        3N meshgrid in tuple (φ1,Φ,φ2)
        Nx3 array [:,(φ1,Φ,φ2)]
    out: 
        None     :single matrix
        'mdarray':3x3xN mdarray, N: #of (φ1,Φ,φ2)
        'ndarray':Nx9 array [:,(φ1,Φ,φ2)]
    """

    if out is None:

        c1=np.cos(bunge[0])
        c2=np.cos(bunge[2])
        c=np.cos(bunge[1])

        s1=np.sin(bunge[0])
        s2=np.sin(bunge[2])
        s=np.sin(bunge[1])

        g=np.zeros((3,3))

        g[0,0]=c1*c2-s1*c*s2
        g[0,1]=s1*c2+c1*c*s2
        g[0,2]=s*s2
        g[1,0]=-c1*s2-s1*c*c2
        g[1,1]=-s1*s2+c1*c*c2
        g[1,2]=s*c2
        g[2,0]=s1*s
        g[2,1]=-c1*s
        g[2,2]=c
    
    if out == 'mdarray':
    
        c1=np.cos(bunge[0])
        c2=np.cos(bunge[2])
        c=np.cos(bunge[1])
        
        s1=np.sin(bunge[0])
        s2=np.sin(bunge[2])
        s=np.sin(bunge[1])
        
        g11=c1*c2-s1*c*s2
        g12=s1*c2+c1*s2*c
        g13=s2*s
        
        g21=-c1*s2-s1*c2*c
        g22=-s1*s2+c1*c2*c
        g23=c2*s
        
        g31=s1*s
        g32=-c1*s
        g33=c
        
        g = np.zeros((3,3,np.prod(bunge[1].shape)))
        b_list = np.zeros_like(bunge[1])
        k = 0
        
        for p2 in range(bunge[1].shape[2]):
            for p in range(bunge[1].shape[1]):
                for p1 in range(bunge[1].shape[0]):
                    
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
                    
        return g,b_list

    elif out == 'ndarray':

        c1=np.cos(bunge[:,0])
        c2=np.cos(bunge[:,2])
        c=np.cos(bunge[:,1])

        s1=np.sin(bunge[:,0])
        s2=np.sin(bunge[:,2])
        s=np.sin(bunge[:,1])

        g = np.empty((len(bunge),9))

        g[:,0]=c1*c2-s1*c*s2
        g[:,1]=s1*c2+c1*c*s2
        g[:,2]=s*s2
        g[:,3]=-c1*s2-s1*c*c2
        g[:,4]=-s1*s2+c1*c*c2
        g[:,5]=s*c2
        g[:,6]=s1*s
        g[:,7]=-c1*s
        g[:,8]=c

        return g
    
    else:
        raise NotImplementedError('out_type not recognized')

### orientation matrices ###

def om2eu(g): ##calcEuler
    """input is 3x3 array rotation matrix"""

    if g.shape == (3,3):

        phi=np.arccos(g[2,2])
        if phi==0:
            phi1=np.arctan2(-g[1,0],g[0,0])
            phi2=0
        elif phi==np.pi:
            phi1=np.arctan2(g[1,0],g[0,0])
            phi2=0
        else:
            phi1=np.arctan2(g[2,0],-g[2,1])
            phi2=np.arctan2(g[0,2],g[1,2])
        if phi1 < 0:
            phi1 += 2*np.pi
        if phi2 < 0:
            phi2 += 2*np.pi

        return [phi1,phi,phi2]

    else:

        phi1 = []
        phi = []
        phi2 = []

        for om in g:

            if om[8] == 1:

                t1 = np.arctan2(om[1],om[0])

                if t1 < 0:
                    t1 += 2*np.pi

                phi1.append(t1)
                phi.append((np.pi/2)*(1-om[8]))
                phi2.append(0)
#                bunge.append(np.array((np.arctan2(g[1],g[0]),(np.pi/2)*(1-g[8]),0)))

            else:

                xi = 1/np.sqrt(1-om[8]**2)

                t1 = np.arctan2(om[6]*xi,-om[7]*xi)
                t2 = np.arctan2(om[2]*xi,om[5]*xi)

                if t1 < 0:
                    t1 += 2*np.pi
                if t2 < 0:
                    t2 += 2*np.pi

                phi1.append(t1)
                phi.append(np.arccos(om[8]))
                phi2.append(t2)

#                bunge.append(np.array((np.arctan2(g[6]*xi,-g[7]*xi),np.arccos(g[8]),np.arctan2(g[2]*xi,g[5]*xi))))

        return np.column_stack((phi1,phi,phi2))

### rodrigues vectors ###

def ro2ho(ro):

    p = np.linalg.norm(ro,axis=1)

    ho = []

    for r,l in zip(ro,p):

        if l == 0:

            ho.append(np.array((0,0,0)))

        elif l == np.inf:

            f = ((3*np.pi)/4)
            ho.append((r/l)*f**(1/3))

        else:

            w = 2*np.arctan(l)
            f = 3*(w-np.sin(w))/4
            ho.append((r/l)*f**(1/3))

    return np.vstack(ho)

def ro2ax(ro):

    p = np.linalg.norm(ro,axis=1)

    with np.errstate(invalid='ignore'):
        n = ro/p[:,np.newaxis]

    n = np.nan_to_num(n) #divide by zero error

    ang = 2*np.arctan(p)

    return np.column_stack((n,ang))

### homochoric/cubochoric ###

def gammaFunc(h,i):

    gamma = np.array(([1.0000000000018852, -0.5000000002194847,
                   -0.024999992127593126, - 0.003928701544781374,
                   -0.0008152701535450438, - 0.0002009500426119712,
                   -0.00002397986776071756, - 0.00008202868926605841,
                   0.00012448715042090092, - 0.0001749114214822577,
                   0.0001703481934140054, - 0.00012062065004116828,
                   0.000059719705868660826, - 0.00001980756723965647,
                   0.000003953714684212874, - 0.00000036555001439719544]))

    return gamma[i]*h**(i) ##h is magnitude of vector |h|

def ho2ro(hc):

    thr = 1E-7

    hc_mag = np.linalg.norm(hc,axis=1)**2

    hPrime = []
    s = []
    axAngPair = []

    for idx,mag in enumerate(hc_mag):

        if mag == 0:

            axAngPair.append(np.array((0,0,1,0)))

        else:

            hPrime = hc[idx,:] / np.sqrt(mag)
            s = np.asarray([gammaFunc(mag,i) for i in range(16)]).sum()

            ang = 2*np.arccos(s)

            if abs(ang-np.pi) < thr:
                ang = np.pi

            axAngPair.append(np.array((hPrime[0],hPrime[1],hPrime[2],ang)))

    ro = []

    for pair in axAngPair:

        if pair[3] == 0:

            ro.append(np.array((0,0,0)))

        elif abs(pair[3]-np.pi) < thr:

            f = np.inf
            ro.append(np.array((pair[0]*f,pair[1]*f,pair[2]*f)))

        else:

            f = np.tan(pair[3]/2)
            ro.append(np.array((pair[0]*f,pair[1]*f,pair[2]*f)))

    return axAngPair,np.vstack(ro)

### axis/angle ###

def ax2om(ax):

    c = np.cos(ax[:,3])
    s = np.sin(ax[:,3])

    om = np.zeros((len(ax),9))

    om[:,0] = c+(1-c)*ax[:,0]**2
    om[:,1] = (1-c)*ax[:,0]*ax[:,1]+s*ax[:,2]
    om[:,2] = (1-c)*ax[:,0]*ax[:,2]-s*ax[:,1]

    om[:,3] = (1-c)*ax[:,0]*ax[:,1]-s*ax[:,2]
    om[:,4] = c+(1-c)*ax[:,1]**2
    om[:,5] = (1-c)*ax[:,1]*ax[:,2]+s*ax[:,0]

    om[:,6] = (1-c)*ax[:,0]*ax[:,2]+s*ax[:,1]
    om[:,7] = (1-c)*ax[:,1]*ax[:,2]-s*ax[:,0]
    om[:,8] = c+(1-c)*ax[:,2]**2

    return om
