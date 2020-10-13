#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:28:24 2019
@author: nate
"""

"""
orientation module
"""

import numpy as _np
from numba import jit as _jit
from pyTex.utils import genSymOps as _gensym

__all__ = ['eu2om',
           'eu2quat',
           'om2eu',
           'ro2ho',
           'ro2ax',
           'ho2ro',
           'ax2om',
           'quat2eu']

## TODO: look into subclassing scipy Rotation to add features

### symmetrise ###

def symmetrise(g, cs, ss):

    """
    return symmetric equivalents
    """

    ## use orientation matrix | sample -> crystal (Bunge notation)

    crysSymOps = _gensym(cs)
    # crysSymOps = crysSymOps.transpose((2,0,1))

    if ss == '1' or ss == '-1': 
        smplSymOps = _np.eye(3)[None,:,:] #add third axes

    else: 
        smplSymOps = _gensym(ss)
        # smplSymOps = smplSymOps.transpose((2,0,1))

    # sym_equiv = _np.zeros((3,3,crysSymOps.shape[2]*smplSymOps.shape[2]))
    test = []

    k = 0
    for sopi in range(smplSymOps.shape[0]):
        
        test.append(smplSymOps[sopi,:,:] @ g @ crysSymOps)
        k+=1

    if len(test) > 1:
        return _np.vstack(test)
    else: return test[0]

### Euler (Bunge) space ### 

def eu2om(bunge, out=None, eps=9): 
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

        c1=_np.cos(bunge[0])
        c2=_np.cos(bunge[2])
        c=_np.cos(bunge[1])

        s1=_np.sin(bunge[0])
        s2=_np.sin(bunge[2])
        s=_np.sin(bunge[1])

        g=_np.zeros((3,3))

        g[0,0]=c1*c2-s1*c*s2
        g[0,1]=s1*c2+c1*s2*c
        g[0,2]=s2*s

        g[1,0]=-c1*s2-s1*c2*c
        g[1,1]=-s1*s2+c1*c2*c
        g[1,2]=c2*s

        g[2,0]=s1*s
        g[2,1]=-c1*s
        g[2,2]=c

        g = _np.round(g, decimals=eps)

        return g
    
    elif out == 'mdarray':
    
        c1=_np.cos(bunge[0])
        c2=_np.cos(bunge[2])
        c=_np.cos(bunge[1])
        
        s1=_np.sin(bunge[0])
        s2=_np.sin(bunge[2])
        s=_np.sin(bunge[1])
        
        g11=c1*c2-s1*c*s2
        g12=s1*c2+c1*s2*c
        g13=s2*s
        
        g21=-c1*s2-s1*c2*c
        g22=-s1*s2+c1*c2*c
        g23=c2*s
        
        g31=s1*s
        g32=-c1*s
        g33=c
        
        g = _np.zeros((3,3,_np.prod(bunge[1].shape)))
        b_list = _np.zeros_like(bunge[1])
        k = 0
        
        for p2 in range(bunge[0].shape[0]):
            for p in range(bunge[0].shape[1]):
                for p1 in range(bunge[0].shape[2]):
                    
                    b_list[p2,p,p1] = k
                    
                    #outputs transpose for crystal -> sample
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
                    
        g = _np.round(g, decimals=eps)
                    
        return g,b_list

    elif out == 'mdarray_2':

        c1=_np.cos(bunge[:,0])
        c2=_np.cos(bunge[:,2])
        c=_np.cos(bunge[:,1])
        
        s1=_np.sin(bunge[:,0])
        s2=_np.sin(bunge[:,2])
        s=_np.sin(bunge[:,1])
        
        g11=c1*c2-s1*c*s2
        g12=s1*c2+c1*s2*c
        g13=s2*s
        
        g21=-c1*s2-s1*c2*c
        g22=-s1*s2+c1*c2*c
        g23=c2*s
        
        g31=s1*s
        g32=-c1*s
        g33=c
        
        g = _np.zeros((3,3,_np.prod(bunge.shape[0])))
        # b_list = _np.zeros_like(bunge[1])
        k = 0
        
        for eu_set in range(bunge.shape[0]):
                    
                    # b_list[p2,p,p1] = k
                    
            #outputs transpose for crystal -> sample
            g[0,0,k] = g11[eu_set]
            g[0,1,k] = g21[eu_set]
            g[0,2,k] = g31[eu_set]
            
            g[1,0,k] = g12[eu_set]
            g[1,1,k] = g22[eu_set]
            g[1,2,k] = g32[eu_set]
            
            g[2,0,k] = g13[eu_set]
            g[2,1,k] = g23[eu_set]
            g[2,2,k] = g33[eu_set]
            k += 1
                    
        g = _np.round(g, decimals=eps)
                    
        return g        

    elif out == 'ndarray':

        c1=_np.cos(bunge[:,0])
        c2=_np.cos(bunge[:,2])
        c=_np.cos(bunge[:,1])

        s1=_np.sin(bunge[:,0])
        s2=_np.sin(bunge[:,2])
        s=_np.sin(bunge[:,1])

        g = _np.empty((len(bunge),9))

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
    
    elif out == 'ewimv':

        """
        input is tuple of 2D arrays (alpha,beta,0) or (0,beta,alpha)
        """

        c1=_np.cos(bunge[0])
        c2=_np.cos(bunge[2])
        c=_np.cos(bunge[1])
        
        s1=_np.sin(bunge[0])
        s2=_np.sin(bunge[2])
        s=_np.sin(bunge[1])
        
        g11=c1*c2-s1*c*s2
        g12=s1*c2+c1*s2*c
        g13=s2*s
        
        g21=-c1*s2-s1*c2*c
        g22=-s1*s2+c1*c2*c
        g23=c2*s
        
        g31=s1*s
        g32=-c1*s
        g33=c
        
        g = _np.zeros((3,3,_np.prod(bunge[1].shape)))
        y_list = _np.zeros_like(bunge[1])
        k = 0

        if isinstance(bunge[0],int):

            for y2 in range(bunge[1].shape[1]):
                for y3 in range(bunge[2].shape[0]):
                        
                    y_list[y3,y2] = k
                    
                    #outputs transpose for crystal -> sample
                    g[0,0,k] = g11[y3,y2]
                    g[0,1,k] = g21[y3,y2]
                    g[0,2,k] = g31[y3,y2]
                    
                    g[1,0,k] = g12[y3,y2]
                    g[1,1,k] = g22[y3,y2]
                    g[1,2,k] = g32[y3,y2]
                    
                    g[2,0,k] = g13[y3,y2]
                    g[2,1,k] = g23[y3,y2]
                    g[2,2,k] = g33[y3,y2]
                    k += 1  

        if isinstance(bunge[2],int):

            for y1 in range(bunge[0].shape[0]):
                for y2 in range(bunge[1].shape[1]):
                        
                    y_list[y1,y2] = k
                    
                    #outputs transpose for crystal -> sample
                    g[0,0,k] = g11[y1,y2]
                    g[0,1,k] = g21[y1,y2]
                    g[0,2,k] = g31[y1,y2]
                    
                    g[1,0,k] = g12[y1,y2]
                    g[1,1,k] = g22[y1,y2]
                    g[1,2,k] = g32[y1,y2]
                    
                    g[2,0,k] = g13[y1,y2]
                    g[2,1,k] = g23[y1,y2]
                    g[2,2,k] = g33[y1,y2]
                    k += 1    

        return g,y_list         
    
    else: raise NotImplementedError('out_type not recognized')

def eu2quat(eu, P=-1):

    """
    euler (bunge) to quaternion
        - no correction for north/south hemisphere
        - input Nx3 ndarray (φ1,Φ,φ2)

    convetion from https://doi.org/10.1088/0965-0393/23/8/083501
    """
    
    sig = 0.5*eu[:,0] + 0.5*eu[:,2]
    delt = 0.5*eu[:,0] - 0.5*eu[:,2]
    c = _np.cos(eu[:,1]/2)
    s = _np.sin(eu[:,1]/2)
    
    q = _np.array((c*_np.cos(sig), -P*s*_np.cos(delt), -P*s*_np.sin(delt), -P*c*_np.sin(sig)))
    
    return q

### orientation matrices ###

def om2eu(g): ##calcEuler
    """input is 3x3 array rotation matrix"""

    if g.shape == (3,3):

        phi=_np.arccos(g[2,2])
        if phi==0:
            phi1=_np.arctan2(-g[1,0],g[0,0])
            phi2=0
        elif phi==_np.pi:
            phi1=_np.arctan2(g[1,0],g[0,0])
            phi2=0
        else:
            phi1=_np.arctan2(g[2,0],-g[2,1])
            phi2=_np.arctan2(g[0,2],g[1,2])
        if phi1 < 0:
            phi1 += 2*_np.pi
        if phi2 < 0:
            phi2 += 2*_np.pi

        return [phi1,phi,phi2]

    elif len(g.shape) == 3:

        phi=_np.arccos(g[:,2,2])

        phi1 = _np.zeros_like(phi)
        phi2 = _np.zeros_like(phi)

        for i,ph in enumerate(phi):

            if ph==0:

                phi1[i]=_np.arctan2(-g[i,1,0],g[i,0,0])
                phi2[i]=0

            elif ph==_np.pi:

                phi1[i]=_np.arctan2(g[i,1,0],g[i,0,0])
                phi2[i]=0

            else:

                phi1[i]=_np.arctan2(g[i,2,0],-g[i,2,1])
                phi2[i]=_np.arctan2(g[i,0,2],g[i,1,2])
        
        phi1 = _np.where(phi1 < 0, phi1 + 2*_np.pi, phi1)
        phi2 = _np.where(phi2 < 0, phi2 + 2*_np.pi, phi2)

        # if phi1 < 0:
        #     phi1 += 2*_np.pi
        # if phi2 < 0:
        #     phi2 += 2*_np.pi

        return _np.stack((phi1,phi,phi2),axis=1)

    else:

        phi1 = []
        phi = []
        phi2 = []

        for om in g:

            if om[8] == 1:

                t1 = _np.arctan2(om[1],om[0])

                if t1 < 0:
                    t1 += 2*_np.pi

                phi1.append(t1)
                phi.append((_np.pi/2)*(1-om[8]))
                phi2.append(0)
                # bunge.append(_np.array((_np.arctan2(g[1],g[0]),(_np.pi/2)*(1-g[8]),0)))

            else:

                xi = 1/_np.sqrt(1-om[8]**2)

                t1 = _np.arctan2(om[6]*xi,-om[7]*xi)
                t2 = _np.arctan2(om[2]*xi,om[5]*xi)

                if t1 < 0:
                    t1 += 2*_np.pi
                if t2 < 0:
                    t2 += 2*_np.pi

                phi1.append(t1)
                phi.append(_np.arccos(om[8]))
                phi2.append(t2)

                # bunge.append(_np.array((_np.arctan2(g[6]*xi,-g[7]*xi),_np.arccos(g[8]),_np.arctan2(g[2]*xi,g[5]*xi))))

        return _np.column_stack((phi1,phi,phi2))

def om2quat(g,P=1):
    """input is 3x3 array rotation matrix"""

    if g.shape == (3,3):

        x0 = 0.5*_np.sqrt(1+g[0,0]+g[1,1]+g[2,2])
        x1 = P*0.5*_np.sqrt(1+g[0,0]-g[1,1]-g[2,2])
        x2 = P*0.5*_np.sqrt(1-g[0,0]+g[1,1]-g[2,2])
        x3 = P*0.5*_np.sqrt(1-g[0,0]-g[1,1]+g[2,2])

        if g[2,1] < g[1,2]:
            x1 = -x1
        if g[0,2] < g[2,0]:
            x2 = -x2
        if g[1,0] < g[0,1]:
            x3 = -x3

        Q = _np.array((x0,x1,x2,x3))
        Q = Q/_np.linalg.norm(Q)
        
        return Q

    else:

        x0 = 0.5*_np.sqrt(1+g[:,0]+g[:,4]+g[:,8])
        x1 = P*0.5*_np.sqrt(1+g[:,0]-g[:,4]-g[:,8])
        x2 = P*0.5*_np.sqrt(1-g[:,0]+g[:,4]-g[:,8])
        x3 = P*0.5*_np.sqrt(1-g[:,0]-g[:,4]+g[:,8])

        # x1L = _np.vstack([_np.where(g[idx,7] < g[idx,5],[-x1[idx],x1[idx]]) for idx in range(len(x1))])
        # x1L = _np.where(g[:,7] < g[:,5],[-_np.copy(x1),x1])
        x1L = _np.asarray([xv if c else yv for (c,xv,yv) in zip(g[:,7] < g[:,5],-x1,x1)])
        # x2L = _np.vstack([_np.where(g[idx,2] < g[idx,6],[-x2[idx],x2[idx]]) for idx in range(len(x2))])
        # x2L = _np.where(g[:,2] < g[:,6],[-_np.copy(x2),x2])
        x2L = _np.asarray([xv if c else yv for (c,xv,yv) in zip(g[:,2] < g[:,6],-x2,x2)])
        # x3L = _np.vstack([_np.where(g[idx,3] < g[idx,1],[-x3[idx],x3[idx]]) for idx in range(len(x3))])
        # x3L = _np.where(g[:,3] < g[:,1],[-_np.copy(x3),x3])
        x3L = _np.asarray([xv if c else yv for (c,xv,yv) in zip(g[:,3] < g[:,1],-x3,x3)])

        QL = _np.nan_to_num(_np.column_stack((x0,x1L,x2L,x3L)),copy=False)
        QN = _np.concatenate([Q/_np.linalg.norm(Q) for Q in QL])

        return QN

### rodrigues vectors ###

def ro2ho(ro):

    p = _np.linalg.norm(ro,axis=1)

    ho = []

    for r,l in zip(ro,p):

        if l == 0:

            ho.append(_np.array((0,0,0)))

        elif l == _np.inf:

            f = ((3*_np.pi)/4)
            ho.append((r/l)*f**(1/3))

        else:

            w = 2*_np.arctan(l)
            f = 3*(w-_np.sin(w))/4
            ho.append((r/l)*f**(1/3))

    return _np.vstack(ho)

def ro2ax(ro):

    p = _np.linalg.norm(ro,axis=1)

    with _np.errstate(invalid='ignore'):
        n = ro/p[:,_np.newaxis]

    n = _np.nan_to_num(n) #divide by zero error

    ang = 2*_np.arctan(p)

    return _np.column_stack((n,ang))

### homochoric/cubochoric ###

def _gammaFunc(h,i):

    gamma = _np.array(([1.0000000000018852, -0.5000000002194847,
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

    hc_mag = _np.linalg.norm(hc,axis=1)**2

    hPrime = []
    s = []
    axAngPair = []

    for idx,mag in enumerate(hc_mag):

        if mag == 0:

            axAngPair.append(_np.array((0,0,1,0)))

        else:

            hPrime = hc[idx,:] / _np.sqrt(mag)
            s = _np.asarray([_gammaFunc(mag,i) for i in range(16)]).sum()

            ang = 2*_np.arccos(s)

            if abs(ang-_np.pi) < thr:
                ang = _np.pi

            axAngPair.append(_np.array((hPrime[0],hPrime[1],hPrime[2],ang)))

    ro = []

    for pair in axAngPair:

        if pair[3] == 0:

            ro.append(_np.array((0,0,0)))

        elif abs(pair[3]-_np.pi) < thr:

            f = _np.inf
            ro.append(_np.array((pair[0]*f,pair[1]*f,pair[2]*f)))

        else:

            f = _np.tan(pair[3]/2)
            ro.append(_np.array((pair[0]*f,pair[1]*f,pair[2]*f)))

    return axAngPair,_np.vstack(ro)

### axis/angle ###

def ax2om(ax, P=-1):

    c = _np.cos(ax[:,3])
    s = _np.sin(ax[:,3])

    om = _np.zeros((len(ax),9))

    if P == -1:
        om[:,0] = c+(1-c)*ax[:,0]**2
        om[:,1] = (1-c)*ax[:,0]*ax[:,1]+s*ax[:,2]
        om[:,2] = (1-c)*ax[:,0]*ax[:,2]-s*ax[:,1]

        om[:,3] = (1-c)*ax[:,0]*ax[:,1]-s*ax[:,2]
        om[:,4] = c+(1-c)*ax[:,1]**2
        om[:,5] = (1-c)*ax[:,1]*ax[:,2]+s*ax[:,0]

        om[:,6] = (1-c)*ax[:,0]*ax[:,2]+s*ax[:,1]
        om[:,7] = (1-c)*ax[:,1]*ax[:,2]-s*ax[:,0]
        om[:,8] = c+(1-c)*ax[:,2]**2

    elif P == 1:
        om[:,0] = c+(1-c)*ax[:,0]**2
        om[:,1] = (1-c)*ax[:,0]*ax[:,1]-s*ax[:,2]
        om[:,2] = (1-c)*ax[:,0]*ax[:,2]+s*ax[:,1]

        om[:,3] = (1-c)*ax[:,0]*ax[:,1]+s*ax[:,2]
        om[:,4] = c+(1-c)*ax[:,1]**2
        om[:,5] = (1-c)*ax[:,1]*ax[:,2]-s*ax[:,0]

        om[:,6] = (1-c)*ax[:,0]*ax[:,2]-s*ax[:,1]
        om[:,7] = (1-c)*ax[:,1]*ax[:,2]+s*ax[:,0]
        om[:,8] = c+(1-c)*ax[:,2]**2

    return om

def _ax2om_2(ax,ang,eps=1E-12):
    
    """
    Produce rotation matrix from given axis, angle combo
    
    Inputs:
        ax: Nx3 ndarray (normalized)
        ang: int
        eps: error truncation
    """
    
    #outer prod | identity
    outprod = _np.zeros((3,3,ax.shape[0]))
    I = _np.zeros_like(outprod)
    for i,a in enumerate(ax):
        outprod[:,:,i] = _np.outer(a,a)
        I[:,:,i] = _np.eye(ax.shape[1])

    I_2 = _np.eye(ax.shape[1])
    a = sum([_np.outer(_np.cross(ax, ei),ei) for ei in I_2])
    
    #reshape (3x3xN)
    a = _np.reshape(_np.ravel(a),(3,3,ax.shape[0]), order = 'F')
    #transpose
    a = _np.swapaxes(a, 0, 1)
    
    om = _np.cos(ang) * I + _np.sin(ang) * a + ( 1 - _np.cos(ang) ) * outprod
    
    #round very small values to 0
    om[_np.abs(om) < eps] = 0
    
    return om  

### quaternions ###

def err_handler(type, flag):
    import traceback
    traceback.print_stack(limit=2)
    print()
    print("Floating point error (%s), with flag %s").format(type, flag)

def quat2eu(quat, P=-1):

    """
    quaternion to euler (bunge)
        - input is NxMx4 mdarray | N - along fibre. M - sym equiv.
        - output is 3 array(N) - (φ1,Φ,φ2)

    convention from https://doi.org/10.1088/0965-0393/23/8/083501
    """

    q03 = quat[:,:,0]**2 + quat[:,:,3]**2 
    q12 = quat[:,:,1]**2 + quat[:,:,2]**2 
    chi = _np.sqrt(q03*q12)

    q03 = _np.round(q03, decimals = 15)
    q12 = _np.round(q12, decimals = 15)
    chi = _np.round(chi, decimals = 15)
    
    case1 = (chi == 0) & (q12 == 0)
    case2 = (chi == 0) & (q03 == 0)
    case3 = (chi != 0)
    
    phi1 = _np.zeros_like(q03)
    Phi = _np.zeros_like(q03)
    phi2 = _np.zeros_like(q03)
    
    q0 = quat[:,:,0]
    q1 = quat[:,:,1]
    q2 = quat[:,:,2]
    q3 = quat[:,:,3]
    
    # saved_handler = _np.seterrcall(err_handler)
    save_err = _np.seterr(all='raise')

    if _np.any(case1): 
        
        phi1_t = _np.arctan2( -2*P*q0[case1]*q3[case1], q0[case1]**2 - q3[case1]**2 )
        Phi_t = _np.zeros_like(phi1_t)
        phi2_t = _np.zeros_like(phi1_t) 
        
        phi1[case1] = phi1_t
        Phi[case1] = Phi_t
        phi2[case1] = phi2_t            
        
    # if len(case2[0]) != 0: 
    if _np.any(case2):
        
        phi1_t = _np.arctan2( 2*q1[case2]*q2[case2], q1[case2]**2 - q2[case2]**2 )
        Phi_t = _np.ones_like(phi1_t)*_np.pi
        phi2_t = _np.zeros_like(phi1_t)

        phi1[case2] = phi1_t
        Phi[case2] = Phi_t
        phi2[case2] = phi2_t
        
    # if len(case3[0]) != 0:
    if _np.any(case3):
        
        phi1_t = _np.arctan2( (q1[case3]*q3[case3] - P*q0[case3]*q2[case3]) / chi[case3],
                                (-P*q0[case3]*q1[case3] - q2[case3]*q3[case3]) / chi[case3] )            
        Phi_t = _np.arctan2( 2*chi[case3], q03[case3] - q12[case3])            
        phi2_t = _np.arctan2( (P*q0[case3]*q2[case3] + q1[case3]*q3[case3]) / chi[case3],
                                (q2[case3]*q3[case3] - P*q0[case3]*q1[case3]) / chi[case3] )  
        
        phi1[case3] = phi1_t
        Phi[case3] = Phi_t
        phi2[case3] = phi2_t

    phi1 = _np.round(phi1, decimals = 8)
    Phi = _np.round(Phi, decimals = 8)
    phi2 = _np.round(phi2, decimals = 8)

    return phi1,Phi,phi2
    # return case1, case2, case3

def quat2om(Q,P=1):

    """
    array Nx4 - quaternions [q0, q1, q2, q3]

    """

    g = _np.empty((len(Q),9))

    Qmean = Q[:,0]**2 - (Q[:,1]**2+Q[:,2]**2+Q[:,3]**2)
    g[:,0] = Qmean + 2*Q[:,1]**2
    g[:,1] = 2*(Q[:,1]*Q[:,2]-P*Q[:,0]*Q[:,3])
    g[:,2] = 2*(Q[:,1]*Q[:,3]+P*Q[:,0]*Q[:,2])
    g[:,3] = 2*(Q[:,1]*Q[:,2]+P*Q[:,0]*Q[:,3])
    g[:,4] = Qmean + 2*Q[:,2]**2
    g[:,5] = 2*(Q[:,2]*Q[:,3]-P*Q[:,0]*Q[:,1])
    g[:,6] = 2*(Q[:,1]*Q[:,3]-P*Q[:,0]*Q[:,2])
    g[:,7] = 2*(Q[:,2]*Q[:,3]+P*Q[:,0]*Q[:,1])
    g[:,8] = Qmean + 2*Q[:,3]**2

    return g

#TODO: get rid of need for numba
@_jit(nopython=True,parallel=False)
def quatMetricNumba(a, b):
    
    """ from DOI 10.1007/s10851-009-0161-2, #4 """
    
    dist = _np.zeros((len(a),len(b)))
    
    for bi in range(len(b)):
        
        dist[:,bi] = 1 - _np.abs(_np.dot(a,b[bi]))
    
    return dist