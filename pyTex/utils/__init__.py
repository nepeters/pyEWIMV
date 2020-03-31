#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:28:24 2019

@author: nate
"""

"""
orientation module
"""

import os as _os

__all__ = ['normalize',
           'genSym',
           'symmetrise',
           'XYZtoSPH',
           'SPHtoXYZ'
           ]

__location__ = _os.path.realpath(_os.path.join(_os.getcwd(), _os.path.dirname(__file__)))

"""
general purpose functions
"""

#data
import numpy as _np
import pandas as _pd

#sym ops
import spglib as _spglib

#rotations - easy
from scipy.spatial.transform import Rotation as _R

#plotting
import matplotlib.pyplot as _plt
from mpl_toolkits.axes_grid1 import ImageGrid as _imgGrid
import copy as _copy

def normalize(v):
    
    """
    normalize vectors
    
    input: list of Nx3 ndarray
    
    """
    
    if isinstance(v,list):

        hkl_n = []
        
        for hkl in v:

            norm = _np.linalg.norm(hkl,axis=1)
            hkl_n.append(hkl / norm[:,_np.newaxis])

    elif isinstance(v,_np.ndarray):

        norm = _np.linalg.norm(v,axis=1)
        hkl_n = v / norm[:,_np.newaxis]


    return hkl_n

def genSymOps(cs):

    """
    generate sym ops for system
    
    laue group: 'm-3', 'm-3m'
    """

    spgtbl_path = _os.path.join(__location__, 'spacegroups.in')

    if cs == 'm-3' or cs == '23':

        raise NotImplementedError('coming soon..')

    elif cs == 'm-3m' or cs == '432':

        spg_tbl = _pd.read_csv(spgtbl_path, sep="\t")
        # use spacegroup #225
        hall_num = spg_tbl[spg_tbl['Table No.'] == 225]['Serial No.'].iloc[0]
        symOps = _spglib.get_symmetry_from_database(hall_num)['rotations']

        symOps = _np.swapaxes(symOps,0,2)
        symOps = _np.swapaxes(symOps,0,1)

    elif cs == 'mmm':

        spg_tbl = _pd.read_csv(spgtbl_path, sep="\t")
        # use spacegroup #225
        hall_num = spg_tbl[spg_tbl['Table No.'] == 47]['Serial No.'].iloc[0]
        symOps = _spglib.get_symmetry_from_database(hall_num)['rotations']

        symOps = _np.swapaxes(symOps,0,2)
        symOps = _np.swapaxes(symOps,0,1)        
    
    return symOps

def symmetrise(cs,hkl):

    """
    symmetrise vector based on crystal sym

    inputs:
        laue group: 'm-3','m-3m'
        hkl: Nx3 ndarray
    """

    if cs == 'm-3' or cs == '23':

        raise NotImplementedError('coming soon..')

    elif cs == 'm-3m' or cs == '432':

        symOps = genSymOps(cs)
        temp = _np.dot(hkl,symOps)

        if len(temp.shape) > 2:

            symHKL = []

            #multiple hkls, separate to return in tuple
            for i in range(temp.shape[0]):
                
                #save only unique vectors
                symHKL.append(_np.unique(temp[i,:,:].T,axis=0))

        elif len(temp.shape) == 2:

            symHKL = _np.unique(temp.T,axis=0)

        else:

            raise ValueError('symHKL is weird shape')

        return symHKL

### scattering vector ###

def calcScatVec( v, r, sample_r=None ):

    """
    v' = r * v 
    passive rotation supplied as matrix, or scipy Rotation inst.

    returns earea proj, xyz, sph
    """

    ## check on r, sample_r

    if isinstance(r, _np.ndarray):
        assert r.shape == (3,3)
        r = _R.from_matrix(r)
    elif isinstance(r, _R): pass
    else: raise ValueError('rotation not valid')

    if sample_r is not None:

        if isinstance(sample_r, _np.ndarray):
            assert sample_r.shape == (3,3)
            sample_r = _R.from_matrix(sample_r)
        elif isinstance(sample_r, _R): pass
        else: raise ValueError('sample rotation not valid')
    
    #default to the identity matrix
    else: sample_r = _R.identity()

    a = _np.linalg.norm(v,axis=1)
    norm = _np.sqrt((2*a**2)-(2*a*v[:,2]))

    vp = _copy.deepcopy(v)
    vp[:,2] = vp[:,2] - a
    
    sv = (1/norm[:,None]) * sample_r.apply(r.apply(vp))

    return XYZtoSPH(_copy.deepcopy(sv),proj='earea',SNS=True), sv, XYZtoSPH(_copy.deepcopy(sv),SNS=True)

### plotting util ###

def _sparseScatterPlot( polar, data, cols, rows, proj, cmap, axes_labels, x_direction='N'):

    """
    wrapper for plt.scatter
    sparse data
    """

    fig, axes = _plt.subplots(rows, cols, subplot_kw=dict(polar=True))


    # find min/max for all
    mx = 0
    mn = 1E9

    for n,ax in enumerate(axes):

        if _np.max(data[n]) > mx: mx = _np.max(data[n])
        if _np.min(data[n]) < mn: mn = _np.min(data[n])

    for n,ax in enumerate(axes):

        alpha = _copy.deepcopy(polar[n][:,1])
        plt_beta = _copy.deepcopy(polar[n][:,0])

        if proj == 'stereo': 
            plt_alpha = _np.tan(_copy.deepcopy(alpha)/2)
            max_alpha = 1
        elif proj == 'earea': 
            plt_alpha = 2*_np.sin(_copy.deepcopy(alpha)/2)
            max_alpha = _np.sqrt(2)
        elif proj == 'none': 
            plt_alpha = _copy.deepcopy(alpha)
            max_alpha = _np.pi/2        

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_ylim([0,max_alpha])
        ax.set_theta_zero_location(x_direction) 
        ax.grid(False)

        norm = _plt.Normalize(vmin=mn, vmax=mx)

        pts = ax.scatter(plt_beta,
                         plt_alpha,
                         c=data[n],
                         s=6,
                         cmap=cmap,
                         norm=norm)

        ax.text(0,max_alpha,axes_labels['X'],
                fontsize=8,
                va='center',
                ha='center',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

        ax.text(max_alpha,max_alpha,axes_labels['Y'],
                fontsize=8,
                va='center',
                ha='center',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

    fig.colorbar(pts, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    _plt.show()

def _scatterPlot( alpha, beta, data, cols, rows, proj, cmap, axes_labels, x_direction='N'):

    """
    wrapper for scatter
    """

    fig, axes = _plt.subplots(rows, cols, subplot_kw=dict(polar=True))

    if proj == 'stereo': 
        plt_alpha = _np.tan(_copy.deepcopy(alpha)/2)
        max_alpha = 1
    elif proj == 'earea': 
        plt_alpha = 2*_np.sin(_copy.deepcopy(alpha)/2)
        max_alpha = _np.sqrt(2)
    elif proj == 'none': 
        plt_alpha = _copy.deepcopy(alpha)
        max_alpha = _np.pi/2

    #append start/end to create complete plots
    dbeta = _np.diff(beta).mean()
    plt_beta  = _np.concatenate( (beta, beta[:,-1][:,None] + dbeta), axis=1 )
    plt_alpha = _np.concatenate( (plt_alpha,plt_alpha[:,0][:,None]), axis=1 )

    for n,ax in enumerate(axes):

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_ylim([0,max_alpha])
        ax.set_theta_zero_location(x_direction) 
        ax.grid(False)

        plt_data = _np.concatenate((data[n], data[n][:, 0:1]), axis=1)

        pts = ax.scatter(plt_beta,
                         plt_alpha,
                         c=plt_data,
                         s=6,
                         cmap=cmap)

        ax.text(0,_np.max(plt_alpha),axes_labels['X'],
                fontsize=8,
                va='center',
                ha='center',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

        ax.text(_np.pi/2,_np.max(plt_alpha),axes_labels['Y'],
                fontsize=8,
                va='center',
                ha='center',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

    fig.colorbar(pts, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    _plt.show()

def _contourPlot( alpha, beta, data, cols, rows, proj, cmap, axes_labels, contourlevels, filled=True, x_direction='N'):

    """ 
    wrapper for contourf
    """

    fig, axes = _plt.subplots(rows, cols, subplot_kw=dict(polar=True))

    if proj == 'stereo': 
        plt_alpha = 1*_np.tan(_copy.deepcopy(alpha)/2)
        max_alpha = 1
    elif proj == 'earea': 
        plt_alpha = 2*_np.sin(_copy.deepcopy(alpha)/2)
        max_alpha = _np.sqrt(2)
    elif proj == 'none': 
        plt_alpha = _copy.deepcopy(alpha)
        max_alpha = _np.pi/2

    #append start/end to create complete plots
    dbeta = _np.diff(beta).mean()
    plt_beta  = _np.concatenate( (beta, beta[:,-1][:,None] + dbeta), axis=1 )
    plt_alpha = _np.concatenate( (plt_alpha,plt_alpha[:,0][:,None]), axis=1 )

    #equal colormap
    if contourlevels is None: pass
    elif isinstance(contourlevels,str) and contourlevels == 'equal': contourlevels = _np.arange(0,_np.ceil(_np.max([_np.max(d) for n,d in data.items()])),0.5)

    for n,ax in enumerate(axes):

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_ylim([0,max_alpha])
        ax.set_theta_zero_location(x_direction) 
        ax.grid(False)

        plt_data = _np.concatenate((data[n], data[n][:, 0:1]), axis=1)

        if contourlevels is None:

            cont = ax.contour(plt_beta,
                              plt_alpha,
                              plt_data,
                              colors='k',
                              linewidths=0.1 )

            if filled: pt = ax.contourf(plt_beta,plt_alpha,plt_data,cmap=cmap)

        elif contourlevels is not None:

            cont = ax.contour(plt_beta,
                              plt_alpha,
                              plt_data,
                              colors='k',
                              linewidths=0.1,
                              levels=contourlevels)

            if filled: pt = ax.contourf(plt_beta,plt_alpha,plt_data,cmap=cmap,levels=contourlevels)

        ax.text(0,_np.max(plt_alpha),axes_labels['X'],
                fontsize=8,
                va='center',
                ha='center',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
        ax.text(_np.pi/2,_np.max(plt_alpha),axes_labels['Y'],
                fontsize=8,
                va='center',
                ha='center',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

    fig.colorbar(pt, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    _plt.show()

### coordinate sys transform ###

def XYZtoSPH( xyz, proj='none', upperOnly=True, SNS=False ):
    
    """
    coordinate system
    xy: in-plane
    z:  north (up)

    x,y,z - supplied in that order
    """

    if SNS: #SNS coordinate system
        up = 1
        ip = (0, 2)
    else:
        up = 2
        ip = (0, 1)

    #3D array
    if len(xyz.shape) == 3:

        ## invert with -y ##

        if upperOnly is True:

            for i in range( xyz.shape[up] ):
            
                temp = xyz[:,:,i]
                
                neg_idx = _np.where(temp[:,up] < 0)
                
                xyz[neg_idx,:,i] = -xyz[neg_idx,:,i]

            """ alternative method? """
            # priorShape = xyz.shape

            # #reshape from mdarray to ndarray
            # xyz = xyz.transpose(2,0,1).reshape(xyz.shape[2]*xyz.shape[0],-1)

            # # neg_idx = _np.where(xyz[:,2] < 0)
            # # xyz[neg_idx,:] = -xyz[neg_idx,:]
            # xyz[xyz[:,2] < 0] = -xyz[xyz[:,2] < 0] 

            # xyz = xyz.reshape(priorShape[2],priorShape[0],priorShape[1]).transpose(1,2,0)

        else: pass

        sph = _np.zeros((xyz.shape[0],2,xyz.shape[2]))
        
        inplane = _np.power(xyz[:,ip[0],:],2) + _np.power(xyz[:,ip[1],:],2)
        sph1 = _np.arctan2(_np.sqrt(inplane), xyz[:,up,:]) #alpha
        
        if proj == 'stereo': sph1 = _np.tan(sph1/2)
        elif proj == 'earea': sph1 = 2*_np.sin(sph1/2)
        elif proj == 'none': pass
        
        sph0 = _np.arctan2(xyz[:,ip[1],:], xyz[:,ip[0],:]) #beta  
        # move all angles between 0-2π
        sph0 = _np.where(sph0 < 0, sph0 + 2*_np.pi, sph0) 
        
        for i in range(sph0.shape[1]):
            
            sph[:,0,i] = sph0[:,i]
            sph[:,1,i] = sph1[:,i]        

    else:

        if upperOnly is True:

            xyz[xyz[:,up] < 0] = -xyz[xyz[:,up] < 0]
        
        sph = _np.zeros((len(xyz),2))
        inplane = _np.power(xyz[:,ip[1]],2) + _np.power(xyz[:,ip[0]],2)
        
        sph[:,1] = _np.arctan2(_np.sqrt(inplane), xyz[:,up])
        if proj == 'stereo': sph[:,1] = _np.tan(sph[:,1]/2)
        elif proj == 'earea': sph[:,1] = 2*_np.sin(sph[:,1]/2)
        elif proj == 'none': pass
        
        # eliminate large values
        sph[:,0] = _np.arctan2(xyz[:,ip[1]], xyz[:,ip[0]])
        sph[:,0] = _np.where(sph[:,0] < 0, sph[:,0] + 2*_np.pi, sph[:,0])

    return sph 

def SPHtoXYZ( azimuth, polar, offset=False ):

    """
    sph to xyz
    Nx1 arrays 
    """

    #xyz
    sph = _np.array((_np.ravel(azimuth),_np.ravel(polar))).T

    if offset:
        #offset (001) direction to prevent errors during path calculation
        sph[:,1] = _np.where(sph[:,1] == 0, _np.deg2rad(0.1), sph[:,1])

    #convert to xyz
    xyz = _np.zeros((sph.shape[0],3))
    xyz[:,0] = _np.sin( sph[:,1] ) * _np.cos( sph[:,0] )
    xyz[:,1] = _np.sin( sph[:,1] ) * _np.sin( sph[:,0] )
    xyz[:,2] = _np.cos( sph[:,1] ) 

    return xyz

""" homogenization """

def voigt2tensor( voigt, compliance=False):

    """
    thanks jishnu
    """

    Atensor = _np.zeros([3, 3, 3, 3])
    fact=1
    for i in range(3):
        for j in range(3):
            ## assigning the index m according to conversion table
            if ((i==0 and j==0)): m = 0
            if  ((i==1 and j==1)): m = 1
            if  ((i==2 and j==2)): m = 2
            if  (((i==1 and j==2)) or ((i==2 and j==1))): m = 3
            if  (((i==0 and j==2)) or ((i==2 and j==0))): m = 4
            if  (((i==0 and j==1)) or ((i==1 and j==0))): m = 5
            
            for k in range(3):
                for l in range(3):
                    ## assigning the index n according to conversion table
                    if k==0 and l==0: n = 0
                    if  k==1 and l==1: n = 1
                    if  k==2 and l==2: n = 2
                    if  ((k==1 and l==2) or (k==2 and l==1)): n = 3
                    if  ((k==0 and l==2) or (k==2 and l==0)): n = 4
                    if  ((k==0 and l==1) or (k==1 and l==0)): n = 5
                    
                    ## assigning the tensor values accordingly
                    if compliance is True:
                        if (m>=3) or (n>=3): fact = 2
                    if compliance is True:
                        if (m>=3) and (n>=3): fact = 4
                    
                    Atensor[i,j,k,l]=voigt[m,n]/fact
                    fact = 1

            
    return Atensor

def tensor2voigt( tensor, compliance=False):

    """
    convert a 4th rank tensor into Voigt notation
    thanks jishnu
    """
    A_voigt = _np.zeros([6, 6])
    fact = 1
    for m in range(6):
        for n in range(6):
            # assigning the index i and j according to conversion table
            if (n==0):
                i = 0
                j = 0
            elif (n==1): 
                i = 1
                j = 1
            elif  (n==2): 
                i = 2
                j = 2
            elif  (n==3): 
                i = 1
                j = 2
            elif  (n==4): 
                i = 0
                j = 2
            elif  (n==5): 
                i = 0
                j = 1
            
            # assigning the index k and l according to conversion table
            if (m==0): 
                k = 0
                l = 0
            elif  (m==1): 
                k = 1
                l = 1
            elif  (m==2):
                k = 2
                l = 2
            elif  (m==3):
                k = 1
                l = 2
            elif  (m==4):
                k = 0
                l = 2
            elif  (m==5): 
                k = 0
                l = 1
            
            if compliance is True:
                if ((m>=3) or (n>=3)): fact = 2
            if compliance is True:
                if ((m>=3) and (n>=3)): fact = 4

            A_voigt[m,n] = fact*tensor[i,j,k,l]
            fact = 1
            
    return A_voigt
        
    