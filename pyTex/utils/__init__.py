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

### plotting util ###

def _scatterPlot( alpha, beta, data, cols, rows, proj, cmap, axes_labels, x_direction='N'):

    """
    wrapper for scatter
    """

    fig, axes = _plt.subplots(rows, cols, subplot_kw=dict(polar=True))

    if proj == 'stereo': plt_alpha = (_np.pi/2)*_np.tan(_copy.deepcopy(alpha)/2)
    elif proj == 'earea': plt_alpha = 2*_np.sin(_copy.deepcopy(alpha)/2)
    elif proj == 'none': plt_alpha = _copy.deepcopy(alpha)

    #append start/end to create complete plots
    dbeta = _np.diff(beta).mean()
    plt_beta  = _np.concatenate( (beta, beta[:,-1][:,None] + dbeta), axis=1 )
    plt_alpha = _np.concatenate( (plt_alpha,plt_alpha[:,0][:,None]), axis=1 )

    for n,ax in enumerate(axes):

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_ylim([0,_np.max(plt_alpha)])
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

    if proj == 'stereo': plt_alpha = (_np.pi/2)*_np.tan(_copy.deepcopy(alpha)/2)
    elif proj == 'earea': plt_alpha = 2*_np.sin(_copy.deepcopy(alpha)/2)
    elif proj == 'none': plt_alpha = _copy.deepcopy(alpha)

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
        ax.set_ylim([0,_np.max(plt_alpha)])
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

def XYZtoSPH( xyz, proj='stereo', upperOnly=True ):
    
    if len(xyz.shape) == 3:

        ## invert with -y ##

        if upperOnly is True:

            for i in range( xyz.shape[2] ):
            
                temp = xyz[:,:,i]
                
                neg_idx = _np.where(temp[:,2] < 0)
                
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
        
        xy = _np.power(xyz[:,0,:],2) + _np.power(xyz[:,1,:],2)
        sph1 = _np.arctan2(_np.sqrt(xy), xyz[:,2,:]) #alpha
        
        if proj == 'stereo': sph1 = (_np.pi/2)*_np.tan(sph1/2)
        elif proj == 'earea': sph1 = 2*_np.sin(sph1/2)
        elif proj == 'none': pass
        
        sph0 = _np.arctan2(xyz[:,1,:], xyz[:,0,:]) #beta  
        # move all angles between 0-2π
        sph0 = _np.where(sph0 < 0, sph0 + 2*_np.pi, sph0) 
        
        for i in range(sph0.shape[1]):
            
            sph[:,0,i] = sph0[:,i]
            sph[:,1,i] = sph1[:,i]        

    else:

        if upperOnly is True:

            xyz[xyz[:,2] < 0] = -xyz[xyz[:,2] < 0]
        
        sph = _np.zeros((len(xyz),2))
        xy = _np.power(xyz[:,1],2) + _np.power(xyz[:,0],2)
        
        sph[:,1] = _np.arctan2(_np.sqrt(xy), xyz[:,2])
        if proj == 'stereo': sph[:,1] = (_np.pi/2)*_np.tan(sph[:,1]/2)
        elif proj == 'earea': sph[:,1] = 2*_np.sin(sph[:,1]/2)
        elif proj == 'none': pass
        
        # eliminate large values
        sph[:,0] = _np.arctan2(xyz[:,0], xyz[:,1])
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