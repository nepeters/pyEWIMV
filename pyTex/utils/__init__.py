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
           ]

__location__ = _os.path.realpath(_os.path.join(_os.getcwd(), _os.path.dirname(__file__)))

"""
general purpose functions
"""

import numpy as _np
import pandas as _pd

import spglib as _spglib

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

### other utils ###

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