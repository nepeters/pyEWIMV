#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:11:12 2019

@author: nate
"""

"""
texture base
"""

import os as _os
import sys as _sys
import copy as _copy
import time as _time

import numpy as _np 
import pandas as _pd
from scipy.interpolate import griddata as _griddata
from scipy.spatial.transform import rotation as _R
from sklearn.neighbors import KDTree as _KDTree
import matplotlib.pyplot as _plt
try:
    import mayavi.mlab as _mlab
except:
    print('mayavi didnt import')
import xrdtools
import rowan as _quat

from tqdm.auto import tqdm as _tqdm

from .orientation import eu2om as _eu2om
from .orientation import eu2quat as _eu2quat
from .orientation import quat2eu as _quat2eu
from .orientation import quatMetricNumba as _quatMetric
from .utils import symmetrise as _symmetrise
from .utils import normalize as _normalize
from .utils import XYZtoSPH as _XYZtoSPH
from .utils import SPHtoXYZ as _SPHtoXYZ
from .utils import _contourPlot
from .utils import _scatterPlot
from .utils import _sparseScatterPlot
from .utils import genSymOps as _genSymOps
from .utils import tensor2voigt as _tensor2voigt, voigt2tensor as _voigt2tensor
from .orientation import om2quat as _om2quat
from .orientation import symmetrise as _symOri
from .orientation import om2eu as _om2eu

### general functions ###
### block print from xrdtools ###

def _blockPrint():
    _sys.stdout = open(_os.devnull, 'w')

def _enablePrint():
    _sys.stdout = _sys.__stdout__
    
### custom objects ###

class poleFigure(object):
    
    """
    pole figure constructor
    """
    
    def __init__( self, files, hkls, cs, pf_type, subtype=None, names=None, resolution=None, arb_y=None, centered=False, ss=None ):
        
        """
        list of file names
        list of (hkl) poles
        type of pole fig
            'xrdml'
            'nd'
        optional: names (list str)
        optional: resolution (int)
        """

        #symmetry
        self.cs = cs
        self.ss = ss

        #reflections
        if isinstance(hkls,list): self.hkls = _np.vstack(hkls)
        else: self.hkls = hkls

        self.refls = _symmetrise(cs, hkls)
        self._normHKLs = _normalize(self.hkls)
        self._symHKL   = _symmetrise(cs, self._normHKLs)
        self._numHKL   = len(self.hkls)

        #data / directions
        self.data  = {}
        self.y     = {}
        self.y_pol = {}

        if pf_type == 'xrdml':
            
            self.twotheta = {}
            
            for i,(f,h) in enumerate(zip(files,hkls)):
                
                if f.endswith('xrdml') and _os.path.exists(f):
                    
                    _blockPrint()
                    temp = xrdtools.read_xrdml(f)
                    _enablePrint()
                        
                    if subtype == 'bkgd':

                        self.data[i]   = temp['data']
                        self.twotheta[i] = temp['2Theta']
                        self.subtype = 'bkgd'

                        continue

                    elif subtype == 'defocus':

                        self.data[i]   = temp['data']
                        self.twotheta[i] = temp['2Theta']
                        self.subtype = 'defocus'

                        continue
                    
                    self.data[i] = temp['data']
                    self.twotheta[i] = temp['2Theta']
                    self.subtype = 'poleFig'
                    
                    #assuming equal grid spacing
                    if resolution is None: self.res = _np.diff(temp['Phi'],1)[0] #pull diff from first 2 points
                    else: self.res = resolution
                    
                    #polar
                    self._alphasteps = _np.arange(0,temp['data'].shape[0]*self.res,self.res)
                    #azimuth
                    self._betasteps = _np.arange(2.5,temp['data'].shape[1]*self.res,self.res)
                    #grids
                    self.alpha, self.beta = _np.meshgrid(self._alphasteps,self._betasteps,indexing='ij')
                    
                    #convert to rad
                    self.alpha = _np.deg2rad(self.alpha)
                    self.beta = _np.deg2rad(self.beta)
                    self.res = _np.deg2rad(self.res)

                    #grid for WIMV
                    k = 0
                    self.pf_grid = _np.zeros((len(self._alphasteps), len(self._betasteps)))

                    for a in range(self.pf_grid.shape[0]):
                        for b in range(self.pf_grid.shape[1]):
                    
                            self.pf_grid[a,b] = k
                            k += 1
                    
                    #store y_pol, y
                    self.y[i] = _SPHtoXYZ( _np.ravel(self.beta), _np.ravel(self.alpha) )

                    #standard method
                    self.cellVolume = self.res * ( _np.cos( self.alpha - (self.res/2) ) - _np.cos( self.alpha + (self.res/2)) )
                    self.cellVolume[0,:] = self.res * ( _np.cos( self.alpha[0,:] + (self.res/2)) ) #only positive side present at alpha=0
                    
                else: raise TypeError('file type not recognized/found, must be ".xrdml"')

        elif pf_type == 'sparse': 

            for i,(f,h) in enumerate(zip(files,hkls)):

                if f.endswith('.pf') and _os.path.exists(f):

                    """ output from pyReducePF """

                    temp = _np.genfromtxt(f)

                    self.data[i] = temp[:,3]
                    self.y[i] = temp[:,:3]
                    self.y_pol[i] = _XYZtoSPH(_np.copy(temp[:,:3]),proj='none')
                    
            self.subtype = 'sparse'

        elif pf_type == 'mtex': 

            for i,(f,h) in enumerate(zip(files,hkls)):

                if f.endswith('.txt') and _os.path.exists(f):

                    temp = _np.genfromtxt(f)

                    self.data[i] = temp[:,2]
                    # self.y_pol[i] = temp[:,:2]
                    # self.y[i] = _SPHtoXYZ( self.y_pol[i][:,1], self.y_pol[i][:,0] )
                    self.y[i] = _SPHtoXYZ( temp[:,1], temp[:,0] )
                    self.y_pol[i] = _XYZtoSPH( self.y[i] )

                    
            self.subtype = 'mtex'         

        elif pf_type == 'recalc': 

            #equal grid spacing
            if resolution is None: self.res = None
            else: self.res = resolution

            if centered is True:

                #polar
                self._alphasteps = _np.arange(self.res/2,
                                            ( 90 - (self.res/2) ) + self.res,
                                            self.res)
                #azimuth
                self._betasteps = _np.arange(self.res/2,
                                            ( 360 - (self.res/2) ) + self.res,
                                            self.res)

            elif centered is False:

                #polar
                self._alphasteps = _np.arange(0,
                                              90 + self.res,
                                              self.res)
                #azimuth
                self._betasteps = _np.arange(2.5,
                                             357.5 + self.res,
                                             self.res)            
            
            #grids
            self.alpha, self.beta = _np.meshgrid(self._alphasteps,self._betasteps,indexing='ij')

            for i,h in enumerate(self.hkls):

                self.y_pol[i] = _np.deg2rad(_np.array((_np.ravel(self.alpha),_np.ravel(self.beta))).T)

            #convert to rad
            self.alpha = _np.deg2rad(self.alpha)

            self.beta = _np.deg2rad(self.beta)
            self.res = _np.deg2rad(self.res)

            inter_x = _copy.deepcopy(self.alpha*_np.cos(self.beta))
            inter_y = _copy.deepcopy(self.alpha*_np.sin(self.beta))
            
            #calculated with equispaced grid
            #must interpolate
            if isinstance(files,dict):

                intMethod='linear'

                if arb_y is None: raise ValueError('Please provide arbitrary pole figure vectors (y) for interpolation')
                else: arb_y_pol = _XYZtoSPH(arb_y,proj='none')

                for i,h in enumerate(hkls):

                    x_pf = _copy.deepcopy(arb_y_pol[:,1]*_np.cos(arb_y_pol[:,0]))
                    y_pf = _copy.deepcopy(arb_y_pol[:,1]*_np.sin(arb_y_pol[:,0]))

                    self.data[i] = abs(_griddata(_np.array((x_pf,y_pf)).T, files[i], (inter_x,inter_y), method=intMethod, fill_value=0.05))                
                    self.hkls[i] = h
                    
            else:

                for i,h in enumerate(hkls):
                                    
                    #reuse files variable... may not be clear
                    self.data[i] = files[:,:,i]
                    self.hkls[i] = h
                    
            self.subtype = 'recalc'
                
            if centered is True: self.cellVolume = self.res * ( _np.cos( self.alpha - (self.res/2) ) - _np.cos( self.alpha + (self.res/2) ) )        
            elif centered is False:
                #regular
                self.cellVolume = self.res * ( _np.cos( self.alpha - (self.res/2) ) - _np.cos( self.alpha + (self.res/2) ) )
                #polar=0deg
                self.cellVolume[0,:] = self.res * ( _np.cos( self.alpha[0,:] - (self.res/2) ) - _np.cos( self.alpha[0,:] ) )
                #polar=90deg
                self.cellVolume[-1,:] = self.res * ( _np.cos( self.alpha[-1,:] ) - _np.cos( self.alpha[-1,:] + (self.res/2) ) ) 

        elif pf_type == 'jul':

            """ HB-2B """

            for i,(f,h) in enumerate(zip(files,hkls)):

                if f.endswith('.jul') and _os.path.exists(f):

                    temp = _np.genfromtxt(f,skip_header=2)

                    self.data[i] = temp[:,2]

                    temp[:,1] = _np.where(temp[:,1] < 0, temp[:,1]+360, temp[:,1])
                    self.y_pol[i] = _np.deg2rad(temp[:,:2]) #alpha, beta
                    self.y[i] = _SPHtoXYZ( self.y_pol[i][:,1], self.y_pol[i][:,0] )

                elif _os.path.exists(f) is False: raise ValueError('file path not found')                    
                else: raise ValueError('not .jul file format')

            self.subtype = 'jul'       

        elif pf_type == 'blank':

            self.subtype = 'blank'

        else: raise NotImplementedError('pf type not recognized')
        
    def plot( self, plt_type='contour', proj='earea', cmap='magma_r', contourlevels='equal', pfs='all', x_direction='N', axes_labels='XY' ):
        
        """
        plotting utility
        """

        #TODO: accept arbitrary pf indices
        if pfs == 'all': ax_cnt = self._numHKL
        else: ax_cnt = pfs
            
        if ax_cnt > 4: 
            cols = 4
            rows = _np.divmod(ax_cnt, 4)
        else:
            cols = ax_cnt
            rows = 1

        if axes_labels == 'XY': axes_labels = {'X':'X', 'Y':'Y'}
        else: pass

        if plt_type == 'scatter':
            
            if hasattr(self,'alpha'):
                #TODO:fix this requirement - all under one scatter func
                _scatterPlot( self.alpha,
                            self.beta,
                            self.data,
                            cols,
                            rows,
                            proj,
                            cmap,
                            axes_labels,
                            x_direction='N')
            
            else:

                _sparseScatterPlot( self.y_pol,
                                    self.data,
                                    cols,
                                    rows,
                                    proj,
                                    cmap,
                                    axes_labels,
                                    x_direction='N')


        elif plt_type == 'contour': 
            if self.subtype == 'poleFig' or self.subtype == 'recalc':
                _contourPlot( self.alpha,
                              self.beta,
                              self.data,
                              cols,
                              rows,
                              proj,
                              cmap,
                              axes_labels, 
                              contourlevels,
                              filled=True,
                              x_direction='N')

            else: raise NotImplementedError('contour not supported for subtype')
            
    def correct( self, bkgd=None, defocus=None ):

        """
        correct for background and defocussing effects from lab diffractometer

        modifies data with corrected value
        stores:
            data: return print(corrected data
            raw_data: original data
            bkdg: background 
        """

        self.raw_data = _copy.deepcopy(self.data)
        self.corrections = []

        if bkgd is not None:

            # background correction
            if bkgd.subtype != 'bkgd': raise ValueError('bkgd pf not supplied')
            else:

                for i,d in self.raw_data.items(): 

                    if len(bkgd.data[i].shape) == 1: #1D array

                        for bi,bv in enumerate(bkgd.data[i]):

                            self.data[i][bi,:] = d[bi,:] - bv                        

                    else: #2D array
                        
                        if bkgd.data[i].shape[0] != d.shape[0]: raise ValueError("bkgd doesn't match chi tilt for pf")
                        elif bkgd.data[i].shape[1] == 2: self.data[i] = d - bkgd.data[i]
                        elif bkgd.data[i].shape[1] == 1:

                            for bi,bv in enumerate(bkgd.data[i]):

                                self.data[i][bi,:] = d[bi,:] - bv
                    
                    self.data[i] = _np.where(self.data[i] < 0, 5, self.data[i])

            self.corrections.append('bkgd')
            

        if defocus is not None:

            # defocus correction
            if defocus.subtype != 'defocus': raise ValueError('defocus pf not supplied')
            else:

                for i,d in self.data.items():

                    # alpha (chi tilt) not equivalent
                    if defocus.data[i].shape[0] != d.shape[0]: raise ValueError("bkgd doesn't match chi tilt for pf")
                    elif defocus.data[i].shape[1] == 2: raise NotImplementedError("not supported")
                    elif defocus.data[i].shape[1] == 1:

                        for bi,bv in enumerate(defocus.data[i]):
                            
                            corr_factor = defocus.data[i][0] / bv
                            self.data[i][bi,:] = d[bi,:] * corr_factor


            self.corrections.append('defocus')

    def normalize( self ):

        """
        normalize pf, based on cell area 
        """

        temp = {}

        #check if normalized
        for i, d in self.data.items():

            if d.shape[0] < 18: 

                # _tqdm.write('warning: only preliminary normalization for incomplete pole figs')

                temp = _np.sum( _np.ravel(d) * _np.ravel(self.cellVolume) ) / _np.sum( _np.ravel(self.cellVolume) )

                # if temp != 1: 

                pg_dg = d * self.cellVolume

                norm = _np.sum( _np.ravel( pg_dg ) ) / _np.sum( _np.ravel( self.cellVolume ) )

                self.data[i] = ( 1 / norm ) * d   
            
            else:

                temp = _np.sum( _np.ravel(d) * _np.ravel(self.cellVolume) ) / _np.sum( _np.ravel(self.cellVolume) ) 

                if temp != 1: 

                    pg_dg = d * self.cellVolume

                    norm = _np.sum( _np.ravel( pg_dg ) ) / ( 2 * _np.pi )

                    self.data[i] = ( 1 / norm ) * d

    def _interpolate( self, res, grid=None, intMethod='linear' ):

        """
        interpolate data points to normal grid | (res x res) cell size
        only for irregular spaced data

        not working
        """        

        self.input_data = _copy.deepcopy(self.data)
        self.res = res

        if grid is None:

            if hasattr(self,'y_pol'): pass
            else: 
                self.y_pol = {}
                for i in range(self._numHKL): self.y_pol[i] = _XYZtoSPH(self.y[i],proj='none') 
                
            #polar
            self._alphasteps = _np.arange(0,_np.pi/2+self.res,self.res)
            #azimuth
            self._betasteps = _np.arange(0,2*_np.pi,self.res)
            #grids
            self.alpha, self.beta = _np.meshgrid(self._alphasteps,self._betasteps,indexing='ij')

            for i in range(self._numHKL):

                self.data[i] = abs(_griddata(self.y_pol[i], _np.ravel(self.data[i]), (self.alpha,self.beta), method=intMethod, fill_value=0.05))

            self.cellVolume = self.res * ( _np.cos( self.alpha - (self.res/2) ) - _np.cos( self.alpha + (self.res/2) ) ) 

        else: raise NotImplementedError('not setup yet')

    def rotate( self, g, eps=1E-7 ):

        """
        rotates pfs by a given g matrix
        """

        if g.shape != (3,3): raise ValueError('g not 3x3')
        if abs(1 - _np.linalg.det(g)) > eps: raise ValueError('g not proper orthogonal')

        if hasattr(self,'y'):

            for i in range(self._numHKL):

                for yi,y in enumerate(self.y[i]):

                    self.y[i][yi,:] = _np.dot(g,y)
                    

        else: raise ValueError('pf subtype not supported - no y')

    def export( self, location, sampleName=None ):

        """
        export pfs for MTEX
        uses .jul format
        """

        if not _os.path.exists(location): _os.mkdir(location)

        for hi,h in enumerate(self.hkls):
            
            hstr = ''.join([str(i) for i in h])

            if sampleName: fname = sampleName+'_pf_'+hstr+'.jul'
            else: fname = 'pf_'+hstr+'.jul'

            fname = _os.path.join(location,fname)

            print(fname)
            writeArr = _np.column_stack((_np.rad2deg(_np.ravel(self.alpha)),
                                        _np.rad2deg(_np.ravel(self.beta)),
                                        _np.ravel(self.data[hi])))
        
            with open(fname,'w') as f:
                f.write('pyTex output\n')
                f.write(' alpha\tbeta\tintensity\n')
                _np.savetxt(f,
                           writeArr,
                           fmt=('%.6f','%.6f','%.6f'),
                           delimiter='\t',
                           newline='\n')

    def export_beartex( self, location, sampleName=None ):

        """
        export pfs in Beartex format

        """
        if not _os.path.exists(location): _os.mkdir(location)

        if sampleName: fname = sampleName+'_pf_out.xpc'
        else: fname = 'pf_out.jul'

        with open(fname, 'w') as fout:

            for hi,h in enumerate(self.hkls):
            
                if self.ss == None:
                    ss = 1
                elif self.ss == 'mmm':
                    ss = 3

                if self.cs == 'm3m' or self.cs == 'm-3m':
                    cs = 7
                else:
                    raise ValueError('bad laue group')


                hstr = ','.join([str(i) for i in h])
                nameLen = len(sampleName)
                gap     = 80 - nameLen ## how many spaces to add
                header = [sampleName]
                for i in range(gap):
                    header.append(' ')
                header[-1] = '#'
                header.append('\r\n\r\n\r\n\r\n\r\n')
                header = ''.join(header)
                fout.write(header)
                fout.write('{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}{:>5d}{:>5d}\r\n'.format(1.0000,1.0000,1.0000,90.0,90.0,90.0,cs,ss))
                fout.write(' {:>3d}{:>3d}{:>3d}{:>5.1f}{:>5.1f}{:>5.1f}{:>5.1f}{:>5.1f}{:>5.1f}{:>2d}{:>2d}\r\n'.format(h[0],
                                                                                                                   h[1],
                                                                                                                   h[2],
                                                                                                                   0.0,
                                                                                                                   90.0,
                                                                                                                   5.0,
                                                                                                                   0.0,
                                                                                                                   360.0,
                                                                                                                   5.0,
                                                                                                                   1,
                                                                                                                   1))

                data = _np.ravel(self.data[hi])
                # sort = _np.lexsort((_np.rad2deg(_np.ravel(self.alpha)),_np.rad2deg(_np.ravel(self.beta))))

                # writeArr = _np.column_stack((_np.rad2deg(_np.ravel(self.alpha)),
                #                             _np.rad2deg(_np.ravel(self.beta)),
                #                             _np.ravel(self.data[hi])))
                
                data = data.reshape((76,18))

                for ri in range(data.shape[0]):
                    
                    data_out = data[ri,:] * 100

                    data_out = ['{:>4d}'.format(int(d)) for d in data_out]
                    data_out = ''.join(data_out)
                    fout.write(' {}\r\n'.format(data_out))

                fout.write('\r\n')

            fout.write('\r\n')

    @staticmethod                      
    def genGrid( res, radians=False, centered=False, ret_ab=False, ret_steps=False, ret_xyz=False, offset=False ):
        
        """
        Returns ndarray of full grid points on pole figure
        
        Inputs:
            full: returns full NxN PF grid - N = res
            ret_ab: returns alpha, beta mgrids
        """
        
        k = 0

        #define bounds
        if radians is True:
            beta_max = 2*_np.pi
            alpha_max = _np.pi/2
            if res > 2*_np.pi:
                print('Are you sure the resolution is radians?')
        elif radians is False:
            beta_max = 360
            alpha_max = 90

        #centered grid
        if centered is True:
            alpha_start = res/2
            beta_start = res/2
            alpha_max = alpha_max - (res/2)
            beta_max = beta_max - (res/2)
        else:
            alpha_start = 0
            if radians is True: 
                beta_start = _np.deg2rad(2.5)
                beta_max = 2*_np.pi - _np.deg2rad(2.5)
            elif radians is False: 
                beta_start = 2.5
                beta_max = 357.5

        #azimuth
        betasteps = _np.arange(beta_start,beta_max+res,res)
        #polar
        alphasteps = _np.arange(alpha_start,alpha_max+res,res)
        
        #grids
        alp, bet = _np.meshgrid(alphasteps,betasteps,indexing='ij')
        
        pf_grid = _np.zeros((len(alphasteps), len(betasteps)))

        for a in range(len(alphasteps)):
            for b in range(len(betasteps)):
        
                pf_grid[a,b] = k
                k += 1

        xyz = _SPHtoXYZ( _np.ravel(bet), _np.ravel(alp), offset )

        out = [pf_grid]

        if ret_ab is True: 
            out.append(alp)
            out.append(bet)
        if ret_steps is True: 
            out.append(alphasteps)
            out.append(betasteps)
        if ret_xyz is True:
            out.append(xyz)
        return out 

class inversePoleFigure(object):

    """
    inverse pole figure constructor
    """

    def __init__( self, files, vector ):

        pass

class pointer ( object ):
    
    """
    constructor for pointer object
    """
    
    def __init__( self, pfs, orient_dist, tube=False, tube_prop=None ):

        pass

class OD( object ):
    
    """
    Parent class for orientation distribution objects
    """
    
    def __init__( self, crystSym, sampleSym ):
        
        """
        inputs:
            crystalSym : 'm-3', 'm-3m'
            sampleSym  : sample symmetry ('1','mmm')        
        """

        self.cs = crystSym
        self.ss = sampleSym
        
class euler( OD ):
    
    """
    Extends OD to Euler (Bunge) space

    inputs:
        cellSize: grid size (radians)
    """

    def __init__( self, cellSize, crystalSym, sampleSym, weights=None, pointer=None, centered=True, convention='bunge' ):
        
        super().__init__( crystalSym, sampleSym )

        if '1' != sampleSym: 
            tempOps = _genSymOps(sampleSym)
            # tempOps = _np.swapaxes(tempOps,2,0)
            # tempOps = tempOps[_np.where( _np.linalg.det(tempOps) == 1 )]
            self.smpl_symOps = _quat.from_matrix(tempOps)
        else: 
            self.smpl_symOps = _quat.from_matrix(_np.identity(3))[None,:]

        self.CS = crystalSym
        self.SS = sampleSym

        if convention == 'bunge':

            self._axesConvention = 'bunge'
            self._axesNames     = {1: 'phi1',
                                   2: 'Phi',
                                   3: 'phi2'}

        elif convention == 'matthies':

            """
            matthies angle convetion

            alpha, beta, gamma

            used in MAUD/Beartex
            """

            self._axesConvention = 'matthies'
            self._axesNames     = {1: 'alpha',
                                   2: 'beta',
                                   3: 'gamma'}

        """
        this should be consistent for every convention

        """
        self.res = cellSize
        
        # set boundary in Bunge space (not rigorous for cubic)
        # if sampleSym == '1': self._ax1max = _np.deg2rad(360)
        # elif sampleSym == 'm': self._ax1max = _np.deg2rad(180)
        # elif sampleSym == 'mmm': self._ax1max = _np.deg2rad(360)
        # else: raise ValueError('invalid sampleSym')

        self._ax1max = _np.deg2rad(360)

        if crystalSym == 'm-3m' or crystalSym == '432': 
            self._ax2max = _np.deg2rad(90)
            self._ax3max = _np.deg2rad(90)
        elif crystalSym == 'm-3' or crystalSym == '23': raise NotImplementedError('coming soon..')
        else: raise ValueError('invalid crystalSym, only cubic so far..')

        # setup grid
        self._ax1range = _np.arange(0,self._ax1max+cellSize,cellSize)
        self._ax2range = _np.arange(0,self._ax2max+cellSize,cellSize)
        self._ax3range = _np.arange(0,self._ax3max+cellSize,cellSize)

        # centroid grid
        self._ax1cen_range = _np.arange( (cellSize/2),( self._ax1max-(cellSize/2) )+cellSize,cellSize )
        self._ax2cen_range = _np.arange( (cellSize/2),( self._ax2max-(cellSize/2) )+cellSize,cellSize )
        self._ax3cen_range = _np.arange( (cellSize/2),( self._ax3max-(cellSize/2) )+cellSize,cellSize )

        self.ax3, self.ax2, self.ax1 = _np.meshgrid(self._ax3range, self._ax2range, self._ax1range, indexing = 'ij')
        self.ax3cen, self.ax2cen, self.ax1cen = _np.meshgrid(self._ax3cen_range, self._ax2cen_range, self._ax1cen_range, indexing = 'ij')

        self.g, self.angList = _eu2om((self.ax1cen,self.ax2cen,self.ax3cen),out='mdarray')
        
        ## vol integral of sin(ax2) dax2 dax1 dax3 
        self.volume = (-_np.cos(self._ax2max) +_np.cos(0)) * self._ax1max * self._ax3max
        #for centered grid
        if centered: 
            
            self.cellVolume = self.res * self.res * ( _np.cos( self.ax2cen - (self.res/2) ) - _np.cos( self.ax2cen + (self.res/2) ) )
            
            temp = _np.zeros(( _np.product(self.ax1cen.shape ) , 3))
            # quaternion grid
            for ang_i, md_i in enumerate(_np.ndindex(self.ax1cen.shape)):
                temp[ang_i,:] = _np.array( ( self.ax1cen[md_i], self.ax2cen[md_i], self.ax3cen[md_i] ) )
            self.q_grid = _eu2quat(temp).T
            
            self.centered = True

        else: #for uncentered grid

            self.g, self.angList = _eu2om((self.ax1,self.ax2,self.ax3),out='mdarray')

            ax2_zero = (self.ax2 == 0)
            ax2_max = (self.ax2 == _np.max(self.ax2))
        
            ax1_zero = (self.ax1 == 0)
            ax1_max = (self.ax1 == _np.max(self.ax1))

            ax3_zero = (self.ax3 == 0)
            ax3_max = (self.ax3 == _np.max(self.ax3))

            dax1_dax3 = _np.ones_like(self.angList) * self.res * self.res
            #ax1 edge cases - 0.5*Δφ1 + Δφ2
            dax1_dax3[ax1_zero+ax1_max] = 0.5*self.res * self.res
            #ax3 edge cases - Δφ1 + 0.5*Δφ2
            dax1_dax3[ax3_zero+ax3_max] = 0.5*self.res * self.res
            #ax1 and ax3 edge case - 0.5*Δφ1 + 0.5*Δφ2
            dax1_dax3[(ax3_zero+ax3_max)*(ax1_zero+ax1_max)] = 0.5*self.res * 0.5*self.res  

            delta_ax2 = _np.ones_like(self.angList) * ( _np.cos( self.ax2 - (self.res/2) ) - _np.cos( self.ax2 + (self.res/2) ) )
            #ax2 = 0
            delta_ax2[ax2_zero] = ( _np.cos( self.ax2[ax2_zero] ) - _np.cos( self.ax2[ax2_zero] + (self.res/2) ) )
            #ax2 = max
            delta_ax2[ax2_max] = ( _np.cos( self.ax2[ax2_max] - (self.res/2) ) - _np.cos( self.ax2[ax2_max] ) )

            self.cellVolume = dax1_dax3 * delta_ax2 

            temp = _np.zeros(( _np.product(self.ax1.shape ) , 3))
            # quaternion grid
            for ang_i, md_i in enumerate(_np.ndindex(self.ax1.shape)):
                temp[ang_i,:] = _np.array( ( self.ax1[md_i], self.ax2[md_i], self.ax3[md_i] ) )
            self.q_grid = _eu2quat(temp).T

            self.centered = False

        if weights is None: self.weights = _np.zeros_like(self.angList)
        else: self.weights = weights

    @staticmethod
    def _genGrid( res, _ax1max, _ax2max, _ax3max, centered=True, returnList=False ):

        """
        generate grids
        """

        # setup grid
        _ax1range = _np.arange(0,_ax1max+res,res)
        _ax2range = _np.arange(0,_ax2max+res,res)
        _ax3range = _np.arange(0,_ax3max+res,res)

        # centroid grid
        _ax1cen_range = _np.arange( (res/2),( _ax1max-(res/2) )+res,res )
        _ax2cen_range = _np.arange( (res/2),( _ax2max-(res/2) )+res,res )
        _ax3cen_range = _np.arange( (res/2),( _ax3max-(res/2) )+res,res )

        ax3, ax2, ax1 = _np.meshgrid(_ax3range, _ax2range, _ax1range, indexing = 'ij')
        ax3cen, ax2cen, ax1cen = _np.meshgrid(_ax3cen_range, _ax2cen_range, _ax1cen_range, indexing = 'ij')

        g, angList = _eu2om((ax1cen,ax2cen,ax3cen),out='mdarray')
        
        ## vol integral of sin(ax2) dax2 dax1 dax3 
        volume = (-_np.cos(_ax2max) +_np.cos(0)) * _ax1max * _ax3max

        #for centered grid
        if centered: 
            
            cellVolume = res * res * ( _np.cos( ax2cen - (res/2) ) - _np.cos( ax2cen + (res/2) ) )
            
            temp = _np.zeros(( _np.product(ax1cen.shape ) , 3))
            # quaternion grid
            for ang_i, md_i in enumerate(_np.ndindex(ax1cen.shape)):
                temp[ang_i,:] = _np.array( ( ax1cen[md_i], ax2cen[md_i], ax3cen[md_i] ) )
            q_grid = _eu2quat(temp).T

        else: #for uncentered grid

            g, angList = _eu2om((ax1,ax2,ax3),out='mdarray')

            ax2_zero = (ax2 == 0)
            ax2_max = (ax2 == _np.max(ax2))
        
            ax1_zero = (ax1 == 0)
            ax1_max = (ax1 == _np.max(ax1))

            ax3_zero = (ax3 == 0)
            ax3_max = (ax3 == _np.max(ax3))

            dax1_dax3 = _np.ones_like(angList) * res * res
            #ax1 edge cases - 0.5*Δφ1 + Δφ2
            dax1_dax3[ax1_zero+ax1_max] = 1.5*res
            #ax3 edge cases - Δφ1 + 0.5*Δφ2
            dax1_dax3[ax3_zero+ax3_max] = 1.5*res
            #ax1 and ax3 edge case - 0.5*Δφ1 + 0.5*Δφ2
            dax1_dax3[(ax3_zero+ax3_max)*(ax1_zero+ax1_max)] = res  

            delta_ax2 = _np.ones_like(angList) * ( _np.cos( ax2 - (res/2) ) - _np.cos( ax2 + (res/2) ) )
            #ax2 = 0
            delta_ax2[ax2_zero] = ( _np.cos( ax2[ax2_zero] ) - _np.cos( ax2[ax2_zero] + (res/2) ) )
            #ax2 = max
            delta_ax2[ax2_max] = ( _np.cos( ax2[ax2_zero] - (res/2) ) - _np.cos( ax2[ax2_zero] ) )

            cellVolume = dax1_dax3 * delta_ax2 

            temp = _np.zeros(( _np.product(ax1.shape ) , 3))
            # quaternion grid
            for ang_i, md_i in enumerate(_np.ndindex(ax1.shape)):
                temp[ang_i,:] = _np.array( ( ax1[md_i], ax2[md_i], ax3[md_i] ) )
            q_grid = _eu2quat(temp).T
       

        if returnList: return angList
        else: return ax1, ax2, ax3

    @classmethod
    def loadMAUD( cls, file ):

        """
        load in Beartex ODF exported from MAUD

        should have crystal symmetry in file
        assumed triclinic symmetry - if it did have higher symmetry enforced that will be expressed in the coefficients
        also gives it in a 5x5x5 grid, even if larger cell size was used

        """

        # alpha →
        # b
        # e
        # t  
        # a
        # ↓  every 19 rows is new gamma section

        ## matthies
        with open(file,'r') as f:

            #read in odf data
            odf_str = f.readlines()
            for line in odf_str: pass
            odf_txt = _np.genfromtxt(odf_str,skip_header=3)
            print('loaded ODF')
            print('header: "'+odf_str[0].strip('\n')+'"')
            file_sym = int(odf_str[1].split(' ')[0])        

        sym_beartex = {11: ['622'],
                       10: ['6'],
                       9: ['32'],
                       8: ['3'],
                       7: ['432','m-3m'],
                       6: ['23','m-3'],
                       5: ['422'],
                       4: ['4'],
                       3: ['222'],
                       2: ['2'],
                       1: ['1']}

        file_sym = sym_beartex.get(file_sym, lambda: 'Unknown Laue group')

        # this instance will have attributes overwritten
        od = euler(_np.deg2rad(5),file_sym[-1],'1',centered=False,convention='matthies')

        ## this is in matthies
        weights = _np.zeros_like(od.ax1)

        for i,p2 in enumerate(od._ax3range):
            for j,p in enumerate(od._ax2range):
                for k,p1 in enumerate(od._ax1range):

                    weights[i,j,k] = odf_txt[j+i*19,k]

        od.weights = _np.ravel(weights)

        return od

    """ modifiers """

    def normalize( self ):

        temp = _np.sum( self.weights * _np.ravel(self.cellVolume) ) / _np.sum( _np.ravel(self.cellVolume) )

        if temp != 1:

            od_dg = self.weights * _np.ravel(self.cellVolume)

            norm = _np.sum ( _np.ravel(od_dg) ) / _np.sum( _np.ravel( self.cellVolume ) )

            self.weights = ( 1 / norm ) * self.weights
            
    def export( self, fname, vol_norm ):
        
        if self.centered:            
            if vol_norm is True: out = _np.array((self.ax1cen.flatten(),self.ax2cen.flatten(),self.ax3cen.flatten(),self.weights*self.cellVolume.flatten())).T
            elif vol_norm is False: out = _np.array((self.ax1cen.flatten(),self.ax2cen.flatten(),self.ax3cen.flatten(),self.weights)).T
        elif self.centered is False:
            if vol_norm is True: out = _np.array((self.ax1.flatten(),self.ax2.flatten(),self.ax3.flatten(),self.weights*self.cellVolume.flatten())).T
            elif vol_norm is False: out = _np.array((self.ax1.flatten(),self.ax2.flatten(),self.ax3.flatten(),self.weights)).T            
        
        with open(fname, 'w') as file:
            file.write('#ax1\tax2\tax3\tweight\n')
            _np.savetxt(file,
                        out,
                        fmt=('%.5f','%.5f','%.5f','%.10f'),
                        delimiter='\t',
                        newline='\n')

        return out

    """ metrics """

    def index( self ):

        """
        calculate texture index (F²)
        mean of f(g)²dg  (normalized to volume)
        https://doi.org/10.1107/S002188989700811X
        """

        return _np.sum( _np.ravel( self.cellVolume ) * self.weights**2 )  / _np.sum( _np.ravel( self.cellVolume ) )

    def strength( self ):

        """
        square root of texture index (F), norm
        https://doi.org/10.1107/S002188989700811X
        """        

        return _np.sqrt(self.index())

    def entropy( self, cellVolume=None ):

        """
        calculate entropy (texture disorder)
        f(g)ln(f(g))Δg - normalized to volume
        https://doi.org/10.1107/S002188989700811X
        """

        if cellVolume is None:
            return -_np.sum( _np.ravel( self.cellVolume ) * self.weights * _np.log( self.weights ) ) / _np.sum( _np.ravel( self.cellVolume ) )
        else:
            return -_np.sum( _np.ravel( cellVolume ) * self.weights * _np.log( self.weights ) ) / _np.sum( _np.ravel( cellVolume ) )

    def compVolume( self, ori, rad, degree=True ):

        """
        calculate volume fractions
        """

        ## check for scipy rotation instance
        # if isinstance(ori,_R):
        #     g = ori.as_matrix()

        # elif isinstance(ori,_np.ndarray): pass
            
        if ori.shape == (3,3): #orientation matrix
            ## get symmetric equivalents
            g_sym = _symOri(ori, self.CS, self.SS)

            #try transpose - om2eu assumes bunge convention for matrix
            g_sym = g_sym.transpose((0,2,1))

            # convert to euler
            eu_sym = _om2eu(g_sym)

            # pick fundamental zone
            fz = (eu_sym[:,0] <= self._ax1max) & (eu_sym[:,1] <= self._ax2max) & (eu_sym[:,2] <= self._ax3max)
            fz_idx = _np.nonzero(fz)
            g_fz = g_sym[fz_idx[0],:,:]

            # generate sym ops
            crysSymOps = _genSymOps(self.CS)
            smplSymOps = _genSymOps(self.SS)

            # create Nx3 array of grid points
            if self.centered: eu_grid = _np.array([self.ax1cen.flatten(),self.ax2cen.flatten(),self.ax3cen.flatten()]).T
            else: eu_grid = _np.array([self.ax1.flatten(),self.ax2.flatten(),self.ax3.flatten()]).T

            g_grid  = _eu2om(eu_grid,out='mdarray_2')
            g_grid  = g_grid.transpose((2,0,1))

            trace = {}
            misori = {}
            mo_cell = []

            for gi,g in enumerate(g_fz):    
                
                trace[gi] = []
                misori[gi] = []
                k = 0
                
                for crys_op in crysSymOps:
                    
                    # for smpl_op in smplSymOps:
                    #SCIPY OUTPUTS TRANSPOSE (CRYSTAL->MATRIX) THEREFORE NO TRANSPOSE IS NEEDED
                    temp = g @ g_grid
                    test = crys_op @ temp 
                    trace[gi].append( _np.trace( test,axis1=1,axis2=2 ) )
                    
                    #calculate misorientation
                    mo = _np.arccos( _np.clip( (trace[gi][k] - 1)/2, -1, 1) )

                    #criteria
                    if degree: crit = _np.where(mo <= _np.deg2rad(rad))
                    else: crit = _np.where(mo <= rad)

                    #store cell id, misorientation angle for each sym equiv.
                    misori[gi].append( _np.array( [crit[0], mo[crit]] ).T )
                    k += 1

                # concatenate, pull true min from sym equiv.
                misori[gi] = _np.vstack(misori[gi])
                # mo_cell.append( misori[gi][ _np.argmin(misori[gi][:,1]), 0 ].astype(int) )
                
                mo_cell.append(_np.unique(misori[gi],axis=0)[:,0].T)    
                # misori.append(_np.argmin(_np.arccos((_np.vstack(trace)-1)/2)))
                # k+=1

            mo_cell = _np.unique(_np.hstack(mo_cell).astype(int))

            #total volume of valid cells / norm to entire volume
            vf = _np.sum( self.weights[mo_cell]*self.cellVolume.flatten()[mo_cell] ) / _np.sum(self.weights * self.cellVolume.flatten())
            # vf = self.weights[mo_cell]

        return vf

    """ homogenization """

    def reuss( self, compliance ):

        """
        Reuss bound
        """

        if compliance.shape != (6,6): raise ValueError('Voigt notation please')

        scale = _copy.deepcopy((self.weights * self.cellVolume.flatten())) / _np.sum(self.weights * self.cellVolume.flatten()) 

        Stensor = _voigt2tensor(compliance,compliance=True)
        Stemp = _np.zeros([3,3,3,3])

        for n,wgt in enumerate(scale):
            
            g_temp = _np.copy(self.g[:,:,n])
            
            Sten_Tr = _np.einsum('im,jn,ko,lp,mnop',g_temp,g_temp,g_temp,g_temp,Stensor)
            Stemp += Sten_Tr*wgt

        return _np.linalg.inv(_tensor2voigt(Stemp,compliance=True))

    def voigt( self, stiffness ):

        """
        voigt bound 
        """

        if stiffness.shape != (6,6): raise ValueError('Voigt notation please')

        scale = _copy.deepcopy((self.weights * self.cellVolume.flatten())) / _np.sum(self.weights * self.cellVolume.flatten()) 

        Ctensor = _voigt2tensor(stiffness)
        Ctemp = _np.zeros([3,3,3,3])

        for n,wgt in enumerate(scale):
            
            g_temp = _np.copy(self.g[:,:,n])
            
            Cten_Tr = _np.einsum('im,jn,ko,lp,mnop',g_temp,g_temp,g_temp,g_temp,Ctensor)
            Ctemp += Cten_Tr*wgt

        return _tensor2voigt(Ctemp)        

    def hill( self, elastic, stiffness=True ):

        """
        hill bound - default entry is stiffness in Voigt notation
        """

        if stiffness:

            if elastic.shape != (6,6): raise ValueError('Voigt notation please')
            
            voigt = self.voigt(elastic)
            reuss = self.reuss(_np.linalg.inv(elastic))

            return 0.5*( voigt + reuss )

        else: 

            if elastic.shape != (6,6): raise ValueError('Voigt notation please')
            
            voigt = self.voigt(_np.linalg.inv(elastic))
            reuss = self.reuss(elastic)

            return 0.5*( voigt + reuss )    


    """ e-wimv specific sub-routines """

    def _calcPath( self, path_type, symHKL, yset, phi, rad, euc_rad, q_tree, hkls_loop_idx=None):

        def fast_mult(qi, qj):
            """
            used to calculate only the scalar portion of quaternion
            
            fast for misorient.
            
            """

            qi = _np.asarray(qi) 
            qj = _np.asarray(qj)
            
            output = qi[..., 0] * qj[..., 0] - _np.sum(qi[..., 1:] * qj[..., 1:], axis=-1)
            
            return output

        """
        calculate paths through orientation
        
        inputs:
            -path_type    - full, arb, custom
            -symHKL  -  
            -yset    - set of y
            -phi     - angular range [0 2π]
            -rad     - radius of tube
            -q_tree  - scipy KDTree object of q_grid
            -euc_rad - euclidean distance (quaternion) for radius of tube 
        """

        #throw q_grid into positive hemisphere (SO3) for euclidean distance
        qgrid_pos = _np.copy(self.q_grid)
        qgrid_pos[qgrid_pos[:,0] < 0] *= -1

        symOps = _genSymOps(self.cs)
        # symOps = _np.unique(_np.swapaxes(symOps,2,0),axis=0)

        # proper = _np.where( _np.linalg.det(symOps) == 1 ) #proper orthogonal/no inversion
        # quatSymOps = _quat.from_matrix(symOps[proper])
        quatSymOps = _quat.from_matrix(symOps)
        quatSymOps = _np.tile(quatSymOps[:,:,_np.newaxis],(1,1,len(phi)))
        quatSymOps = quatSymOps.transpose((0,2,1))

        ## create symOps matrix for fast multiply
        crys_symOps = _quat.from_matrix(symOps)

        tube_rad = _np.deg2rad(8)
        tube_thres = _np.cos(tube_rad) / 2
        
        cphi = _np.cos(phi/2)
        sphi = _np.sin(phi/2)
        
        # q0 = {}
        # q = {}
        # qf = {}
        
        # axis = {}
        # omega = {}
        
        path_e = {}
        path_q = {}
        
        gridPts = {}
        gridDist = {}
            
        for fi,fam in enumerate(_tqdm(symHKL,desc='Calculating paths')):
            
            path_e[fi] = {}
            path_q[fi] = {}
            
            gridPts[fi] = {}
            gridDist[fi] = {}
            
            # q0[fi] = {}
            # q[fi] = {}
            
            # axis[fi] = {}
            # omega[fi] = {}
            
            """ set proper iterator """
            if isinstance(yset,dict): it = yset[fi]
            else: it = yset
            
            for yi,y in enumerate(_tqdm(it,desc='Looping over y')): 
                
                """ symmetry method """
                # t0 = _time.time()
                #calculate path for single (hkl)
                axis = _np.cross(fam,y)
                axis = axis / _np.linalg.norm(axis,axis=-1)
                omega = _np.arccos(_np.dot(fam,y))

                q0 = _np.hstack( [ _np.cos(omega/2), _np.sin(omega/2) * axis ] )
                q  = _np.hstack( [ cphi[:, _np.newaxis], _np.tile( y, (len(cphi),1) ) * sphi[:, _np.newaxis] ] )
                qf = _quat.multiply(q, q0)

                for smpl_symOp in self.smpl_symOps: 

                    #multiply by sym ops, first sample then crystal
                    qf_smplSym = _quat.multiply(smpl_symOp, qf)
                    qfib = _quat.multiply(qf_smplSym, quatSymOps)
                    #transpose to correct format for conversion
                    qfib = qfib.transpose((1,0,2))
                    
                    #convert to bunge euler
                    ax1, ax2, ax3 = _quat2eu(qfib)
                    
                    ax1 = _np.where(ax1 < 0, ax1 + 2*_np.pi, ax1) #brnng back to 0 - 2pi
                    ax2 = _np.where(ax2 < 0, ax2 + _np.pi, ax2) #brnng back to 0 - pi
                    ax3 = _np.where(ax3 < 0, ax3 + 2*_np.pi, ax3) #brnng back to 0 - 2pi
                    
                    #fundamental zone calc (not true!)
                    eu_fib = _np.stack( (ax1, ax2, ax3), axis=2 )
                    eu_fib = _np.reshape( eu_fib, (eu_fib.shape[0]*eu_fib.shape[1], eu_fib.shape[2]) ) #new method       
            
                    fz = (eu_fib[:,0] <= self._ax1max) & (eu_fib[:,1] <= self._ax2max) & (eu_fib[:,2] <= self._ax3max)
                    fz_idx = _np.nonzero(fz)
                    
                    #pull only unique points? - not sure why there are repeated points, something with symmetry for certain hkls
                    #should only be ~73 points per path, but three fold symmetry is also present
                    path_e[fi][yi],uni_path_idx = _np.unique(eu_fib[fz],return_index=True,axis=0)
                    fib_idx = _np.unravel_index(fz_idx[0], (qfib.shape[0],qfib.shape[1]))            
                    path_q[fi][yi] = qfib[fib_idx][uni_path_idx]

                    # ### new way - true misorientation ###

                    # ## reduce geodesic query size
                    # qfib_pos = _np.copy(qfib[fib_idx])
                    # qfib_pos[qfib_pos[:,0] < 0] *= -1

                    # query = _np.concatenate(q_tree.query_radius(qfib_pos,euc_rad))
                    # query_uni = _np.unique(query)
                    # qgrid_trun_idx = _np.arange(len(self.q_grid))[query_uni] #store indexes to retrieve original grid pts later

                    # ## truncated qgrid
                    # qgrid_trun = _np.copy(self.q_grid[query_uni])
                    # qgrid_trun[qgrid_trun[:,0] < 0] *= -1

                    # ## create qgrid_pos for fast multiply
                    # qgrid_trun_arr = _np.broadcast_to(qgrid_trun,(len(phi),len(qgrid_trun),4))
                    # # len(qgrid_trun) x 73 x 4
                    # qgrid_trun_arr = qgrid_trun_arr.transpose((1,0,2))

                    # ## need to use a new name so don't overwrite
                    # g_C = _np.broadcast_to(crys_symOps, (len(qgrid_trun), len(phi), len(crys_symOps), 4))
                    # # len(crys_symOps) x len(qgrid_trun) x 73 x 4
                    # g_C = g_C.transpose((2,0,1,3))        

                    # ## okay now we have the fiber for each y
                    # # now need to get the true misorientation as a "distance"

                    # ## tricky multi
                    # # q_fp_grid = _quat.multiply(qf, qgrid_pos)
                    # q_fp_grid = _quat.multiply( qgrid_trun_arr, qf )
                    
                    # ## another multi
                    # q_mis = fast_mult( q_fp_grid, g_C )
                    
                    # ## get min angle (along axis = 0 | across 24 equiv.)
                    # ## maximum value - avoid trig
                    # ang_min  = _np.max(q_mis,axis=0)
                
                    # ## get mask (every point within tube)
                    # mask_tube  = (ang_min >= tube_thres)
                    
                    # ## get unique cell points and distances
                    # pts_in_tube = _np.argwhere(mask_tube)
                    # dist_pts_in_tube = ang_min[pts_in_tube[:,0],pts_in_tube[:,1]]
                    
                    # ## combine together
                    # pts_in_tube = _np.hstack((pts_in_tube,dist_pts_in_tube[:,None]))
                    
                    # ## sort by the distance (ascending)
                    # pts_in_tube = pts_in_tube[_np.argsort(-1*pts_in_tube[:,2]),:]

                    # ## get unique indicies
                    # uniq_pts,uniq_idx = _np.unique(pts_in_tube[:,0],return_index=True)

                    # ## get unique distance -- slow...
                    # uniq_dist = _quat.geometry.angle(qgrid_trun[uniq_pts.astype(int)])

                    # ## get unique distances, store both indicies and distances
                    # if yi not in gridPts[fi]:

                    #     gridPts[fi][yi] = [qgrid_trun_idx[uniq_pts.astype(int)]]
                    #     gridDist[fi][yi] = [uniq_dist]
                    
                    # else:
                        
                    #     gridPts[fi][yi].append(qgrid_trun_idx[uniq_pts.astype(int)])
                    #     gridDist[fi][yi].append(uniq_dist)                    

                    ### loop method ### 

                    # try:
                    #     axis = axis / _np.linalg.norm(axis,axis=1)[:,None]
                    # except FloatingPointError:
                    #     return axis
                    
                    # omega = _np.arccos(_np.dot(fam,y))
                    
                    # q0 = {}
                    # q = {}
                    # qf = {}
                    # qfib = _np.zeros((len(phi),len(fam),4))
                    
                    # for hi,HxY in enumerate(axis):
                    
                    #     q0[hi] = _np.hstack( [ _np.cos(omega[hi]/2), _np.sin(omega[hi]/2) * HxY ] )
                    #     q[hi]  = _np.hstack( [ cphi[:, _np.newaxis], _np.tile( y, (len(cphi),1) ) * sphi[:, _np.newaxis] ] )
                        
                    #     qf[hi] = _quat.multiply(q[hi], q0[hi])
                    
                    #     for qi in range(qf[hi].shape[0]):
                            
                    #         qfib[qi,hi,:] = qf[hi][qi,:]
                    
                    # ax1, ax2, ax3 = _quat2eu(qfib)
                    
                    # ax1 = _np.where(ax1 < 0, ax1 + 2*_np.pi, ax1) #bring back to 0 - 2pi
                    # ax2 = _np.where(ax2 < 0, ax2 + _np.pi, ax2) #bring back to 0 - pi
                    # ax3 = _np.where(ax3 < 0, ax3 + 2*_np.pi, ax3) #bring back to 0 - 2pi
                    
                    # eu_fib = _np.stack( (ax1, ax2, ax3), axis=2 )
                    # eu_fib = _np.reshape( eu_fib, (eu_fib.shape[0]*eu_fib.shape[1], eu_fib.shape[2]) ) #new method       
            
                    # fz = (eu_fib[:,0] <= self._ax1max) & (eu_fib[:,1] <= self._ax2max) & (eu_fib[:,2] <= self._ax3max)
                    # fz_idx = _np.nonzero(fz)
                    
                    # path_e[fi][yi] = eu_fib[fz]    
                    # fib_idx = _np.unravel_index(fz_idx[0], (qfib.shape[0],qfib.shape[1]))            
                    # path_q[fi][yi] = qfib[fib_idx]

                    # ## geodesic distance calculation - dot product ###

                    # # reduce geodesic query size 
                    # qfib_pos = _np.copy(qfib[fib_idx])
                    # qfib_pos[qfib_pos[:,0] < 0] *= -1
                    
                    # query = _np.concatenate(q_tree.query_radius(qfib_pos,euc_rad))
                    # query_uni = _np.unique(query)
                    # qgrid_trun = self.q_grid[query_uni]
                    # qgrid_trun_idx = _np.arange(len(self.q_grid))[query_uni] #store indexes to retrieve original grid pts later
                    
                    # """ distance calc """
                    # temp = _quatMetric(qgrid_trun,qfib[fib_idx])
                    # """ find tube """
                    # tube = (temp <= rad)
                    # temp = _np.column_stack((_np.argwhere(tube)[:,0],temp[tube]))
                    
                    # """ round very small values """
                    # temp = _np.round(temp, decimals=7)
                    
                    # """ move values at zero to very small (1E-5) """
                    # temp[:,1] = _np.where(temp[:,1] == 0, 1E-5, temp[:,1])
                    
                    # """ sort by min distance """
                    # temp = temp[_np.argsort(temp[:,1],axis=0)]
                    # """ return unique pts (first in list) """
                    # uni_pts = _np.unique(temp[:,0],return_index=True)                    

                    # if yi not in gridPts[fi]:

                    #     gridPts[fi][yi] = [qgrid_trun_idx[uni_pts[0].astype(int)]]
                    #     gridDist[fi][yi] = [temp[uni_pts[1],1]]
                    
                    # else:
                        
                    #     gridPts[fi][yi].append(qgrid_trun_idx[uni_pts[0].astype(int)])
                    #     gridDist[fi][yi].append(temp[uni_pts[1],1])

                    """ euclidean distance calculation - KDTree """
                    
                    qfib_pos = _np.copy(qfib[fib_idx])
                    qfib_pos[qfib_pos[:,0] < 0] *= -1
                    
                    # returns tuple - first array are points, second array is distances
                    query = q_tree.query_radius(qfib_pos,euc_rad,return_distance=True)
                
                    # concatenate arrays
                    query = _np.column_stack([_np.concatenate(ar) for ar in query])
                    
                    # round very small values
                    query = _np.round(query, decimals=7)
                    
                    # move values at zero to very small (1E-5)
                    query[:,1] = _np.where(query[:,1] == 0, 1E-5, query[:,1])            
                    
                    # sort by minimum distance - unique function takes first appearance of index
                    query_sort = query[_np.argsort(query[:,1],axis=0)]
                    
                    # return unique points
                    uni_pts = _np.unique(query_sort[:,0],return_index=True)
                    
                    if yi not in gridPts[fi]:

                        gridPts[fi][yi] = [uni_pts[0].astype(int)]
                        gridDist[fi][yi] = [query_sort[uni_pts[1],1]]
                    
                    else:
                        
                        gridPts[fi][yi].append(uni_pts[0].astype(int))
                        gridDist[fi][yi].append(query_sort[uni_pts[1],1])

                gridDist[fi][yi] = _np.concatenate(gridDist[fi][yi])
                gridPts[fi][yi]  = _np.concatenate(gridPts[fi][yi])

                # t1 = _time.time()
                # print(t1-t0)

        try:
            self.paths[path_type] = {'grid points':gridPts,
                               'grid distances':gridDist,
                               'euler path':path_e,
                               'quat path':path_q}
        except AttributeError: 
            self.paths = {}
            self.paths[path_type] = {'grid points':gridPts,
                                     'grid distances':gridDist,
                                     'euler path':path_e,
                                     'quat path':path_q}

        # TODO: better way to handle this?
        if path_type == 'full_trun':

            self.paths.pop('full_trun')

            path_e_full = {}
            path_q_full = {}

            gridPts_full = {}
            gridDist_full = {}

            #did not calculate duplicate paths (multi orders of reflections) must re-assign
            for i,hi in enumerate(hkls_loop_idx):

                gridPts_full[i] = gridPts[hi]
                gridDist_full[i] = gridDist[hi]
                path_e_full[i] = path_e[hi]
                path_q_full[i] = path_q[hi]

            self.paths['full'] = {'grid points':gridPts_full,
                                     'grid distances':gridDist_full,
                                     'euler path':path_e_full,
                                     'quat path':path_q_full}

    def _calcPointer( self, inv_method, pfs, tube_exp=1 ):

        """
        calculate pointer dictionaries

        both for e-wimv & wimv
        """

        if inv_method == 'e-wimv' and hasattr(self,'paths'):

            self.pointer = {}

            for p_type, p in self.paths.items():

                pf_od = {}
                odwgts_tot = _np.zeros( ( len(pfs.hkls), self.angList.shape[0]*self.angList.shape[1]*self.angList.shape[2] ) )

                test = []

                for hi, h in _tqdm(enumerate(pfs.hkls),desc='Calculating '+p_type+' pointer'):
                    
                    pf_od[hi] = {}
                    
                    for yi in range(len(p['grid points'][hi].keys())):

                        od_cells = p['grid points'][hi][yi]

                        #handle no od_cells
                        if len(od_cells) == 0: continue
                        else:

                            scaled_dist = p['grid distances'][hi][yi]
                            weights = 1 / ( ( abs(scaled_dist) )**tube_exp )
                            
                            if _np.any(weights < 0): raise ValueError('neg weight')
                            if _np.any(weights == 0): raise ValueError('zero weight')
                            
                            pf_od[hi][yi] = {'cell': od_cells, 'weight': weights}
                            
                            odwgts_tot[hi,od_cells.astype(int)] += weights
                        
                odwgts_tot = _np.where(odwgts_tot == 0, 1, odwgts_tot)
                odwgts_tot = 1 / odwgts_tot

                self.pointer[p_type] = {'pf to od':pf_od,
                                        'od weights':odwgts_tot}

        elif inv_method == 'wimv':

            self.pointer = {}

            fullPFgrid, alp, bet = pfs.genGrid(pfs.res,
                                            radians=True,
                                            centered=True,
                                            ret_ab=True)

            xyz = {}   
            sph = {}
            od_pf = {}
            pf_od = {}

            a_binNum = (((_np.pi/2+pfs.res)/pfs.res)-1).astype(int)
            b_binNum = (((2*_np.pi+pfs.res)/pfs.res)-1).astype(int)

            a_bins = _np.histogram_bin_edges(_np.arange(0,_np.pi/2+pfs.res,pfs.res),a_binNum)
            b_bins = _np.histogram_bin_edges(_np.arange(0,2*_np.pi+pfs.res,pfs.res),b_binNum)

            for fi,fam in enumerate(_tqdm(pfs._symHKL,desc='Calculating pointer')):
        
                # Mx3xN array | M - hkl multi. N - # of unique g
                xyz[fi] = _np.dot(fam,self.g)

                sph[fi] = _XYZtoSPH(xyz[fi],proj='none')
                
                od_pf[fi] = {}
                pf_od[fi] = {}
                
                for od_cell in _np.ravel(self.angList):
                    
                    ai = _np.searchsorted(a_bins, sph[fi][:,1,int(od_cell)], side='left')
                    bi = _np.searchsorted(b_bins, sph[fi][:,0,int(od_cell)], side='left')
                    
                    #value directly at pi/2
                    ai = _np.where(ai==a_binNum, a_binNum-1, ai)

                    #value directly at 2pi
                    bi = _np.where(bi==b_binNum, 0, bi)
                
                    pfi = fullPFgrid[ai.astype(int),bi.astype(int)] #pole figure index
                    
                    od_pf[fi][od_cell] = pfi
                        
                    for p in pfi:
                        
                        try:
                            pf_od[fi][p].append(od_cell)
                            
                        except:
                            pf_od[fi][p] = []
                            pf_od[fi][p].append(od_cell)

            self.pointer['full'] = {'pf to od':pf_od,
                                    'od to pf':od_pf,
                                    'poles':xyz,
                                    'poles - sph':sph} 

        elif inv_method == 'e-wimv' and hasattr(self,'paths') is False: raise ValueError('please calculate paths first')

    def calcPF( self, hkls, tube_rad, tube_exp, tube_proj=True ):

        """
        recalculate pole figures

        input list of hkls
        """

        """ dummy pole figure - change this """

        fullPFgrid, alp, bet, xyz_pf = poleFigure.genGrid(res=_np.deg2rad(5),
                                                          radians=True,
                                                          centered=False,
                                                          ret_ab=True,
                                                          ret_xyz=True,
                                                          offset=True)

        #TODO: clean this up
        # if hasattr(self,'pointer') is False or 'full' not in self.pointer:

        """ use sklearn KDTree for reduction of points for query (euclidean) """

        #throw q_grid into positive hemisphere (SO3) for euclidean distance
        qgrid_pos = _np.copy(self.q_grid)
        qgrid_pos[qgrid_pos[:,0] < 0] *= -1
        tree = _KDTree(qgrid_pos)

        #gnomic rotation angle
        rad = ( 1 - _np.cos(tube_rad) ) / 2
        #euclidean rotation angle
        euc_rad = 4*_np.sin(tube_rad)**2

        # rotations around y (integration variable along path)
        phi = _np.linspace(0,2*_np.pi,73)

        """ search for unique hkls to save time during path calculation """

        if isinstance(hkls,list): hkls = _np.vstack(hkls)
        
        pfs = poleFigure([],hkls,self.cs,'blank')

        hkls_loop, uni_hkls_idx, hkls_loop_idx = _np.unique(pfs._normHKLs,axis=0,return_inverse=True,return_index=True)

        if len(uni_hkls_idx) < len(hkls): 
            #time can be saved by only calculating paths for unique reflections
            # symHKL_loop = _symmetrise(self.cs, hkls_loop)
            # symHKL_loop = _normalize(symHKL_loop)

            _tqdm.write('reduced')  
            #calculate paths
            self._calcPath('full_trun', hkls_loop, xyz_pf, phi, rad, euc_rad, tree, hkls_loop_idx=hkls_loop_idx)
            
        #time can't be saved.. calculate all paths
        else: self._calcPath('full', pfs._normHKLs, xyz_pf, phi, rad, euc_rad, tree)     

        """ calculate pointer """

        self._calcPointer('e-wimv', pfs, tube_exp=tube_exp)

        """ recalculate full pole figures """
        ##for reduced grid
        # recalc_pf_full[i] = {}

        numPoles = len(hkls)

        #for 5x5 grid
        recalc_pf = _np.zeros((fullPFgrid.shape[0],fullPFgrid.shape[1],numPoles))
        
        for fi in range(numPoles):
            
            ##for reduced grid
            # recalc_pf_full[i][fi] = _np.zeros(len(xyz_pf))

            # for yi in range(len(xyz_pf)):
            for yi in _np.ravel(fullPFgrid):
                
                if yi in self.pointer['full']['pf to od'][fi]: #pf_cell is defined
                    
                    od_cells = _np.array(self.pointer['full']['pf to od'][fi][yi]['cell'])

                    ##for reduced grid
                    # recalc_pf_full[i][fi][yi] = ( 1 / _np.sum(self.pointer['full']['pf to od'][fi][yi]['weight']) ) * _np.sum( self.pointer['full']['pf to od'][fi][yi]['weight'] * self.weights[od_cells.astype(int)] )
                    
                    #for 5x5 grid
                    ai, bi = _np.divmod(yi, fullPFgrid.shape[1])
                    recalc_pf[int(ai),int(bi),fi] = ( 1 / _np.sum(self.pointer['full']['pf to od'][fi][yi]['weight']) ) * _np.sum( self.pointer['full']['pf to od'][fi][yi]['weight'] * self.weights[od_cells.astype(int)] )
            
        #for reduced grid    
        # recalc_pf_full[i] = _poleFigure(recalc_pf_full[i], pfs.hkls, self.cs, 'recalc', resolution=5, arb_y=xyz_pf)
        #for 5x5 grid
        recalc_pf = poleFigure(recalc_pf, hkls, self.cs, 'recalc', resolution=5)
        recalc_pf.normalize()

        return recalc_pf

    def _calcIPF( self, hkl ):

        pass

    @staticmethod
    def calcDiff( orient_dist1, orient_dist2, type='L1' ):

        """
        calculate difference odfs
        """

        #check for dimensionality match
        if orient_dist1.cs != orient_dist1.cs: raise ValueError('crystal symmetry mismatch')
        if orient_dist1.ss != orient_dist1.ss: raise ValueError('sample symmetry mismatch')

        if type == 'l1':
            diff = _np.abs( orient_dist1.weights - orient_dist2.weights ) 
        if type == 'RP': #TODO: check this? 
            diff = 0.5 * _np.sum( _np.abs( orient_dist1.weights - orient_dist2.weights) ) 

        return euler(orient_dist1.cellSize,orient_dist1.cs,orient_dist1.ss,weights=diff)

    """ plotting """

    def sectionPlot( self, sectionAxis, sectionVal, cmap='magma_r' ):
        
        cols = 1
        rows = 2 #change this later??
        fig, axes = _plt.subplots(rows, cols)
            
        for n,ax in enumerate(axes):   
            
            if self._axesConvention == 'bunge':
                assert sectionAxis.lower() in ['phi1', 'phi', 'phi2']
            elif self._axesConvention == 'matthies':
                assert sectionAxis.lower() in ['alpha', 'beta', 'gamma']

            # get axis number
            for k,v in self._axesNames.items():
                if v.lower() == sectionAxis.lower():
                    axn = k

            if axn == 1: raise NotImplementedError('only phi2/gamma sections..')
            elif axn == 2: raise NotImplementedError('only phi2/gamma sections..')
            elif axn == 3: 
                
                if sectionVal in self._ax3range: section_id = _np.where(self._ax3range == sectionVal)[0] - 1
                else: raise ValueError('section must be in discrete grid')
        
                try: #see if centered
                    pltX, pltY = _np.rad2deg(self.ax1cen[section_id,:,:]), _np.rad2deg(self.ax2cen[section_id,:,:])
                    pltX = pltX.reshape(pltX.shape[1:])
                    pltY = pltY.reshape(pltY.shape[1:])
                    pltWgt = _np.copy(self.weights)
                    pltWgt = pltWgt.reshape(self.ax2cen.shape)
                    pltWgt = pltWgt[section_id,:,:]
                    pltWgt = pltWgt.reshape(pltWgt.shape[1:])

                except:
                    pltX, pltY = _np.rad2deg(self.ax1[section_id,:,:]), _np.rad2deg(self.ax2[section_id,:,:])
                    pltX = pltX.reshape(pltX.shape[1:])
                    pltY = pltY.reshape(pltY.shape[1:])                    
                    pltWgt = _np.copy(self.weights)
                    pltWgt = pltWgt.reshape(self.ax2.shape)
                    pltWgt = pltWgt[section_id,:,:]
                    pltWgt = pltWgt.reshape(pltWgt.shape[1:])

            
            pt = ax.contourf(pltX,pltY,pltWgt,cmap=cmap)
            fig.colorbar(pt, ax=ax, fraction=0.046, pad=0.04) 

    def plot3d( self, n_contours=10, contour_range=None ):

        """
        3d plot using Mayavi (VTK)
        """

        # fig = _mlab.figure(bgcolor=(0.75,0.75,0.75))

        #reshape pts
        if self.centered: data = _np.copy(self.weights.reshape(self.ax1cen.shape))
        else: data = _np.copy(self.weights.reshape(self.ax1.shape))
        
        #round small values (<1E-5)
        # data[data < 1E-5] = 0

        #needs work
        # vol = _mlab.pipeline.volume(_mlab.pipeline.scalar_field(data), vmin=0, vmax=0.8)
        # vol.volume_mapper_type = 'FixedPointVolumeRayCastMapper'

        # TODO: add option for custom contours?
        #plot contours
        cont = _mlab.pipeline.contour_surface(_mlab.pipeline.scalar_field(data),
                                            contours=list(_np.linspace(0,_np.max(data),n_contours)),
                                            transparent=True)

        #plot grid outline box
        _mlab.outline(color=(0,0,0),
                      extent=[1, data.shape[0],
                              1, data.shape[1],
                              1, data.shape[2]])

        ax = _mlab.axes(color=(0,0,0),
                        xlabel=self._axesNames[3],
                        ylabel=self._axesNames[2],
                        zlabel=self._axesNames[1],
                        ranges=[0, _np.rad2deg(self._ax3max),
                                0, _np.rad2deg(self._ax2max),
                                0, _np.rad2deg(self._ax1max)])  

        ax.axes.number_of_labels = 5
        ax.axes.corner_offset = 0.04
        #font size doesn't work @ mayavi v4.7.1
        ax.axes.font_factor = 1
        #adjust ratio of font size between axis title/label?
        ax.label_text_property.line_offset = 3
        #axis labels
        ax.label_text_property.font_family = 'arial'
        ax.label_text_property.shadow = True
        ax.label_text_property.bold = True
        ax.label_text_property.italic = False
        #axis titles
        ax.title_text_property.shadow = True
        ax.title_text_property.bold = True
        ax.title_text_property.italic = False

        #colorbar key
        cbar = _mlab.scalarbar(cont)
        cbar.shadow = True

        if contour_range:
            cbar.use_default_range = False
            if isinstance(contour_range,_np.ndarray): cbar.data_range = contour_range
            else: cbar.data_range = _np.array(contour_range)
            
        cbar.number_of_labels = 10
        #adjust label position
        cbar.label_text_property.justification = 'centered'
        cbar.label_text_property.font_family = 'arial'
        cbar.scalar_bar.text_pad = 10
        cbar.scalar_bar.unconstrained_font_size = True
        cbar.label_text_property.italic = False
        cbar.label_text_property.font_size = 20
        #turn off parallel projection
        _mlab.gcf().scene.parallel_projection = False

        _mlab.show(stop=True)

        # return fig

class rodrigues( OD ):
    
    """
    Extends OD to rodrigues space
    """
        
    def __init__( self, gridPtSize, crystalSym, sampleSym ):
                 
        super(rodrigues, self).__init__( crystalSym, sampleSym )

#default
if __name__ == "__main__": pass