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
from math import ceil as _ceil

import numpy as _np 
from scipy.interpolate import griddata as _griddata
import matplotlib.pyplot as _plt
import mayavi.mlab as _mlab
import xrdtools
import rowan as _quat

# #determine if ipython -> import tqdm.notebook
# try:
#     from IPython.Debugger import Tracer
#     debug = Tracer()
#     from tqdm import tqdm as _tqdm
# except ImportError:
#     from tqdm.notebook import tqdm as _tqdm
#     pass # or set "debug" to something else or whatever

try:
    __IPYTHON__
    from tqdm.notebook import tqdm as _tqdm
except NameError:
    from tqdm import tqdm as _tqdm

from .orientation import eu2om as _eu2om
from .orientation import eu2quat as _eu2quat
from .orientation import quat2eu as _quat2eu
from .orientation import quatMetricNumba as _quatMetric
from .utils import symmetrise as _symmetrise
from .utils import normalize as _normalize
from .utils import XYZtoSPH as _XYZtoSPH

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
    
    def __init__( self, files, hkls, cs, pf_type, subtype=None, names=None, resolution=None, arb_y=None ):
        
        """
        list of file names
        list of (hkl) poles
        type of pole fig
            'xrdml'
            'nd'
        optional: names (list str)
        optional: resolution (int)
        """
        
        if pf_type == 'xrdml':
            
            self.data = {}
            self.hkl = hkls
            self.twotheta = {}
            self._numHKL = len(hkls)
            
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

                    #standard method
                    self.cellVolume = self.res * ( _np.cos( self.alpha - (self.res/2) ) - _np.cos( self.alpha + (self.res/2)) )
                    self.cellVolume[0,:] = self.res * ( _np.cos( self.alpha[0,:] + (self.res/2)) ) #only positive side present at alpha=0
                    
                else: raise TypeError('file type not recognized/found, must be ".xrdml"')

            self.symHKL = _symmetrise(cs,hkls)
            self.symHKL = _normalize(self.symHKL)

        elif pf_type == 'nd': 

            self.data = {}
            self.y = {}
            self.y_pol = {}
            self.hkl = hkls
            self._numHKL = len(hkls)

            for i,(f,h) in enumerate(zip(files,hkls)):

                if f.endswith('.pf') and _os.path.exists(f):

                    """ output from pyReducePF """

                    temp = _np.genfromtxt(f)

                    self.data[i] = temp[:,3]
                    self.y[i] = temp[:,:3]
                    self.y_pol[i] = _XYZtoSPH(_np.copy(temp[:,:3]),proj='none')
                    
            self.subtype = 'nd_poleFig'
            self.symHKL = _symmetrise(cs,hkls)
            self.symHKL = _normalize(self.symHKL)

        elif pf_type == 'mtex': 

            self.data = {}
            self.y = {}
            self.y_pol = {}
            self.hkl = hkls
            self._numHKL = len(hkls)

            for i,(f,h) in enumerate(zip(files,hkls)):

                if f.endswith('.txt') and _os.path.exists(f):

                    """ output from pyReducePF """

                    temp = _np.genfromtxt(f)

                    self.data[i] = temp[:,2]
                    self.y_pol[i] = temp[:,:2]

                    xyz_pf = _np.zeros((self.y_pol[i].shape[0],3))
                    xyz_pf[:,0] = _np.sin( self.y_pol[i][:,1] ) * _np.cos( self.y_pol[i][:,0] )
                    xyz_pf[:,1] = _np.sin( self.y_pol[i][:,1] ) * _np.sin( self.y_pol[i][:,0] )
                    xyz_pf[:,2] = _np.cos( self.y_pol[i][:,1] )

                    self.y[i] = xyz_pf
                    
            self.subtype = 'nd_poleFig'
            self.symHKL = _symmetrise(cs,hkls)
            self.symHKL = _normalize(self.symHKL)            

        elif pf_type == 'recalc': 
            
            self.data = {}
            self.hkl = {}
            self._numHKL = len(hkls)

            #equal grid spacing
            if resolution is None: self.res = None
            else: self.res = resolution

            #polar
            self._alphasteps = _np.arange(self.res/2,
                                          ( 90 - (self.res/2) ) + self.res,
                                          self.res)
            #azimuth
            self._betasteps = _np.arange(self.res/2,
                                         ( 360 - (self.res/2) ) + self.res,
                                         self.res)
            #grids
            self.alpha, self.beta = _np.meshgrid(self._alphasteps,self._betasteps,indexing='ij')

            #convert to rad
            self.alpha = _np.deg2rad(self.alpha)
            #stereo proj
            self.alpha_stereo = (_np.pi/2)*_np.tan(_copy.deepcopy(self.alpha)/2)

            self.beta = _np.deg2rad(self.beta)
            self.res = _np.deg2rad(self.res)

            inter_x = _copy.deepcopy(self.alpha_stereo*_np.cos(self.beta))
            inter_y = _copy.deepcopy(self.alpha_stereo*_np.sin(self.beta))
            
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
                    self.hkl[i] = h
                    
            else:

                for i,h in enumerate(hkls):
                                    
                    #reuse files variable... may not be clear
                    self.data[i] = files[:,:,i]
                    self.hkl[i] = h
                    
            self.subtype = 'recalc'
                
            try:
                self.cellVolume = self.res * ( _np.cos( self.alpha - (self.res/2) ) - _np.cos( self.alpha + (self.res/2) ) )        
            except:
                pass

        elif pf_type == 'jul':

            """ HB-2B """

            self.data = {}
            self.y = {}
            self.y_pol = {}
            self.hkl = hkls
            self._numHKL = len(hkls)

            for i,(f,h) in enumerate(zip(files,hkls)):

                if f.endswith('.jul') and _os.path.exists(f):

                    temp = _np.genfromtxt(f,skip_header=2)

                    self.data[i] = temp[:,2]

                    temp[:,1] = _np.where(temp[:,1] < 0, temp[:,1]+360, temp[:,1])
                    self.y_pol[i] = _np.deg2rad(temp[:,:2]) #alpha, beta

                    xyz_pf = _np.zeros((self.y_pol[i].shape[0],3))
                    xyz_pf[:,0] = _np.sin( self.y_pol[i][:,0] ) * _np.cos( self.y_pol[i][:,1] )
                    xyz_pf[:,1] = _np.sin( self.y_pol[i][:,0] ) * _np.sin( self.y_pol[i][:,1] )
                    xyz_pf[:,2] = _np.cos( self.y_pol[i][:,0] )

                    self.y[i] = xyz_pf
                
                elif _os.path.exists(f) is False: raise ValueError('file path not found')                    
                else: raise ValueError('not .jul file format')

            self.subtype = 'nd_poleFig'
            self.symHKL = _symmetrise(cs,hkls)
            self.symHKL = _normalize(self.symHKL)            

        else: raise NotImplementedError('pf type not recognized')
        
    def plot( self, plt_type='contour', proj='stereo', cmap='magma_r', contourlevels=None, pfs='all', x_direction='N' ):
        
        """
        plotting utility
        """

        if pfs == 'all':

            cols = self._numHKL
            rows = 1 #change this later??

            if self.subtype != 'poleFig':

                if self.subtype != 'recalc':

                    raise NotImplementedError('plotting not supported for bkgd/defocus')

            fig, axes = _plt.subplots(rows, cols, subplot_kw=dict(polar=True))

            if proj == 'stereo': plt_alpha = (_np.pi/2)*_np.tan(_copy.deepcopy(self.alpha)/2)
            elif proj == 'earea': plt_alpha = 2*_np.sin(_copy.deepcopy(self.alpha)/2)
            elif proj == 'none': plt_alpha = _copy.deepcopy(self.alpha)

            #append start/end to create complete plots
            dbeta = _np.diff(self.beta).mean()
            plt_beta  = _np.concatenate( (self.beta, self.beta[:,-1][:,None] + dbeta), axis=1 )
            plt_alpha = _np.concatenate( (plt_alpha,plt_alpha[:,0][:,None]), axis=1 )

            #equal colormap
            if isinstance(contourlevels,str) and 'equal' in contourlevels:
                contourlevels = _np.arange(0,_ceil(_np.max([_np.max(d) for n,d in self.data.items()])),0.5)

            for n,ax in enumerate(axes):
                
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.set_ylim([0,_np.max(plt_alpha)])
                ax.set_theta_zero_location(x_direction) 

                plt_data = _np.concatenate((self.data[n], self.data[n][:, 0:1]), axis=1)

                if plt_type == 'scatter':
                    
                    pt = ax.scatter(plt_beta,plt_alpha)
                    # _plt.close()
                    # raise NotImplementedError('choose contour')
                elif contourlevels is None:
                    pt = ax.contourf(plt_beta,plt_alpha,plt_data,cmap=cmap)
                    fig.colorbar(pt, ax=ax, fraction=0.046, pad=0.04)
                elif contourlevels is not None:
                    pt = ax.contourf(plt_beta,plt_alpha,plt_data,levels=contourlevels,cmap=cmap)
                    if n == (len(axes)-1): fig.colorbar(pt, ax=ax, fraction=0.046, pad=0.04)

                ax.text(0,_np.pi/2,'X',
                        fontsize=8,
                        va='center',
                        ha='center',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
                ax.text(_np.pi/2,_np.pi/2,'Y',
                        fontsize=8,
                        va='center',
                        ha='center',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
                
            _plt.tight_layout()

        else: 

            """ working for nd_poleFig """

            cols = pfs
            rows = 1 #change this later??

            if self.subtype != 'nd_poleFig':

                if self.subtype != 'recalc':

                    raise NotImplementedError('plotting not supported for bkgd/defocus')

                pass
            
            fig, axes = _plt.subplots(rows, cols, subplot_kw=dict(polar=True))

            if proj == 'stereo': plt_alpha = (_np.pi/2)*_np.tan(_copy.deepcopy(self.alpha)/2)
            elif proj == 'earea': plt_alpha = 2*_np.sin(_copy.deepcopy(self.alpha)/2)
            elif proj == 'none': plt_alpha = _copy.deepcopy(self.alpha)
            
            #append start/end to create complete plots
            dbeta = _np.diff(self.beta).mean()
            plt_beta  = _np.concatenate( (self.beta, self.beta[:,-1][:,None] + dbeta), axis=1 )
            plt_alpha = _np.concatenate( (plt_alpha,plt_alpha[:,0][:,None]), axis=1 )
        
            for n,ax in enumerate(axes):
                
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.set_ylim([0,_np.max(plt_alpha)])
                ax.set_theta_zero_location(x_direction)                
                plt_data = _np.concatenate((self.data[n], self.data[n][:, 0:1]), axis=1)
                

                if plt_type == 'scatter':

                    pt = ax.scatter(plt_beta,plt_alpha,s=3)
                    # _plt.close()
                    # raise NotImplementedError('choose contour')
                elif contourlevels is None:
                    pt = ax.contourf(plt_beta,plt_alpha,plt_data,cmap=cmap)                    
                    fig.colorbar(pt, ax=ax, fraction=0.046, pad=0.04)
                elif contourlevels is not None:
                    
                    pt = ax.contourf(plt_beta,plt_alpha,plt_data,levels=contourlevels,cmap=cmap)
                    if n == (len(axes)-1): fig.colorbar(pt, ax=ax, fraction=0.046, pad=0.04)
                    
                ax.text(0,_np.pi/2,'X',
                        fontsize=8,
                        va='center',
                        ha='center',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
                ax.text(_np.pi/2,_np.pi/2,'Y',
                        fontsize=8,
                        va='center',
                        ha='center',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
                
            _plt.tight_layout()

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

    def _interpolate( self, res, grid=None, intMethod='cubic' ):

        """
        interpolate data points to normal grid | (res x res) cell size
        only for irregular spaced data

        not working
        """        

        if self.subtype != 'nd_poleFig': raise ValueError('only supported for nd currently')

        self.input_data = _copy.deepcopy(self.data)
        self.res = res

        if grid is None:

            self.y_pol = {}

            #polar
            self._alphasteps = _np.arange(0,_np.pi/2+self.res,self.res)
            #azimuth
            self._betasteps = _np.arange(_np.deg2rad(2.5),2*_np.pi+self.res,self.res)
            #grids
            self.alpha, self.beta = _np.meshgrid(self._alphasteps,self._betasteps,indexing='ij')

            for i in range(self._numHKL):

                self.y_pol[i] = _XYZtoSPH(self.y[i])
                self.data[i] = abs(_griddata(self.y_pol[i], self.data[i], (self.beta,self.alpha), method=intMethod, fill_value=0.05))

            self.cellVolume = self.res * ( _np.cos( self.alpha - (self.res/2) ) - _np.cos( self.alpha + (self.res/2) ) ) 

        else: raise NotImplementedError('not setup yet')

    def rotate( self, g, eps=1E-7 ):

        """
        rotates pfs by a given g matrix
        """

        if g.shape != (3,3): raise ValueError('g not 3x3')
        if abs(1 - _np.linalg.det(g)) > eps: raise ValueError('g not proper orthogonal')

        if self.subtype == 'nd_poleFig':

            for i in range(self._numHKL):

                for yi,y in enumerate(self.y[i]):

                    self.y[i][yi,:] = _np.dot(g,y)
                    

        else: raise ValueError('only nd supported..')

    def export( self, location, sampleName=None ):

        """
        export pfs for MTEX
        uses .jul format
        """

        if not _os.path.exists(location): _os.mkdir(location)

        for hi,h in self.hkl.items():
            
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
                f.write(' alpha\tbeta\tintensity')
                _np.savetxt(f,
                           writeArr,
                           fmt=('%.6f','%.6f','%.6f'),
                           delimiter='\t',
                           newline='\n')

    @staticmethod                      
    def grid( res, radians=False, cen=False, ret_ab=False, ret_steps=False ):
        
        """
        Returns ndarray of full grid points on pole figure
        
        Inputs:
            full: returns full NxN PF grid - N = res
            ret_ab: returns alpha, beta mgrids
        """
        
        k = 0

        if radians is True:
            beta_max = 2*_np.pi
            alpha_max = _np.pi/2
            if res > 2*_np.pi:
                print('Are you sure the resolution is radians?')
        elif radians is False:
            beta_max = 360
            alpha_max = 90

        if cen is True:
            start = res/2
            alpha_max = alpha_max - (res/2)
            beta_max = beta_max - (res/2)
        else:
            start = 0

        #azimuth
        betasteps = _np.arange(start,beta_max+res,res)
        #polar
        alphasteps = _np.arange(start,alpha_max+res,res)
        
        #grids
        alp, bet = _np.meshgrid(alphasteps,betasteps,indexing='ij')
        
        pf_grid = _np.zeros((len(alphasteps), len(betasteps)))

        for a in range(len(alphasteps)):
            
            for b in range(len(betasteps)):
        
                pf_grid[a,b] = k
                k += 1

        if ret_ab is True: return pf_grid, alp, bet
        elif ret_steps is True: return pf_grid, alphasteps, betasteps
        else: return pf_grid 

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
        
class bunge( OD ):
    
    """
    Extends OD to Euler (Bunge) space

    inputs:
        cellSize: grid size (radians)
    """

    def __init__( self, cellSize, crystalSym, sampleSym, weights=None, pointer=None, centered=True ):
        
        super().__init__( crystalSym, sampleSym )

        self.res = cellSize
        
        # set boundary in Bunge space (not rigorous for cubic)
        if sampleSym == '1': self._phi1max = _np.deg2rad(360)
        elif sampleSym == 'm': self._phi1max = _np.deg2rad(180)
        elif sampleSym == 'mmm': self._phi1max = _np.deg2rad(90)
        else: raise ValueError('invalid sampleSym')

        if crystalSym == 'm-3m' or crystalSym == '432': 
            self._Phimax = _np.deg2rad(90)
            self._phi2max = _np.deg2rad(90)
        elif crystalSym == 'm-3' or crystalSym == '23': raise NotImplementedError('coming soon..')
        else: raise ValueError('invalid crystalSym, only cubic so far..')

        # setup grid
        self._phi1range = _np.arange(0,self._phi1max+cellSize,cellSize)
        self._Phirange = _np.arange(0,self._Phimax+cellSize,cellSize)
        self._phi2range = _np.arange(0,self._phi2max+cellSize,cellSize)

        # centroid grid
        self._phi1cen_range = _np.arange( (cellSize/2),( self._phi1max-(cellSize/2) )+cellSize,cellSize )
        self._Phicen_range = _np.arange( (cellSize/2),( self._Phimax-(cellSize/2) )+cellSize,cellSize )
        self._phi2cen_range = _np.arange( (cellSize/2),( self._phi2max-(cellSize/2) )+cellSize,cellSize )

        self.phi2, self.Phi, self.phi1 = _np.meshgrid(self._phi2range, self._Phirange, self._phi1range, indexing = 'ij')
        self.phi2cen, self.Phicen, self.phi1cen = _np.meshgrid(self._phi2cen_range, self._Phicen_range, self._phi1cen_range, indexing = 'ij')

        self.g, self.bungeList = _eu2om((self.phi1cen,self.Phicen,self.phi2cen),out='mdarray')
        
        ## vol integral of sin(Phi) dPhi dphi1 dphi2 
        self.volume = (-_np.cos(self._Phimax) +_np.cos(0)) * self._phi1max * self._phi2max
        #for centered grid
        if centered: 
            
            self.cellVolume = self.res * self.res * ( _np.cos( self.Phicen - (self.res/2) ) - _np.cos( self.Phicen + (self.res/2) ) )
            
            temp = _np.zeros(( _np.product(self.phi1cen.shape ) , 3))
            # quaternion grid
            for ang_i, md_i in enumerate(_np.ndindex(self.phi1cen.shape)):
                temp[ang_i,:] = _np.array( ( self.phi1cen[md_i], self.Phicen[md_i], self.phi2cen[md_i] ) )
            self.q_grid = _eu2quat(temp).T
            
            self.centered = True

        else: #for uncentered grid

            self.g, self.bungeList = _eu2om((self.phi1,self.Phi,self.phi2),out='mdarray')

            Phi_zero = (self.Phi == 0)
            Phi_max = (self.Phi == _np.max(self.Phi))
        
            phi1_zero = (self.phi1 == 0)
            phi1_max = (self.phi1 == _np.max(self.phi1))

            phi2_zero = (self.phi2 == 0)
            phi2_max = (self.phi2 == _np.max(self.phi2))

            dphi1_dphi2 = _np.ones_like(self.bungeList) * self.res * self.res
            #phi1 edge cases - 0.5*Δφ1 + Δφ2
            dphi1_dphi2[phi1_zero+phi1_max] = 1.5*self.res
            #phi2 edge cases - Δφ1 + 0.5*Δφ2
            dphi1_dphi2[phi2_zero+phi2_max] = 1.5*self.res
            #phi1 and phi2 edge case - 0.5*Δφ1 + 0.5*Δφ2
            dphi1_dphi2[(phi2_zero+phi2_max)*(phi1_zero+phi1_max)] = self.res  

            delta_Phi = _np.ones_like(self.bungeList) * ( _np.cos( self.Phi - (self.res/2) ) - _np.cos( self.Phi + (self.res/2) ) )
            #Phi = 0
            delta_Phi[Phi_zero] = ( _np.cos( self.Phi[Phi_zero] ) - _np.cos( self.Phi[Phi_zero] + (self.res/2) ) )
            #Phi = max
            delta_Phi[Phi_max] = ( _np.cos( self.Phi[Phi_zero] - (self.res/2) ) - _np.cos( self.Phi[Phi_zero] ) )

            self.cellVolume = dphi1_dphi2 * delta_Phi 

            temp = _np.zeros(( _np.product(self.phi1.shape ) , 3))
            # quaternion grid
            for ang_i, md_i in enumerate(_np.ndindex(self.phi1.shape)):
                temp[ang_i,:] = _np.array( ( self.phi1[md_i], self.Phi[md_i], self.phi2[md_i] ) )
            self.q_grid = _eu2quat(temp).T

            self.centered = False

        if weights is None: self.weights = _np.zeros_like(self.bungeList)
        else: self.weights = weights

    @classmethod
    def loadMAUD( cls, file, cellSize, crystalSym, sampleSym ):

        """
        load in Beartex ODF exported from MAUD
        """

        # phi1 →
        # p
        # h
        # i  each 19 rows is new phi2 section
        # ↓

        with open(file,'r') as f:

            #read in odf data
            odf_str = f.readlines()
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
        if any([crystalSym in sym for sym in file_sym]): pass
        else: print('Supplied crystal sym does not match file sym')

        # TODO: handle other than 5deg grid - has same number of values, but with duplicates
        od = bunge(cellSize,crystalSym,sampleSym,centered=False)

        weights = _np.zeros_like(od.phi1)

        for i,p2 in enumerate(od._phi2range):
            for j,p in enumerate(od._Phirange):
                for k,p1 in enumerate(od._phi1range):
                    
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
            
    def export( self, fname ):
        
        out = _np.array((self.phi1cen.flatten(),self.Phicen.flatten(),self.phi2cen.flatten(),self.weights)).T
        
        with open(fname, 'a') as file:
            file.write('#phi1\tPhi\tphi2\tweight\n')
            _np.savetxt(file,
                        out,
                        fmt=('%.5f','%.5f','%.5f','%.5f'),
                        delimiter='\t',
                        newline='\n')

    """ metrics """

    def index( self, cellVolume=None ):

        """
        calculate texture index
        mean squared norm of the OD (normalized to volume)
        https://doi.org/10.1107/S002188989700811X
        """

        if cellVolume is None:
            return _np.sum( _np.ravel( self.cellVolume ) * self.weights**2 ) / _np.sum( _np.ravel( self.cellVolume ) )
        else:
            return _np.sum( _np.ravel( cellVolume ) * self.weights**2 ) / _np.sum( _np.ravel( cellVolume ) )

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

    """ e-wimv specific sub-routines """

    def _calcPath( self, path_type, symHKL, yset, phi, rad, q_tree, euc_rad):

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
        
        cphi = _np.cos(phi/2)
        sphi = _np.sin(phi/2)
        
        q0 = {}
        q = {}
        qf = {}
        
        axis = {}
        omega = {}
        
        path_e = {}
        path_q = {}
        
        gridPts = {}
        gridDist = {}

        egrid_trun = {}
            
        for fi,fam in enumerate(_tqdm(symHKL),desc='Calculating paths'):
            
            path_e[fi] = {}
            path_q[fi] = {}
            
            gridPts[fi] = {}
            gridDist[fi] = {}
            
            egrid_trun[fi] = {}
            
            q0[fi] = {}
            q[fi] = {}
            
            axis[fi] = {}
            omega[fi] = {}
            
            """ set proper iterator """
            if isinstance(yset,dict): it = yset[fi]
            else: it = yset
            
            for yi,y in enumerate(it): 
                
                axis[fi][yi] = _np.cross(fam,y)
                axis[fi][yi] = axis[fi][yi] / _np.linalg.norm(axis[fi][yi],axis=1)[:,None]
                omega[fi][yi] = _np.arccos(_np.dot(fam,y))
                
                q0[fi][yi] = {}
                q[fi][yi] = {}
                qf[yi] = {}
                qfib = _np.zeros((len(phi),len(fam),4))
                
                for hi,HxY in enumerate(axis[fi][yi]):
                
                    q0[fi][yi][hi] = _np.hstack( [ _np.cos(omega[fi][yi][hi]/2), _np.sin(omega[fi][yi][hi]/2) * HxY ] )
                    q[fi][yi][hi]  = _np.hstack( [ cphi[:, _np.newaxis], _np.tile( y, (len(cphi),1) ) * sphi[:, _np.newaxis] ] )
                    
                    qf[yi][hi] = _quat.multiply(q[fi][yi][hi], q0[fi][yi][hi])
                
                    for qi in range(qf[yi][hi].shape[0]):
                        
                        qfib[qi,hi,:] = qf[yi][hi][qi,:]
                
                phi1, Phi, phi2 = _quat2eu(qfib)
                
                phi1 = _np.where(phi1 < 0, phi1 + 2*_np.pi, phi1) #bring back to 0 - 2pi
                Phi = _np.where(Phi < 0, Phi + _np.pi, Phi) #bring back to 0 - pi
                phi2 = _np.where(phi2 < 0, phi2 + 2*_np.pi, phi2) #bring back to 0 - 2pi
                
                eu_fib = _np.stack( (phi1, Phi, phi2), axis=2 )
                eu_fib = _np.reshape( eu_fib, (eu_fib.shape[0]*eu_fib.shape[1], eu_fib.shape[2]) ) #new method       
        
                fz = (eu_fib[:,0] <= self._phi1max) & (eu_fib[:,1] <= self._Phimax) & (eu_fib[:,2] <= self._phi2max)
                fz_idx = _np.nonzero(fz)
                
                path_e[fi][yi] = eu_fib[fz]    
                fib_idx = _np.unravel_index(fz_idx[0], (qfib.shape[0],qfib.shape[1]))            
                path_q[fi][yi] = qfib[fib_idx]
                
                """ reduce geodesic query size """
                qfib_pos = _np.copy(qfib[fib_idx])
                qfib_pos[qfib_pos[:,0] < 0] *= -1
                
                query = _np.concatenate(q_tree.query_radius(qfib_pos,euc_rad))
                query_uni = _np.unique(query)
                qgrid_trun = self.q_grid[query_uni]
                qgrid_trun_idx = _np.arange(len(self.q_grid))[query_uni] #store indexes to retrieve original grid pts later
                
                """ distance calc """
                temp = _quatMetric(qgrid_trun,qfib[fib_idx])
                """ find tube """
                tube = (temp <= rad)
                temp = _np.column_stack((_np.argwhere(tube)[:,0],temp[tube]))
                
                """ round very small values """
                temp = _np.round(temp, decimals=7)
                
                """ move values at zero to very small (1E-5) """
                temp[:,1] = _np.where(temp[:,1] == 0, 1E-5, temp[:,1])
                
                """ sort by min distance """
                temp = temp[_np.argsort(temp[:,1],axis=0)]
                """ return unique pts (first in list) """
                uni_pts = _np.unique(temp[:,0],return_index=True)
                
                gridPts[fi][yi] = qgrid_trun_idx[uni_pts[0].astype(int)]
                gridDist[fi][yi] = temp[uni_pts[1],1]
                
                # egrid_trun[fi][yi] = bungeAngs[query_uni]

        path_opt = {'full': 'full',
                    'arb': 'arb',
                    'custom': 'custom'}

        opt = path_opt.get(path_type, lambda: 'Unknown path type')       

        if opt == 'Unknown path type': raise ValueError('Invalid path type specified..')

        self.paths = {}
        self.paths[opt] = {'grid points':gridPts,
                           'grid distances':gridDist,
                           'euler path':path_e,
                           'quat path':path_q}

    def _calcPointer( self, inv_method, pfs, tube_exp=None ):

        """
        calculate pointer dictionaries

        both for e-wimv & wimv
        """

        if inv_method == 'e-wimv' and hasattr(self,'paths'):

            self.pointer = {}

            for p_type, p in self.paths.items():

                pf_od = {}
                odwgts_tot = _np.zeros( ( len(pfs.hkls), self.bungeList.shape[0]*self.bungeList.shape[1]*self.bungeList.shape[2] ) )

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

            fullPFgrid, alp, bet = pfs.grid(pfs.res,
                                            radians=True,
                                            cen=True,
                                            ret_ab=True)

            xyz = {}   
            sph = {}
            od_pf = {}
            pf_od = {}

            a_bins = _np.histogram_bin_edges(_np.arange(0,_np.pi/2+pfs.res,pfs.res),18)
            b_bins = _np.histogram_bin_edges(_np.arange(0,2*_np.pi+pfs.res,pfs.res),72)

            for fi,fam in enumerate(_tqdm(pfs.symHKL,desc='Calculating pointer')):
        
                # Mx3xN array | M - hkl multi. N - # of unique g
                xyz[fi] = _np.dot(fam,self.g)

                sph[fi] = _XYZtoSPH(xyz[fi],proj='none')
                
                od_pf[fi] = {}
                pf_od[fi] = {}
                
                for od_cell in _np.ravel(self.bungeList):
                    
                    ai = _np.searchsorted(a_bins, sph[fi][:,1,int(od_cell)], side='left')
                    bi = _np.searchsorted(b_bins, sph[fi][:,0,int(od_cell)], side='left')
                    
                    #value directly at pi/2
                    ai = _np.where(ai==18, 17, ai)

                    #value directly at 2pi
                    bi = _np.where(bi==72, 0, bi)
                
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

    def _calcPF( self, hkl ):

        """
        recalculate pole figures
        """

        pass

    def _calcIPF( self, hkl ):

        pass

    def _volume( self, ori ):

        """
        calculate volume fractions
        """

        pass

    def sectionPlot( self, sectionAxis, sectionVal, cmap='magma_r' ):
        
        cols = 1
        rows = 2 #change this later??
        fig, axes = _plt.subplots(rows, cols)
            
        for n,ax in enumerate(axes):   
            
            if sectionAxis == 'phi1': pass
            elif sectionAxis == 'phi': pass
            elif sectionAxis == 'phi2':
                
                if sectionVal in self._phi2range: section_id = _np.where(self._phi2range == sectionVal)[0] - 1
                else: raise ValueError('section must be in discrete grid')
        
                pltX, pltY = _np.rad2deg(self.phi1cen[section_id,:,:]), _np.rad2deg(self.Phicen[section_id,:,:])
                pltX = pltX.reshape(pltX.shape[1:])
                pltY = pltY.reshape(pltY.shape[1:])
                
                pltWgt = _np.copy(self.weights)
                pltWgt = pltWgt.reshape(self.Phicen.shape)
                pltWgt = pltWgt[section_id,:,:]
                pltWgt = pltWgt.reshape(pltWgt.shape[1:])
            
            pt = ax.contourf(pltX,pltY,pltWgt,cmap=cmap)
            fig.colorbar(pt, ax=ax, fraction=0.046, pad=0.04) 

    def plot3d( self, n_contours=10, contour_range=None ):

        """
        3d plot using Mayavi (VTK)
        """

        fig = _mlab.figure(figure='1',bgcolor=(0.75,0.75,0.75))

        #reshape pts
        if self.centered: data = _np.copy(self.weights.reshape(self.phi1cen.shape))
        else: data = _np.copy(self.weights.reshape(self.phi1.shape))
        
        #round small values (<1E-5)
        data[data < 1E-5] = 0

        #needs work
        # vol = _mlab.pipeline.volume(_mlab.pipeline.scalar_field(data), vmin=0, vmax=0.8)
        # vol.volume_mapper_type = 'FixedPointVolumeRayCastMapper'

        # TODO: add option for custom contours?
        #plot contours
        cont = _mlab.pipeline.contour_surface(_mlab.pipeline.scalar_field(data),
                                            contours=list(_np.linspace(0,_np.max(data),n_contours)),
                                            transparent=True)

        #plot grid outline box
        _mlab.outline()

        ax = _mlab.axes(color=(0,0,0),
                        xlabel='phi2',
                        ylabel='Phi',
                        zlabel='phi1',
                        ranges=[0, _np.rad2deg(self._phi2max),
                                0, _np.rad2deg(self._Phimax),
                                0, _np.rad2deg(self._phi1max)])  

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

        return fig

class rodrigues( OD ):
    
    """
    Extends OD to rodrigues space
    """
        
    def __init__( self, gridPtSize, crystalSym, sampleSym ):
                 
        super(rodrigues, self).__init__( crystalSym, sampleSym )
        
### TESTING ###
        
if __name__ == "__main__":
        
    pf111path = '/home/nate/wimv/111pf_2T=38.xrdml'
    pf200path = '/home/nate/wimv/200pf_2T=45.xrdml'
    pf220path = '/home/nate/wimv/220pf_2theta=65.xrdml'
    
    pfs = [pf111path,pf200path,pf220path]
    
    hkl = []
    hkl.append([1,1,1])
    hkl.append([2,0,0])
    hkl.append([2,2,0])
    
    #test = xrdtools.read_xrdml(pf111path)
    
    test = poleFigure(pfs,hkl,'xrdml')
    test.plot()