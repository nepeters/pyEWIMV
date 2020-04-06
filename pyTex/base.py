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
from scipy.interpolate import griddata as _griddata
from sklearn.neighbors import KDTree as _KDTree
import matplotlib.pyplot as _plt
import mayavi.mlab as _mlab
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
    
    def __init__( self, files, hkls, cs, pf_type, subtype=None, names=None, resolution=None, arb_y=None, centered=False ):
        
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
        # if sampleSym == '1': self._phi1max = _np.deg2rad(360)
        # elif sampleSym == 'm': self._phi1max = _np.deg2rad(180)
        # elif sampleSym == 'mmm': self._phi1max = _np.deg2rad(360)
        # else: raise ValueError('invalid sampleSym')

        if '1' != sampleSym: 
            tempOps = _genSymOps(sampleSym)
            tempOps = _np.swapaxes(tempOps,2,0)
            tempOps = tempOps[_np.where( _np.linalg.det(tempOps) == 1 )]
            self.smpl_symOps = _quat.from_matrix(tempOps)
        else: 
            self.smpl_symOps = _quat.from_matrix(_np.identity(3))[None,:]

        self._phi1max = _np.deg2rad(360)

        if crystalSym == 'm-3m' or crystalSym == '432': 
            self._Phimax = _np.deg2rad(90)
            self._phi2max = _np.deg2rad(90)
        elif crystalSym == 'm-3' or crystalSym == '23': raise NotImplementedError('coming soon..')
        else: raise ValueError('invalid crystalSym, only cubic so far..')

        self.CS = crystalSym
        self.SS = sampleSym

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

    @staticmethod
    def _genGrid( res, _phi1max, _Phimax, _phi2max, centered=True, returnList=False ):

        """
        generate grids
        """

        # setup grid
        _phi1range = _np.arange(0,_phi1max+res,res)
        _Phirange = _np.arange(0,_Phimax+res,res)
        _phi2range = _np.arange(0,_phi2max+res,res)

        # centroid grid
        _phi1cen_range = _np.arange( (res/2),( _phi1max-(res/2) )+res,res )
        _Phicen_range = _np.arange( (res/2),( _Phimax-(res/2) )+res,res )
        _phi2cen_range = _np.arange( (res/2),( _phi2max-(res/2) )+res,res )

        phi2, Phi, phi1 = _np.meshgrid(_phi2range, _Phirange, _phi1range, indexing = 'ij')
        phi2cen, Phicen, phi1cen = _np.meshgrid(_phi2cen_range, _Phicen_range, _phi1cen_range, indexing = 'ij')

        g, bungeList = _eu2om((phi1cen,Phicen,phi2cen),out='mdarray')
        
        ## vol integral of sin(Phi) dPhi dphi1 dphi2 
        volume = (-_np.cos(_Phimax) +_np.cos(0)) * _phi1max * _phi2max

        #for centered grid
        if centered: 
            
            cellVolume = res * res * ( _np.cos( Phicen - (res/2) ) - _np.cos( Phicen + (res/2) ) )
            
            temp = _np.zeros(( _np.product(phi1cen.shape ) , 3))
            # quaternion grid
            for ang_i, md_i in enumerate(_np.ndindex(phi1cen.shape)):
                temp[ang_i,:] = _np.array( ( phi1cen[md_i], Phicen[md_i], phi2cen[md_i] ) )
            q_grid = _eu2quat(temp).T

        else: #for uncentered grid

            g, bungeList = _eu2om((phi1,Phi,phi2),out='mdarray')

            Phi_zero = (Phi == 0)
            Phi_max = (Phi == _np.max(Phi))
        
            phi1_zero = (phi1 == 0)
            phi1_max = (phi1 == _np.max(phi1))

            phi2_zero = (phi2 == 0)
            phi2_max = (phi2 == _np.max(phi2))

            dphi1_dphi2 = _np.ones_like(bungeList) * res * res
            #phi1 edge cases - 0.5*Δφ1 + Δφ2
            dphi1_dphi2[phi1_zero+phi1_max] = 1.5*res
            #phi2 edge cases - Δφ1 + 0.5*Δφ2
            dphi1_dphi2[phi2_zero+phi2_max] = 1.5*res
            #phi1 and phi2 edge case - 0.5*Δφ1 + 0.5*Δφ2
            dphi1_dphi2[(phi2_zero+phi2_max)*(phi1_zero+phi1_max)] = res  

            delta_Phi = _np.ones_like(bungeList) * ( _np.cos( Phi - (res/2) ) - _np.cos( Phi + (res/2) ) )
            #Phi = 0
            delta_Phi[Phi_zero] = ( _np.cos( Phi[Phi_zero] ) - _np.cos( Phi[Phi_zero] + (res/2) ) )
            #Phi = max
            delta_Phi[Phi_max] = ( _np.cos( Phi[Phi_zero] - (res/2) ) - _np.cos( Phi[Phi_zero] ) )

            cellVolume = dphi1_dphi2 * delta_Phi 

            temp = _np.zeros(( _np.product(phi1.shape ) , 3))
            # quaternion grid
            for ang_i, md_i in enumerate(_np.ndindex(phi1.shape)):
                temp[ang_i,:] = _np.array( ( phi1[md_i], Phi[md_i], phi2[md_i] ) )
            q_grid = _eu2quat(temp).T
       

        if returnList: return bungeList
        else: return phi1, Phi, phi2

    @classmethod
    def _loadMAUD( cls, file, cellSize, crystalSym, sampleSym ):

        """
        load in Beartex ODF exported from MAUD

        not working - need to fix
        """

        # phi1 →
        # p
        # h
        # i  each 19 rows is new phi2 section
        # ↓

        ## bunge
        with open(file,'r') as f:

            #read in odf data
            odf_str = f.readlines()
            for line in odf_str: pass
            odf_txt = _np.genfromtxt(odf_str,skip_header=3)
            print('loaded ODF')
            print('header: "'+odf_str[0].strip('\n')+'"')
            file_sym = int(odf_str[1].split(' ')[0])        

        ## matthies -- this doesn't match expected, maybe it doesn't use it?
        # with open(file,'r') as f:

        #     odf_data = []
            
        #     #read in odf data
        #     odf_str = f.readlines()
        #     counter = [0,0]
        #     for n,line in enumerate(odf_str):
        #         if n < 3: 
        #             counter[0] += 1
        #             counter[1] += 1
        #         else:
        #             if line in ['\n']: #check for blank line
        #                 odf_data.append(_np.genfromtxt(odf_str[counter[0]:counter[1]]))
        #                 # print(counter)
        #                 counter[0] = n+1
        #                 counter[1] += 1
        #             else: counter[1] += 1 
                        
        #     print('loaded ODF')
        #     print('header: "'+odf_str[0].strip('\n')+'"')
        #     file_sym = int(odf_str[1].split(' ')[0])

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
        ## this instance will have attributes overwritten
        od = bunge(cellSize,crystalSym,sampleSym,centered=False)

        ## bunge
        # override edge corrections
        od.cellVolume = od.res * od.res * ( _np.cos( od.Phi - ( od.res/2 ) ) - _np.cos( od.Phi + ( od.res/2 ) ) )

        weights = _np.zeros_like(od.phi1)

        for i,p2 in enumerate(od._phi2range):
            for j,p in enumerate(od._Phirange):
                for k,p1 in enumerate(od._phi1range):

                    weights[i,j,k] = odf_txt[j+i*19,k]

        od.weights = _np.ravel(weights)

        ## matthies -- this doesn't match expected, maybe it doesn't use it?

        # # set boundary in Matthies space (not rigorous for cubic)
        # if sampleSym == '1': alphaMax = _np.deg2rad(360)
        # elif sampleSym == 'm': alphaMax = _np.deg2rad(180)
        # elif sampleSym == 'mmm': alphaMax = _np.deg2rad(90)
        # else: raise ValueError('invalid sampleSym')

        # if crystalSym == 'm-3m' or crystalSym == '432': 
        #     betaMax = _np.deg2rad(90)
        #     gammaMax = _np.deg2rad(90)
        # elif crystalSym == 'm-3' or crystalSym == '23': raise NotImplementedError('coming soon..')
        # else: raise ValueError('invalid crystalSym, only cubic so far..')

        # gammaRange = _np.arange(0,gammaMax+cellSize,cellSize)
        # betaRange  = _np.arange(0,betaMax+cellSize,cellSize)
        # alphaRange = _np.arange(0,alphaMax+cellSize,cellSize)

        # gam, bet, alp = _np.meshgrid(gammaRange,betaRange,alphaRange,indexing='ij')

        # weights = _np.zeros_like(gam)

        # for gi,g in enumerate(gammaRange):
        #     for bi,b in enumerate(betaRange):
        #         for ai,a in enumerate(alphaRange):
                    
        #             weights[gi,bi,ai] = odf_data[gi][bi,ai]

        # # out = _np.array([gam.flatten(),bet.flatten(),alp.flatten(),weights.flatten()]).T

        # ## shift back to phi1, Phi, phi2
        # phi1 = alp + _np.pi/2
        # Phi  = _np.copy(bet)
        # phi2 = -gam + _np.pi/2
        # phi1 = _np.where(phi1 > alphaMax, phi1 - alphaMax, phi1) #brnng back to 0 - 2pi
        # phi2 = _np.where(phi2 > gammaMax, phi2 - gammaMax, phi2) #brnng back to 0 - 2pi

        # ## phi1_360 index - where to insert new slice
        # phi1_0 = _np.argmax(phi1[0,0,:])

        # ## add duplicate slice (phi1=360) at phi1 = 0
        # phi1 = _np.insert(phi1,0,_np.zeros_like(phi1[:,:,phi1_0]),axis=2)
        # Phi  = _np.insert(Phi,0,Phi[:,:,0],axis=2)
        # phi2 = _np.insert(phi2,0,phi2[:,:,0],axis=2)
        # weights  = _np.insert(weights,0,weights[:,:,phi1_0],axis=2)

        # ## clunky
        # out = _np.array([phi1.flatten(),Phi.flatten(),phi2.flatten(),weights.flatten()]).T
        # out_sort = out[_np.lexsort((out[:,0],out[:,1],out[:,2]))]

        # od.phi1 =  out_sort[:,0].reshape(phi1.shape)
        # od.Phi  =  out_sort[:,1].reshape(Phi.shape)
        # od.phi2 =  out_sort[:,2].reshape(phi2.shape)
        # od.weights = _np.copy(out_sort[:,3])

        # od.g, od.bungeList = _eu2om((od.phi1,od.Phi,od.phi2),out='mdarray')

        # od.cellVolume = cellSize * cellSize * ( _np.cos( Phi - ( cellSize/2 ) ) - _np.cos( Phi + ( cellSize/2 ) ) )

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
            if vol_norm is True: out = _np.array((self.phi1cen.flatten(),self.Phicen.flatten(),self.phi2cen.flatten(),self.weights*self.cellVolume.flatten())).T
            elif vol_norm is False: out = _np.array((self.phi1cen.flatten(),self.Phicen.flatten(),self.phi2cen.flatten(),self.weights)).T
        elif self.centered is False:
            if vol_norm is True: out = _np.array((self.phi1.flatten(),self.Phi.flatten(),self.phi2.flatten(),self.weights*self.cellVolume.flatten())).T
            elif vol_norm is False: out = _np.array((self.phi1.flatten(),self.Phi.flatten(),self.phi2.flatten(),self.weights)).T            
        
        with open(fname, 'w') as file:
            file.write('#phi1\tPhi\tphi2\tweight\n')
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

        symOps = _genSymOps(self.cs)
        symOps = _np.unique(_np.swapaxes(symOps,2,0),axis=0)

        proper = _np.where( _np.linalg.det(symOps) == 1 ) #proper orthogonal/no inversion
        quatSymOps = _quat.from_matrix(symOps[proper])
        quatSymOps = _np.tile(quatSymOps[:,:,_np.newaxis],(1,1,len(phi)))
        quatSymOps = quatSymOps.transpose((0,2,1))
        
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
            
        for fi,fam in enumerate(_tqdm(symHKL,desc='Calculating paths')):
            
            path_e[fi] = {}
            path_q[fi] = {}
            
            gridPts[fi] = {}
            gridDist[fi] = {}
            
            q0[fi] = {}
            q[fi] = {}
            
            axis[fi] = {}
            omega[fi] = {}
            
            """ set proper iterator """
            if isinstance(yset,dict): it = yset[fi]
            else: it = yset
            
            for yi,y in enumerate(it): 
                
                """ symmetry method """
                t0 = _time.time()
                #calculate path for single (hkl)
                axis[fi][yi] = _np.cross(fam,y)
                axis[fi][yi] = axis[fi][yi] / _np.linalg.norm(axis[fi][yi],axis=-1)
                omega[fi][yi] = _np.arccos(_np.dot(fam,y))

                q0[fi][yi] = _np.hstack( [ _np.cos(omega[fi][yi]/2), _np.sin(omega[fi][yi]/2) * axis[fi][yi] ] )
                q[fi][yi]  = _np.hstack( [ cphi[:, _np.newaxis], _np.tile( y, (len(cphi),1) ) * sphi[:, _np.newaxis] ] )
                qf[yi] = _quat.multiply(q[fi][yi], q0[fi][yi])

                for smpl_symOp in self.smpl_symOps: 

                    #multiply by sym ops, first sample then crystal
                    qf_smplSym = _quat.multiply(smpl_symOp, qf[yi])
                    qfib = _quat.multiply(qf_smplSym, quatSymOps)
                    #transpose to correct format for conversion
                    qfib = qfib.transpose((1,0,2))
                    
                    # #convert to bunge euler
                    # phi1, Phi, phi2 = _quat2eu(qfib)
                    
                    # phi1 = _np.where(phi1 < 0, phi1 + 2*_np.pi, phi1) #brnng back to 0 - 2pi
                    # Phi = _np.where(Phi < 0, Phi + _np.pi, Phi) #brnng back to 0 - pi
                    # phi2 = _np.where(phi2 < 0, phi2 + 2*_np.pi, phi2) #brnng back to 0 - 2pi
                    
                    # #fundamental zone calc (not true!)
                    # eu_fib = _np.stack( (phi1, Phi, phi2), axis=2 )
                    # eu_fib = _np.reshape( eu_fib, (eu_fib.shape[0]*eu_fib.shape[1], eu_fib.shape[2]) ) #new method       
            
                    # fz = (eu_fib[:,0] <= self._phi1max) & (eu_fib[:,1] <= self._Phimax) & (eu_fib[:,2] <= self._phi2max)
                    # fz_idx = _np.nonzero(fz)
                    
                    # #pull only unique points? - not sure why there are repeated points, something with symmetry for certain hkls
                    # #should only be ~73 points per path, but three fold symmetry is also present
                    # path_e[fi][yi],uni_path_idx = _np.unique(eu_fib[fz],return_index=True,axis=0)
                    # fib_idx = _np.unravel_index(fz_idx[0], (qfib.shape[0],qfib.shape[1]))            
                    # path_q[fi][yi] = qfib[fib_idx][uni_path_idx]

                    """ loop method """

                    # try:
                    #     axis[fi][yi] = axis[fi][yi] / _np.linalg.norm(axis[fi][yi],axis=1)[:,None]
                    # except FloatingPointError:
                    #     return axis[fi][yi]
                    
                    # omega[fi][yi] = _np.arccos(_np.dot(fam,y))
                    
                    # q0[fi][yi] = {}
                    # q[fi][yi] = {}
                    # qf[yi] = {}
                    # qfib = _np.zeros((len(phi),len(fam),4))
                    
                    # for hi,HxY in enumerate(axis[fi][yi]):
                    
                    #     q0[fi][yi][hi] = _np.hstack( [ _np.cos(omega[fi][yi][hi]/2), _np.sin(omega[fi][yi][hi]/2) * HxY ] )
                    #     q[fi][yi][hi]  = _np.hstack( [ cphi[:, _np.newaxis], _np.tile( y, (len(cphi),1) ) * sphi[:, _np.newaxis] ] )
                        
                    #     qf[yi][hi] = _quat.multiply(q[fi][yi][hi], q0[fi][yi][hi])
                    
                    #     for qi in range(qf[yi][hi].shape[0]):
                            
                    #         qfib[qi,hi,:] = qf[yi][hi][qi,:]
                    
                    # phi1, Phi, phi2 = _quat2eu(qfib)
                    
                    # phi1 = _np.where(phi1 < 0, phi1 + 2*_np.pi, phi1) #bring back to 0 - 2pi
                    # Phi = _np.where(Phi < 0, Phi + _np.pi, Phi) #bring back to 0 - pi
                    # phi2 = _np.where(phi2 < 0, phi2 + 2*_np.pi, phi2) #bring back to 0 - 2pi
                    
                    # eu_fib = _np.stack( (phi1, Phi, phi2), axis=2 )
                    # eu_fib = _np.reshape( eu_fib, (eu_fib.shape[0]*eu_fib.shape[1], eu_fib.shape[2]) ) #new method       
            
                    # fz = (eu_fib[:,0] <= self._phi1max) & (eu_fib[:,1] <= self._Phimax) & (eu_fib[:,2] <= self._phi2max)
                    # fz_idx = _np.nonzero(fz)
                    
                    # path_e[fi][yi] = eu_fib[fz]    
                    # fib_idx = _np.unravel_index(fz_idx[0], (qfib.shape[0],qfib.shape[1]))            
                    # path_q[fi][yi] = qfib[fib_idx]

                    """ euclidean distance calculation - KDTree """
                    
                #     qfib_pos = _np.copy(qfib[fib_idx])
                #     qfib_pos[qfib_pos[:,0] < 0] *= -1
                    
                #     # returns tuple - first array are points, second array is distances
                #     query = q_tree.query_radius(qfib_pos,euc_rad,return_distance=True)
                
                #     # concatenate arrays
                #     query = _np.column_stack([_np.concatenate(ar) for ar in query])
                    
                #     # round very small values
                #     query = _np.round(query, decimals=7)
                    
                #     # move values at zero to very small (1E-5)
                #     query[:,1] = _np.where(query[:,1] == 0, 1E-5, query[:,1])            
                    
                #     # sort by minimum distance - unique function takes first appearance of index
                #     query_sort = query[_np.argsort(query[:,1],axis=0)]
                    
                #     # return unique points
                #     uni_pts = _np.unique(query_sort[:,0],return_index=True)
                    
                #     if yi not in gridPts[fi]:

                #         gridPts[fi][yi] = [uni_pts[0].astype(int)]
                #         gridDist[fi][yi] = [query_sort[uni_pts[1],1]]
                    
                #     else:
                        
                #         gridPts[fi][yi].append(uni_pts[0].astype(int))
                #         gridDist[fi][yi].append(query_sort[uni_pts[1],1])

                # gridDist[fi][yi] = _np.concatenate(gridDist[fi][yi])
                # gridPts[fi][yi]  = _np.concatenate(gridPts[fi][yi])
                
                t1 = _time.time()
                print(t1-t0)
                """ geodesic distance calculation - dot product """
                # """ reduce geodesic query size """
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
                
                # gridPts[fi][yi] = qgrid_trun_idx[uni_pts[0].astype(int)]
                # gridDist[fi][yi] = temp[uni_pts[1],1]
                
                # egrid_trun[fi][yi] = bungeAngs[query_uni]

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

        from scipy.spatial.transform import Rotation as R

        if '1' not in self.SS: 
            ## only works with 'mmm' yet
            smplSymOps = _genSymOps(self.SS)
            smplSymOps = _np.swapaxes(smplSymOps,2,0)
            ## only select proper orthogonal
            smplSymOps = smplSymOps[_np.where( _np.linalg.det(smplSymOps) == 1 )]

        ## try to set pointer to all equivalent points
        # if '1' in self.SS:

            ## true fz is 90,90,90
            # tmp_p1, tmp_P, tmp_p2 = bunge._genGrid(self.cellSize,_np.deg2rad(90),self._Phimax,self._phi2max,centered=self.centered)

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

                        # if '1' not in self.SS: od_cells = p['grid points'][hi][yi]
                        # else:
                        #     ### loop through each grid point, calculate sym. equivalent
                        #     for cell in p['grid points'][hi][yi]:
                                
                        #         #retrieve orient. matrix
                        #         g_temp = R.from_matrix(self.g[cell])

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
                
                for od_cell in _np.ravel(self.bungeList):
                    
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

    def _volume( self, ori ):

        """
        calculate volume fractions
        """

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

        return bunge(orient_dist1.cellSize,orient_dist1.cs,orient_dist1.ss,weights=diff)

    """ plotting """

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
        
                try: #see if centered
                    pltX, pltY = _np.rad2deg(self.phi1cen[section_id,:,:]), _np.rad2deg(self.Phicen[section_id,:,:])
                    pltX = pltX.reshape(pltX.shape[1:])
                    pltY = pltY.reshape(pltY.shape[1:])
                    pltWgt = _np.copy(self.weights)
                    pltWgt = pltWgt.reshape(self.Phicen.shape)
                    pltWgt = pltWgt[section_id,:,:]
                    pltWgt = pltWgt.reshape(pltWgt.shape[1:])

                except:
                    pltX, pltY = _np.rad2deg(self.phi1[section_id,:,:]), _np.rad2deg(self.Phi[section_id,:,:])
                    pltX = pltX.reshape(pltX.shape[1:])
                    pltY = pltY.reshape(pltY.shape[1:])                    
                    pltWgt = _np.copy(self.weights)
                    pltWgt = pltWgt.reshape(self.Phi.shape)
                    pltWgt = pltWgt[section_id,:,:]
                    pltWgt = pltWgt.reshape(pltWgt.shape[1:])

            
            pt = ax.contourf(pltX,pltY,pltWgt,cmap=cmap)
            fig.colorbar(pt, ax=ax, fraction=0.046, pad=0.04) 

    def plot3d( self, n_contours=10, contour_range=None ):

        """
        3d plot using Mayavi (VTK)
        """

        fig = _mlab.figure(bgcolor=(0.75,0.75,0.75))

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
        
#default
if __name__ == "__main__": pass