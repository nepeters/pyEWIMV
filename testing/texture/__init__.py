#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:11:12 2019

@author: nate
"""

"""
texture objects
"""

import os,sys,copy
from math import sin,cos

import numpy as np 
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import xrdtools

sys.path.insert(0,'/Users/nate/wimv')

from utils.orientation import eu2om, symmetrise, normalize, XYZtoSPH

### general functions ###

def _blockPrint():
    sys.stdout = open(os.devnull, 'w')

def _enablePrint():
    sys.stdout = sys.__stdout__
    
### custom objects ###

class poleFigure(object):
    
    """
    pole figure constructor
    """
    
    def __init__( self, files, hkls, cs, pf_type, subtype=None, names=None, resolution=None ):
        
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
                
                if f.endswith('xrdml') and os.path.exists(f):
                    
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
                    if resolution is None: self.res = np.diff(temp['Phi'],1)[0] #pull diff from first 2 points
                    else: self.res = resolution
                    
                    #polar
                    self._alphasteps = np.arange(0,temp['data'].shape[0]*self.res,self.res)
                    #azimuth
                    self._betasteps = np.arange(2.5,temp['data'].shape[1]*self.res,self.res)
                    #grids
                    self.alpha, self.beta = np.meshgrid(self._alphasteps,self._betasteps,indexing='ij')
                    
                    #convert to rad
                    self.alpha = np.deg2rad(self.alpha)
                    self.beta = np.deg2rad(self.beta)
                    self.res = np.deg2rad(self.res)

                    self.cellVolume = self.res * ( np.cos( self.alpha - (self.res/2) ) - np.cos( self.alpha + (self.res/2) ) ) 
                    
                else: raise TypeError('file type not recognized/found, must be ".xrdml"')

            self.symHKL = symmetrise(cs,hkls)
            self.symHKL = normalize(self.symHKL)

        elif pf_type == 'nd': 

            self.data = {}
            self.y = {}
            self.y_pol = {}
            self.hkl = hkls
            self._numHKL = len(hkls)

            for i,(f,h) in enumerate(zip(files,hkls)):

                if f.endswith('.pf') and os.path.exists(f):

                    """ output from pyReducePF """

                    temp = np.genfromtxt(f)

                    self.data[i] = temp[:,3]
                    self.y[i] = temp[:,:3]
                    self.y_pol[i] = XYZtoSPH(self.y[i])
                    
            self.subtype = 'nd_poleFig'
            self.symHKL = symmetrise(cs,hkls)
            self.symHKL = normalize(self.symHKL)

        elif pf_type == 'recalc': 
            
            self.data = {}
            self.hkl = {}
            self._numHKL = len(hkls)
            
            for i,h in enumerate(hkls):
                                
                #trick to reuse variable... may not be clear
                self.data[i] = files[:,:,i]
                self.hkl[i] = h
                
                self.subtype = 'poleFig'
                
                #assuming equal grid spacing
                if resolution is None: self.res = None
                else: self.res = resolution
                
                #polar
                self._alphasteps = np.arange(0,files.shape[0]*self.res,self.res)
                #azimuth
                self._betasteps = np.arange(2.5,files.shape[1]*self.res,self.res)
                #grids
                self.alpha, self.beta = np.meshgrid(self._alphasteps,self._betasteps,indexing='ij')
                
                #convert to rad
                self.alpha = np.deg2rad(self.alpha)
                self.beta = np.deg2rad(self.beta)
                self.res = np.deg2rad(self.res)

            try:
                self.cellVolume = self.res * ( np.cos( self.alpha - (self.res/2) ) - np.cos( self.alpha + (self.res/2) ) )        
            except:
                pass

        else: raise NotImplementedError('pf type not recognized')
        
    def plot( self, plt_type='contour', cmap='viridis', contourlevels=None, pfs='all' ):
        
        """
        plotting utility
        """

        if pfs == 'all':

            cols = self._numHKL
            rows = 1 #change this later??

            if self.subtype != 'poleFig':

                raise NotImplementedError('plotting not supported for bkgd/defocus')
            
            fig, axes = plt.subplots(rows, cols, subplot_kw=dict(polar=True))
            
            for n,ax in enumerate(axes):
                
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.set_ylim([0,np.pi/2])

                if plt_type != 'contour':
                    plt.close()
                    raise NotImplementedError('choose contour')
                elif contourlevels is None:
                    pt = ax.contourf(self.beta+np.deg2rad(90),self.alpha,self.data[n],cmap=cmap)
                elif contourlevels is not None:
                    pt = ax.contourf(self.beta+np.deg2rad(90),self.alpha,self.data[n],levels=contourlevels,cmap=cmap)
                    
                fig.colorbar(pt, ax=ax, fraction=0.046, pad=0.04)

        else: 

            """ working for nd_poleFig """

            cols = pfs
            rows = 1 #change this later??

            if self.subtype != 'nd_poleFig':

                raise NotImplementedError('plotting not supported for bkgd/defocus')
            
            fig, axes = plt.subplots(rows, cols, subplot_kw=dict(polar=True))
            
            for n,ax in enumerate(axes):
                
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.set_ylim([0,np.pi/2])

                if plt_type != 'contour':
                    plt.close()
                    raise NotImplementedError('choose contour')
                elif contourlevels is None:
                    pt = ax.contourf(self.beta+np.deg2rad(90),self.alpha,self.data[n],cmap=cmap)
                elif contourlevels is not None:
                    pt = ax.contourf(self.beta+np.deg2rad(90),self.alpha,self.data[n],levels=contourlevels,cmap=cmap)
                    
                fig.colorbar(pt, ax=ax, fraction=0.046, pad=0.04)            
            
    def grid( self, full=False, ret_ab=False, ret_steps=True ):
        
        """
        Returns ndarray of grid points on pole figure
        
        Inputs:
            full: returns full PF grid
            ret_ab: returns alpha, beta mgrids
        """
        
        k = 0
        
        if self.subtype != 'nd_poleFig': pass
        else:

            self.res = np.deg2rad(5)

            #azimuth
            self._betasteps = np.rad2deg(np.arange(np.deg2rad(2.5),np.deg2rad(357.5)+self.res,self.res))

            full=True

        if full is not False: alpha_max = 19
        else: alpha_max = self.alpha.shape[0]
        
        pf_grid = np.zeros((alpha_max, len(self._betasteps)))
        
        #polar
        alphasteps = np.arange(0,alpha_max*self.res,self.res)

        #grids
        alp, bet = np.meshgrid(alphasteps,self._betasteps,indexing='ij')
        
        for a in range(alpha_max):
            
            for b in range(len(self._betasteps)):
        
                pf_grid[a,b] = k
                k += 1
    
        if ret_ab is True: return pf_grid, alp, np.deg2rad(bet)
        elif ret_steps is True: return pf_grid, alphasteps, np.deg2rad(self._betasteps)
        else: return pf_grid 

    def correct( self, bkgd=None, defocus=None ):

        """
        correct for background and defocussing effects from lab diffractometer

        modifies data with corrected value
        stores:
            data: return print(corrected data
            raw_data: original data
            bkdg: background 
        """

        self.raw_data = copy.deepcopy(self.data)
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
                    
                    self.data[i] = np.where(self.data[i] < 0, 5, self.data[i])

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

            if d.shape[0] < 19: 

                print('warning: only preliminary normalization for incomplete pole figs')

                temp = np.sum( np.ravel(d) * np.ravel(self.cellVolume) ) / np.sum( np.ravel(self.cellVolume) )

                # if temp != 1: 

                pg_dg = d * self.cellVolume

                norm = np.sum( np.ravel( pg_dg ) ) / np.sum( np.ravel( self.cellVolume ) )

                self.data[i] = ( 1 / norm ) * d   
            
            else:

                temp = np.sum( np.ravel(d) * np.ravel(self.cellVolume) ) / np.sum( np.ravel(self.cellVolume) ) 

                if temp != 1: 

                    pg_dg = d * self.cellVolume

                    norm = np.sum( np.ravel( pg_dg ) ) / ( 2 * np.pi )

                    self.data[i] = ( 1 / norm ) * d

    def interpolate( self, res, grid=None, intMethod='cubic' ):

        """
        interpolate data points to normal grid | (res x res) cell size
        
        only for irregular spaced data
        """        

        if self.subtype != 'nd_poleFig': raise ValueError('only supported for nd currently')

        self.input_data = copy.deepcopy(self.data)
        self.res = res

        if grid is None:

            self.y_pol = {}

            #polar
            self._alphasteps = np.arange(0,np.pi/2+self.res,self.res)
            #azimuth
            self._betasteps = np.arange(np.deg2rad(2.5),2*np.pi+self.res,self.res)
            #grids
            self.alpha, self.beta = np.meshgrid(self._alphasteps,self._betasteps,indexing='ij')

            for i in range(self._numHKL):

                self.y_pol[i] = XYZtoSPH(self.y[i])
                self.data[i] = abs(griddata(self.y_pol[i], self.data[i], (self.beta,self.alpha), method=intMethod, fill_value=0.05))

            self.cellVolume = self.res * ( np.cos( self.alpha - (self.res/2) ) - np.cos( self.alpha + (self.res/2) ) ) 

        else: raise NotImplementedError('not setup yet')

    def rotate( self, g, eps=1E-7 ):

        """
        rotates pfs by a given g matrix

        """

        if g.shape != (3,3): raise ValueError('g not 3x3')
        if abs(1 - np.linalg.det(g)) > eps: raise ValueError('g not proper orthogonal')

        if self.subtype == 'nd_poleFig':

            for i in range(self._numHKL):

                temp = np.zeros_like(self.y[i])

                for hi,h in enumerate(self.y[i]):

                    self.y[i][hi,:] = np.dot(g,h)
                    

        else: raise ValueError('only nd supported..')

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
    
    def __init__( self, poleFigCells, ODcells ):
        
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

    def __init__( self, cellSize, crystalSym, sampleSym, weights=None ):
        
        super(bunge, self).__init__( crystalSym, sampleSym )
        
        # set boundary in Bunge space (not rigorous for cubic)
        if sampleSym == '1': self._phi1max = np.deg2rad(360)
        elif sampleSym == 'm': self._phi1max = np.deg2rad(180)
        elif sampleSym == 'mmm': self._phi1max = np.deg2rad(90)
        else: raise ValueError('invalid sampleSym')

        if crystalSym == 'm-3m': 
            self._Phimax = np.deg2rad(90)
            self._phi2max = np.deg2rad(90)
        elif crystalSym == 'm-3': raise NotImplementedError('coming soon..')
        else: raise ValueError('invalid crystalSym, only cubic so far..')

        # setup grid
        self._phi1range = np.arange(0,self._phi1max+cellSize,cellSize)
        self._Phirange = np.arange(0,self._Phimax+cellSize,cellSize)
        self._phi2range = np.arange(0,self._phi2max+cellSize,cellSize)

        # centroid grid
        self._phi1cen_range = np.arange( (cellSize/2),( self._phi1max-(cellSize/2) )+cellSize,cellSize )
        self._Phicen_range = np.arange( (cellSize/2),( self._Phimax-(cellSize/2) )+cellSize,cellSize )
        self._phi2cen_range = np.arange( (cellSize/2),( self._phi2max-(cellSize/2) )+cellSize,cellSize )

        self.phi2, self.Phi, self.phi1 = np.meshgrid(self._phi2range, self._Phirange, self._phi1range, indexing = 'ij')
        self.phi2cen, self.Phicen, self.phi1cen = np.meshgrid(self._phi2cen_range, self._Phicen_range, self._phi1cen_range, indexing = 'ij')

        self.g, self.bungeList = eu2om((self.phi1cen,self.Phicen,self.phi2cen),out='mdarray')
        
        self.res = cellSize
        
        ## vol integral of sin(Phi) dPhi dphi1 dphi2 
        self.volume = (-cos(self._Phimax) + cos(0)) * self._phi1max * self._phi2max
        self.cellVolume = self.res * self.res * ( np.cos( self.Phicen - (self.res/2) ) - np.cos( self.Phicen + (self.res/2) ) )

        if weights is None: self.weights = np.zeros_like(self.bungeList)
        else: self.weights = weights

    def normalize( self ):

        temp = np.sum( np.ravel(self.weights) * np.ravel(self.cellVolume) ) / np.sum( np.ravel(self.cellVolume) )

        if temp != 1:

            od_dg = self.weights * np.ravel(self.cellVolume)

            norm = np.sum ( np.ravel(od_dg) ) / np.sum( np.ravel( self.cellVolume ) )

            self.weights = ( 1 / norm ) * self.weights

class rodrigues( OD ):
    
    """
    Extends OD to rodrigues space
    """
        
    def __init__( self, gridPtSize, crystalSym, sampleSym ):
                 
        super(rodrigues, self).__init__( crystalSym, sampleSym )
        

        
# %%
        
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