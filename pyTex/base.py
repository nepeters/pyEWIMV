#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:11:12 2019

@author: nate
"""

"""
texture base
"""

import os,sys,copy
from math import sin,cos

import numpy as np 
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import xrdtools

from .orientation import eu2om
from .utils import symmetrise, normalize, XYZtoSPH

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
                    self.y_pol[i] = XYZtoSPH(np.copy(temp[:,:3]))
                    
            self.subtype = 'nd_poleFig'
            self.symHKL = symmetrise(cs,hkls)
            self.symHKL = normalize(self.symHKL)

        elif pf_type == 'mtex': 

            self.data = {}
            self.y = {}
            self.y_pol = {}
            self.hkl = hkls
            self._numHKL = len(hkls)

            for i,(f,h) in enumerate(zip(files,hkls)):

                if f.endswith('.txt') and os.path.exists(f):

                    """ output from pyReducePF """

                    temp = np.genfromtxt(f)

                    self.data[i] = temp[:,2]
                    self.y_pol[i] = temp[:,:2]

                    xyz_pf = np.zeros((self.y_pol[i].shape[0],3))
                    xyz_pf[:,0] = np.sin( self.y_pol[i][:,1] ) * np.cos( self.y_pol[i][:,0] )
                    xyz_pf[:,1] = np.sin( self.y_pol[i][:,1] ) * np.sin( self.y_pol[i][:,0] )
                    xyz_pf[:,2] = np.cos( self.y_pol[i][:,1] )

                    self.y[i] = xyz_pf
                    
            self.subtype = 'nd_poleFig'
            self.symHKL = symmetrise(cs,hkls)
            self.symHKL = normalize(self.symHKL)            

        elif pf_type == 'recalc': 
            
            self.data = {}
            self.hkl = {}
            self._numHKL = len(hkls)

            #equal grid spacing
            if resolution is None: self.res = None
            else: self.res = resolution

            #polar
            self._alphasteps = np.arange(0,19*self.res,self.res)
            #azimuth
            self._betasteps = np.arange(0,72*self.res,self.res)
            #grids
            self.alpha, self.beta = np.meshgrid(self._alphasteps,self._betasteps,indexing='ij')

            #convert to rad
            self.alpha = np.deg2rad(self.alpha)
            #stereo proj
            self.alpha_stereo = (np.pi/2)*np.tan(copy.deepcopy(self.alpha)/2)
            self.beta = np.deg2rad(self.beta)
            self.res = np.deg2rad(self.res)

            inter_x = copy.deepcopy(self.alpha_stereo*np.cos(self.beta))
            inter_y = copy.deepcopy(self.alpha_stereo*np.sin(self.beta))
            
            #calculated with equispaced grid
            #must interpolate
            if isinstance(files,dict):

                intMethod='linear'

                if arb_y is None: raise ValueError('Please provide arbitrary pole figure vectors (y) for interpolation')
                else: arb_y_pol = XYZtoSPH(arb_y)

                for i,h in enumerate(hkls):

                    x_pf = copy.deepcopy(arb_y_pol[:,1]*np.cos(arb_y_pol[:,0]))
                    y_pf = copy.deepcopy(arb_y_pol[:,1]*np.sin(arb_y_pol[:,0]))

                    self.data[i] = abs(griddata(np.array((x_pf,y_pf)).T, files[i], (inter_x,inter_y), method=intMethod, fill_value=0.05))                
                    self.hkl[i] = h
                    
            else:

                for i,h in enumerate(hkls):
                                    
                    #reuse files variable... may not be clear
                    self.data[i] = files[:,:,i]
                    self.hkl[i] = h
                    
            self.subtype = 'recalc'
                
            try:
                self.cellVolume = self.res * ( np.cos( self.alpha - (self.res/2) ) - np.cos( self.alpha + (self.res/2) ) )        
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

                if f.endswith('.jul') and os.path.exists(f):

                    temp = np.genfromtxt(f,skip_header=2)

                    self.data[i] = temp[:,2]

                    temp[:,1] = np.where(temp[:,1] < 0, temp[:,1]+360, temp[:,1])
                    self.y_pol[i] = np.deg2rad(temp[:,:2]) #alpha, beta

                    xyz_pf = np.zeros((self.y_pol[i].shape[0],3))
                    xyz_pf[:,0] = np.sin( self.y_pol[i][:,0] ) * np.cos( self.y_pol[i][:,1] )
                    xyz_pf[:,1] = np.sin( self.y_pol[i][:,0] ) * np.sin( self.y_pol[i][:,1] )
                    xyz_pf[:,2] = np.cos( self.y_pol[i][:,0] )

                    self.y[i] = xyz_pf
                
                elif os.path.exists(f) is False: raise ValueError('file path not found')                    
                else: raise ValueError('not .jul file format')

            self.subtype = 'nd_poleFig'
            self.symHKL = symmetrise(cs,hkls)
            self.symHKL = normalize(self.symHKL)            

        else: raise NotImplementedError('pf type not recognized')
        
    def plot( self, plt_type='contour', proj='stereo', cmap='magma_r', contourlevels=None, pfs='all' ):
        
        """
        plotting utility
        """

        if pfs == 'all':

            cols = self._numHKL
            rows = 1 #change this later??

            if self.subtype != 'poleFig':

                if self.subtype != 'recalc':

                    raise NotImplementedError('plotting not supported for bkgd/defocus')

            fig, axes = plt.subplots(rows, cols, subplot_kw=dict(polar=True))

            if proj == 'stereo': plt_alpha = (np.pi/2)*np.tan(copy.deepcopy(self.alpha)/2)
            elif proj == 'earea': plt_alpha = 2*np.sin(copy.deepcopy(self.alpha)/2)
            elif proj == 'none': plt_alpha = copy.deepcopy(self.alpha)

            for n,ax in enumerate(axes):
                
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.set_ylim([0,np.max(plt_alpha)])

                if plt_type != 'contour':
                    plt.close()
                    raise NotImplementedError('choose contour')
                elif contourlevels is None:
                    pt = ax.contourf(self.beta,plt_alpha,self.data[n],cmap=cmap)
                    fig.colorbar(pt, ax=ax, fraction=0.046, pad=0.04)
                elif contourlevels is not None:
                    pt = ax.contourf(self.beta,plt_alpha,self.data[n],levels=contourlevels,cmap=cmap)
                    if n == (len(axes)-1): fig.colorbar(pt, ax=ax, fraction=0.046, pad=0.04)
                
        else: 

            """ working for nd_poleFig """

            cols = pfs
            rows = 1 #change this later??

            if self.subtype != 'nd_poleFig':

                if self.subtype != 'recalc':

                    raise NotImplementedError('plotting not supported for bkgd/defocus')

                pass
            
            fig, axes = plt.subplots(rows, cols, subplot_kw=dict(polar=True))

            if proj == 'stereo': plt_alpha = (np.pi/2)*np.tan(copy.deepcopy(self.alpha)/2)
            elif proj == 'earea': plt_alpha = 2*np.sin(copy.deepcopy(self.alpha)/2)
            elif proj == 'none': plt_alpha = copy.deepcopy(self.alpha)

            for n,ax in enumerate(axes):
                
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.set_ylim([0,np.max(plt_alpha)])

                if plt_type != 'contour':
                    plt.close()
                    raise NotImplementedError('choose contour')
                elif contourlevels is None:
                    pt = ax.contourf(self.beta,plt_alpha,self.data[n],cmap=cmap)
                    fig.colorbar(pt, ax=ax, fraction=0.046, pad=0.04)
                elif contourlevels is not None:
                    pt = ax.contourf(self.beta,plt_alpha,self.data[n],levels=contourlevels,cmap=cmap)
                    if n == (len(axes)-1): fig.colorbar(pt, ax=ax, fraction=0.046, pad=0.04)
                                    
    def grid( self, full=False, ret_ab=False, ret_steps=True ):
        
        """
        Returns ndarray of grid points on pole figure
        
        Inputs:
            full: returns full NxN PF grid - N = res
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

    def _interpolate( self, res, grid=None, intMethod='cubic' ):

        """
        interpolate data points to normal grid | (res x res) cell size
        only for irregular spaced data

        not working
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

                for yi,y in enumerate(self.y[i]):

                    self.y[i][yi,:] = np.dot(g,y)
                    

        else: raise ValueError('only nd supported..')

    def export( self, location, sampleName=None ):

        """
        export pfs for MTEX
        uses .jul format
        """

        if not os.path.exists(location): os.mkdir(location)

        for hi,h in self.hkl.items():
            
            hstr = ''.join([str(i) for i in h])

            if sampleName: fname = sampleName+'_pf_'+hstr+'.jul'
            else: fname = 'pf_'+hstr+'.jul'

            fname = os.path.join(location,fname)

            print(fname)
            writeArr = np.column_stack((np.rad2deg(np.ravel(self.alpha)),
                                        np.rad2deg(np.ravel(self.beta)),
                                        np.ravel(self.data[hi])))
        
            with open(fname,'w') as f:
                f.write('pyTex output\n')
                f.write(' alpha\tbeta\tintensity')
                np.savetxt(f,
                           writeArr,
                           fmt=('%.6f','%.6f','%.6f'),
                           delimiter='\t',
                           newline='\n')

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

        temp = np.sum( self.weights * np.ravel(self.cellVolume) ) / np.sum( np.ravel(self.cellVolume) )

        if temp != 1:

            od_dg = self.weights * np.ravel(self.cellVolume)

            norm = np.sum ( np.ravel(od_dg) ) / np.sum( np.ravel( self.cellVolume ) )

            self.weights = ( 1 / norm ) * self.weights
            
    def sectionPlot( self, sectionAxis, sectionVal, cmap='magma_r' ):
        
        cols = 1
        rows = 2 #change this later??
        fig, axes = plt.subplots(rows, cols)
            
        for n,ax in enumerate(axes):   
            
            if sectionAxis == 'phi1': pass
            elif sectionAxis == 'phi': pass
            elif sectionAxis == 'phi2':
                
                if sectionVal in self._phi2range: section_id = np.where(self._phi2range == sectionVal)[0] - 1
                else: raise ValueError('section must be in discrete grid')
        
                pltX, pltY = np.rad2deg(self.phi1cen[section_id,:,:]), np.rad2deg(self.Phicen[section_id,:,:])
                pltX = pltX.reshape(pltX.shape[1:])
                pltY = pltY.reshape(pltY.shape[1:])
                
                pltWgt = np.copy(self.weights)
                pltWgt = pltWgt.reshape(self.Phicen.shape)
                pltWgt = pltWgt[section_id,:,:]
                pltWgt = pltWgt.reshape(pltWgt.shape[1:])
            
            pt = ax.contourf(pltX,pltY,pltWgt,cmap=cmap)
            fig.colorbar(pt, ax=ax, fraction=0.046, pad=0.04) 

    def export( self, fname ):
        
        out = np.array((self.phi1cen.flatten(),self.Phicen.flatten(),self.phi2cen.flatten(),self.weights)).T
        
        with open(fname, 'a') as file:
            file.write('#phi1\tPhi\tphi2\tweight\n')
            np.savetxt(file,
                        out,
                        fmt=('%.5f','%.5f','%.5f','%.5f'),
                        delimiter='\t',
                        newline='\n')

    def index( self, print=False ):

        """
        calculate texture index

        """

        return np.mean(self.weights**2)

    def _plotPF( self, hkl ):
        
        pass
        
        # """ recalculate full pole figures """
        # wgts = {}
        
        # for fi in range(numPoles):
            
        #     recalc_pf_full[i][fi] = np.zeros(len(xyz_pf))
    
        #     for yi in range(len(xyz_pf)):
                
        #         if yi in pf_od_full[fi]: #pf_cell is defined
                    
        #             od_cells = np.array(pf_od_full[fi][yi]['cell'])
    
        #             recalc_pf_full[i][fi][yi] = ( 1 / np.sum(pf_od_full[fi][yi]['weight']) ) * np.sum( pf_od_full[fi][yi]['weight'] * calc_od[i].weights[od_cells.astype(int)] )
            
        # recalc_pf_full[i] = poleFigure(recalc_pf_full[i], pf.hkl, od.cs, 'recalc', resolution=5, arb_y=xyz_pf)
        # recalc_pf_full[i].normalize()  

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