#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:30:23 2019

@author: nate
"""

"""
inversion module
"""

import os as _os

import numpy as _np

import h5py as _h5
import rowan as _quat
from numba import jit

# #determine if ipython -> import tqdm.notebook
# try:
#     from IPython.Debugger import Tracer
#     debug = Tracer()
#     from tqdm import tqdm as _tqdm
# except ImportError:
    
#     pass # or set "debug" to something else or whatever

try:
    __IPYTHON__
    from tqdm.notebook import tqdm as _tqdm
except NameError:
    from tqdm import tqdm as _tqdm

from pyTex.base import poleFigure as _poleFigure
from pyTex.base import bunge as _bunge
from pyTex.utils import XYZtoSPH as _XYZtoSPH
from pyTex.orientation import quat2eu as _quat2eu

__all__ = ['wimv',
           'e_wimv']

""" internal functions """

def wimv( pfs, orient_dist, iterations=12, ret_pointer=False ):
    
    """
    perform WIMV inversion
    fixed grid in PF space requiredpointer
    
    input:
        exp_pfs    : poleFigure object
        orient_dist: orientDist object
        iterations : number of iterations
    """
    
    fullPFgrid, alp, bet = pfs.grid(pfs.res,
                                 radians=True,
                                 cen=True,
                                 ret_ab=True)

    xyz = {}   
    sph = {}

    a_bins = _np.histogram_bin_edges(_np.arange(0,_np.pi/2+pfs.res,pfs.res),18)
    b_bins = _np.histogram_bin_edges(_np.arange(0,2*_np.pi+pfs.res,pfs.res),72)

    #dict | key:od cell# value:pf cell#s
    od_pf = {}

    #dict | key:pf_cell# value:od cell#s
    pf_od = {}

    numPoles = pfs._numHKL
    numHKLs = [len(fam) for fam in pfs.symHKL]

    """ generate pointer matrix od->pf / pf->od """

    for fi,fam in enumerate(pfs.symHKL):
        
        # Mx3xN array | M - hkl multi. N - # of unique g
        xyz[fi] = _np.dot(fam,orient_dist.g)

        sph[fi] = _XYZtoSPH(xyz[fi],proj='none')
        
        od_pf[fi] = {}
        pf_od[fi] = {}
        
        for od_cell in _np.ravel(orient_dist.bungeList):
            
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

    """ done with pointer generation """

    od_data = _np.ones( orient_dist.bungeList.shape[0]*orient_dist.bungeList.shape[1]*orient_dist.bungeList.shape[2] )
    calc_od = {}
    recalc_pf = {}

    for i in _tqdm(range(iterations),position=0):
        
        """ first iteration, skip recalc of PF """
        
        if i == 0: #first iteration is direct from PFs
            
            od_data = _np.ones( orient_dist.bungeList.shape[0]*orient_dist.bungeList.shape[1]*orient_dist.bungeList.shape[2] )
            calc_od[0] = _np.zeros( (od_data.shape[0], numPoles) )        
            
            for fi in range(numPoles): 
                    
                for pf_cell in _np.ravel(fullPFgrid):
                    
                    if pf_cell in pf_od[fi]:

                        od_cells = _np.array(pf_od[fi][pf_cell])
                        ai, bi = _np.divmod(pf_cell, fullPFgrid.shape[1])
                    
                        if pf_cell < pfs.data[fi].shape[0]*pfs.data[fi].shape[1]: #inside of measured PF range
                            
                            od_data[od_cells.astype(int)] *= pfs.data[fi][int(ai),int(bi)]
                
                """ loop over od_cells (alternative) """
            #    for od_cell in _np.ravel(orient_dist.bungeList):
                   
            #        pf_cells = od_pf[fi][od_cell]
                   
            #        pf_cellMax = pf.data[fi].shape[0]*pf.data[fi].shape[1]
            #        pf_cells = pf_cells[pf_cells < pf_cellMax]
                   
            #        ai, bi = _np.divmod(pf_cells, pf_grid.shape[1])
            #        od_data[int(od_cell)] = _np.product( pf.data[fi][ai.astype(int),bi.astype(int)] )            
                            
                calc_od[0][:,fi] = _np.power(od_data,(1/numHKLs[fi]))
                # calc_od[0][:,fi] = _np.power(od_data,1)
                
            calc_od[0] = _np.product(calc_od[0],axis=1)**(1/numPoles)
            #place into OD object
            calc_od[0] = _bunge(orient_dist.res, orient_dist.cs, orient_dist.ss, weights=calc_od[0])
            calc_od[0].normalize()

        """ recalculate pole figures """
        recalc_pf[i] = _np.zeros((fullPFgrid.shape[0],fullPFgrid.shape[1],numPoles))
        
        for fi in range(numPoles):
            
            for pf_cell in _np.ravel(fullPFgrid):
                
                if pf_cell in pf_od[fi]: #pf_cell is defined
                    
                    od_cells = _np.array(pf_od[fi][pf_cell])
                    ai, bi = _np.divmod(pf_cell, fullPFgrid.shape[1])
                    recalc_pf[i][int(ai),int(bi),fi] = ( 1 / len(od_cells) ) * _np.sum( calc_od[i].weights[od_cells.astype(int)] )
            
        recalc_pf[i] = _poleFigure(recalc_pf[i], pfs.hkl, orient_dist.cs, 'recalc', resolution=5)
        recalc_pf[i].normalize()
        
        """ compare recalculated to experimental """
            
        RP_err = {}
        prnt_str = None
        
        _np.seterr(divide='ignore')

        for fi in range(numPoles):
            
            expLim = pfs.data[fi].shape
            RP_err[fi] = _np.abs( recalc_pf[i].data[fi][:expLim[0],:expLim[1]] - pfs.data[fi] ) / recalc_pf[i].data[fi][:expLim[0],:expLim[1]]
            RP_err[fi][_np.isinf(RP_err[fi])] = 0
            RP_err[fi] = _np.sqrt(_np.mean(RP_err[fi]**2))
            
            if prnt_str is None: prnt_str = 'RP Error: {:.4f}'.format(_np.round(RP_err[fi],decimals=4))
            else: prnt_str += ' | {:.4f}'.format(_np.round(RP_err[fi],decimals=4))
            
        _tqdm.write(prnt_str)
            
        """ (i+1)th inversion """

        od_data = _np.ones( orient_dist.bungeList.shape[0]*orient_dist.bungeList.shape[1]*orient_dist.bungeList.shape[2] )
        calc_od[i+1] = _np.zeros( (od_data.shape[0], numPoles) )        
        
        for fi in range(numPoles):
                
            for pf_cell in _np.ravel(fullPFgrid):
                
                if pf_cell in pf_od[fi]:

                    od_cells = _np.array(pf_od[fi][pf_cell])
                    ai, bi = _np.divmod(pf_cell, fullPFgrid.shape[1])

                    if pf_cell < pfs.data[fi].shape[0]*pfs.data[fi].shape[1]: #inside of measured PF range
                        
                        if recalc_pf[i].data[fi][int(ai),int(bi)] == 0: continue
                        else: od_data[od_cells.astype(int)] *= ( pfs.data[fi][int(ai),int(bi)] / recalc_pf[i].data[fi][int(ai),int(bi)] )
            
            """ loop over od_cells (alternative) """
        #    for od_cell in _tqdm(_np.ravel(orient_dist.bungeList)):
               
        #        pf_cells = od_pf[fi][od_cell]
               
        #        pf_cellMax = pf.data[fi].shape[0]*pf.data[fi].shape[1]
        #        pf_cells = pf_cells[pf_cells < pf_cellMax]
               
        #        ai, bi = _np.divmod(pf_cells, pf_grid.shape[1])
        #        od_data[int(od_cell)] = _np.product( pf.data[fi][ai.astype(int),bi.astype(int)] / recalc_pf[i].data[fi][ai.astype(int), bi.astype(int)] )
                    
            calc_od[i+1][:,fi] = _np.power(od_data,(1/numHKLs[fi]))
        
        calc_od[i+1] = calc_od[i].weights * _np.power(_np.product(calc_od[i+1],axis=1),(0.8/numPoles))
        
        #place into OD object
        calc_od[i+1] = _bunge(orient_dist.res, orient_dist.cs, orient_dist.ss, weights=calc_od[i+1])
        calc_od[i+1].normalize()    


    if ret_pointer is True: return recalc_pf, calc_od, pf_od, od_pf, prnt_str
    else: return recalc_pf, calc_od

def _wimv_test( pfs, orient_dist, iterations=12 ):

    """
    perform WIMV inversion
    fixed grid in PF space requiredpointer
    
    # TODO: remove requirement to pre-generate odf

    input:
        exp_pfs    : poleFigure object
        orient_dist: orientDist object
        iterations : number of iterations
    """
    
    """ calculate pointer """

    orient_dist._calcPointer( 'wimv', pfs )

    """ done with pointer generation """

    od_data = _np.ones( orient_dist.bungeList.shape[0]*orient_dist.bungeList.shape[1]*orient_dist.bungeList.shape[2] )
    calc_od = {}
    recalc_pf = {}

    numPoles = pfs._numHKL
    numHKLs = [len(fam) for fam in pfs.symHKL]

    fullPFgrid = pfs.grid(pfs.res,
                          radians=True,
                          cen=True)

    for i in _tqdm(range(iterations),desc='Performing WIMV iterations',position=0,leave=True):
        
        """ first iteration, skip recalc of PF """
        
        if i == 0: #first iteration is direct from PFs
            
            od_data = _np.ones( orient_dist.bungeList.shape[0]*orient_dist.bungeList.shape[1]*orient_dist.bungeList.shape[2] )
            calc_od[0] = _np.zeros( (od_data.shape[0], numPoles) )        
            
            for fi in range(numPoles): 
                    
                for pf_cell in _np.ravel(fullPFgrid):

                    if pf_cell in orient_dist.pointer['full']['pf to od'][fi]:

                        od_cells = _np.array(orient_dist.pointer['full']['pf to od'][fi][pf_cell])
                        ai, bi = _np.divmod(pf_cell, fullPFgrid.shape[1])
                    
                        if pf_cell < pfs.data[fi].shape[0]*pfs.data[fi].shape[1]: #inside of measured PF range
                            
                            od_data[od_cells.astype(int)] *= pfs.data[fi][int(ai),int(bi)]
                
                """ loop over od_cells (alternative) """
            #    for od_cell in _np.ravel(orient_dist.bungeList):
                   
            #        pf_cells = orient_dist.pointer['full']['od to pf'][fi][od_cell]
                   
            #        pf_cellMax = pf.data[fi].shape[0]*pf.data[fi].shape[1]
            #        pf_cells = pf_cells[pf_cells < pf_cellMax]
                   
            #        ai, bi = _np.divmod(pf_cells, pf_grid.shape[1])
            #        od_data[int(od_cell)] = _np.product( pf.data[fi][ai.astype(int),bi.astype(int)] )            
                            
                calc_od[0][:,fi] = _np.power(od_data,(1/numHKLs[fi]))
                # calc_od[0][:,fi] = _np.power(od_data,1)
                
            calc_od[0] = _np.product(calc_od[0],axis=1)**(1/numPoles)
            #place into OD object
            calc_od[0] = _bunge(orient_dist.res, orient_dist.cs, orient_dist.ss, weights=calc_od[0])
            calc_od[0].normalize()

        """ recalculate pole figures """
        recalc_pf[i] = _np.zeros((fullPFgrid.shape[0],fullPFgrid.shape[1],numPoles))
        
        for fi in range(numPoles):
            
            for pf_cell in _np.ravel(fullPFgrid):
                
                if pf_cell in orient_dist.pointer['full']['pf to od'][fi]: #pf_cell is defined
                    
                    od_cells = _np.array(orient_dist.pointer['full']['pf to od'][fi][pf_cell])
                    ai, bi = _np.divmod(pf_cell, fullPFgrid.shape[1])
                    recalc_pf[i][int(ai),int(bi),fi] = ( 1 / len(od_cells) ) * _np.sum( calc_od[i].weights[od_cells.astype(int)] )
            
        recalc_pf[i] = _poleFigure(recalc_pf[i], pfs.hkl, orient_dist.cs, 'recalc', resolution=5)
        recalc_pf[i].normalize()
        
        """ compare recalculated to experimental """
            
        RP_err = {}
        prnt_str = None
        
        _np.seterr(divide='ignore')

        for fi in range(numPoles):
            
            expLim = pfs.data[fi].shape
            RP_err[fi] = _np.abs( recalc_pf[i].data[fi][:expLim[0],:expLim[1]] - pfs.data[fi] ) / recalc_pf[i].data[fi][:expLim[0],:expLim[1]]
            RP_err[fi][_np.isinf(RP_err[fi])] = 0
            RP_err[fi] = _np.sqrt(_np.mean(RP_err[fi]**2))
            
            if prnt_str is None: prnt_str = 'RP Error: {:.4f}'.format(_np.round(RP_err[fi],decimals=4))
            else: prnt_str += ' | {:.4f}'.format(_np.round(RP_err[fi],decimals=4))
            
        _tqdm.write(prnt_str)
            
        """ (i+1)th inversion """

        od_data = _np.ones( orient_dist.bungeList.shape[0]*orient_dist.bungeList.shape[1]*orient_dist.bungeList.shape[2] )
        calc_od[i+1] = _np.zeros( (od_data.shape[0], numPoles) )        
        
        for fi in range(numPoles):
                
            for pf_cell in _np.ravel(fullPFgrid):
                
                if pf_cell in orient_dist.pointer['full']['pf to od'][fi]:

                    od_cells = _np.array(orient_dist.pointer['full']['pf to od'][fi][pf_cell])
                    ai, bi = _np.divmod(pf_cell, fullPFgrid.shape[1])

                    if pf_cell < pfs.data[fi].shape[0]*pfs.data[fi].shape[1]: #inside of measured PF range
                        
                        if recalc_pf[i].data[fi][int(ai),int(bi)] == 0: continue
                        else: od_data[od_cells.astype(int)] *= ( pfs.data[fi][int(ai),int(bi)] / recalc_pf[i].data[fi][int(ai),int(bi)] )
            
            """ loop over od_cells (alternative) """
        #    for od_cell in _tqdm(_np.ravel(orient_dist.bungeList)):
               
        #        pf_cells = orient_dist.pointer['full']['od to pf'][fi][od_cell]
               
        #        pf_cellMax = pf.data[fi].shape[0]*pf.data[fi].shape[1]
        #        pf_cells = pf_cells[pf_cells < pf_cellMax]
               
        #        ai, bi = _np.divmod(pf_cells, pf_grid.shape[1])
        #        od_data[int(od_cell)] = _np.product( pf.data[fi][ai.astype(int),bi.astype(int)] / recalc_pf[i].data[fi][ai.astype(int), bi.astype(int)] )
                    
            calc_od[i+1][:,fi] = _np.power(od_data,(1/numHKLs[fi]))
        
        calc_od[i+1] = calc_od[i].weights * _np.power(_np.product(calc_od[i+1],axis=1),(0.8/numPoles))
        
        #place into OD object
        calc_od[i+1] = _bunge(orient_dist.res, orient_dist.cs, orient_dist.ss, weights=calc_od[i+1])
        calc_od[i+1].normalize()    

    return recalc_pf, calc_od    

def e_wimv( exp_pfs, orient_dist ):

    """
    perform e-WIMV inversion
    arbitrary PF directions allowed
    minimium entropy solution
    
    input:
        exp_pfs    : poleFigure object
        orient_dist: orientDist object
        
    """

    pass

"""
generate HDF5 file containing fibres/distances for given grid
"""

def genFibreH5(cellSize, hkl_str, uni_hkls_idx, symHKL_loop, xyz_pf, omega, qgrid, od):

    """
    wrapper
    """

    if not _os.path.exists('fibres.h5'): f = _h5.File('fibres.h5','w')
    else: f = _h5.File('fibres.h5','r+')
    f.close()

    hkl_loop_str = _np.array(hkl_str)[uni_hkls_idx]
        
    for hi, hfam in _tqdm(enumerate(symHKL_loop)):
        _calcFibreHDF5(hfam,xyz_pf,omega,qgrid,od,'fibres.h5',hkl_loop_str[hi]+'_'+str(int(round(_np.rad2deg(cellSize)))))

    return



def _calcFibreHDF5(hfam, yset, omega, qgrid, od, h5fname, h5gname):
    
    fibre_e = {}
    fibre_q = {}
    
    nn_gridPts = {}
    nn_gridDist = {}

    f = _h5.File(h5fname,'r+')
    grp = f.create_group(h5gname)
    fib_grp = grp.create_group('fibre')
    dist_grp = grp.create_group('dist')

    for yi,y in enumerate(yset):
            
        axis = _np.cross(hfam,y)
        angle = _np.arccos(_np.dot(hfam,y))
        
        q0 = _quat.from_axis_angle(axis, angle)        
        q1 = [_quat.from_axis_angle(h, omega) for h in hfam]
                
        qfib = _np.zeros((len(q1[0]),len(q0),4))
        
        for sym_eq,(qA,qB) in enumerate(zip(q0,q1)):
            
            temp = _quat.multiply(qA, qB)
            qfib[:,sym_eq,:] = temp
          
        phi1, Phi, phi2 = _quat2eu(qfib)
        
        phi1 = _np.where(phi1 < 0, phi1 + 2*_np.pi, phi1) #brnng back to 0 - 2pi
        Phi = _np.where(Phi < 0, Phi + _np.pi, Phi) #brnng back to 0 - pi
        phi2 = _np.where(phi2 < 0, phi2 + 2*_np.pi, phi2) #brnng back to 0 - 2pi
        
        eu_fib = _np.stack( (phi1, Phi, phi2), axis=2 )
        eu_fib = _np.reshape( eu_fib, (eu_fib.shape[0]*eu_fib.shape[1], eu_fib.shape[2]) ) #new method       

        fz = (eu_fib[:,0] < od._phi1max) & (eu_fib[:,1] < od._Phimax) & (eu_fib[:,2] < od._phi2max)
        fz_idx = _np.nonzero(fz)
        
        fibre_e[yi] = eu_fib[fz]

        fib_idx = _np.unravel_index(fz_idx[0], (qfib.shape[0],qfib.shape[1]))
        
        fibre_q[yi] = qfib[fib_idx]
        
        """ distance calc """
        temp = quatMetricNumba(qgrid,qfib[fib_idx])

        fib_grp.create_dataset(str(yi),data=fibre_q[yi], compression="gzip", compression_opts=9)
        dist_grp.create_dataset(str(yi),data=temp, compression="gzip", compression_opts=9)
        
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
        
        # nn_gridPts[yi] = uni_pts[0]
        # nn_gridDist[yi] = temp[uni_pts[1],1]
        
    # return nn_gridPts, nn_gridDist, fibre_e, fibre_q
    f.close()
    return 