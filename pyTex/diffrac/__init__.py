#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:30:23 2019

@author: nate
"""

"""
diffrac module

a = 4.046 #Angstrom
latt = Lattice.cubic(a)
structure = Structure(latt, ["Al", "Al", "Al", "Al"], [[0, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]])
rad = 'CuKa'

"""

import numpy as _np
import pandas as _pd
import os as _os

# for XRD
from pymatgen.analysis.diffraction.xrd import XRDCalculator as _xrdcalc

# for ND
from neutronpy import Material

__location__ = _os.path.realpath(_os.path.join(_os.getcwd(), _os.path.dirname(__file__)))

hkls = [(1,1,1),(2,0,0),(2,2,0)]

def calc_XRDreflWeights(pymat_struct, hkls, rad='CuKa'):
    
    """ 
    uses pymatgen for struct factor calc
    https://pymatgen.org
    
    input: 
        - pymatgen struct object
        - radiation: - str used in pymatgen calc
        - hkls: measured with pfs
    """
    
    calc = _xrdcalc(wavelength=rad)
    pat = calc.get_pattern(pymat_struct)
    
    sim_int = {}
    
    """ generate list for all reflections """
    
    for peak,inten in zip(pat.hkls,pat.y):
        
        if len(peak) > 1: #overlapping reflections
            
            total_m = sum([refl['multiplicity'] for refl in peak])
            
            for refl in peak:
            
                sim_int[refl['hkl']] = ( (refl['multiplicity']/total_m) * inten ) / 100
                
        else:
            
            sim_int[peak[0]['hkl']] = inten / 100
        
    
    """ check based off given hkl list """

    refl_wgt = {}

    for hi,hkl in enumerate(hkls):
        
        if hkl in sim_int: refl_wgt[hi] = sim_int[hkl]       
        else: raise ValueError('hkl not in simulation.. did you use correct radiation?')
        
    return refl_wgt  

def calc_NDreflWeights(npy_materialDef, hkls):
    
    if isinstance(npy_materialDef,dict): 

        mat = Material(npy_materialDef)
        
        str_fac = {}
        
        for fi,fam in enumerate(hkls):
            
            str_fac[fi] = []
            
            for h in fam:
                
                str_fac[fi].append( _np.abs( mat.calc_nuc_str_fac(h) )**2 )
                
            str_fac[fi] = _np.average(str_fac[fi])
            
            
        #normalize
        norm = 1 / _np.max(list(str_fac.values()))
            
        for fi,fam in enumerate(hkls):
            
            str_fac[fi] *= norm
            
        return str_fac
    
    else: raise ValueError('supplied mat def not valid')









