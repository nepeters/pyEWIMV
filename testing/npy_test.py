#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 21:42:05 2019

@author: nate
"""

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

# for ND
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.diffraction.neutron import NDCalculator as _ndcalc

from pyTex.utils import symmetrise

__location__ = _os.path.realpath(_os.path.join(_os.getcwd(), _os.path.dirname(__file__)))

hkls = [(1,1,1),(2,0,0),(2,2,0)]
            
a = 4.046 #Angstrom
latt = Lattice.cubic(a)
structure = Structure(latt, ["Al", "Al", "Al", "Al"], [[0, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]])
rad = 'CuKa'

""" 
uses pymatgen for struct factor calc
https://pymatgen.org

input: 
    - pymatgen struct object
    - radiation: - str used in pymatgen calc
    - hkls: measured with pfs
"""

calc = _ndcalc(wavelength=1.0)
pat = calc.get_pattern(structure, two_theta_range=None)

print(pat.hkls)
print(pat.y)

#sim_int = {}
#
#""" generate list for all reflections """
#
#for peak,inten in zip(pat.hkls,pat.y):
#    
#    if len(peak) > 1: #overlapping reflections
#        
#        total_m = sum([refl['multiplicity'] for refl in peak])
#        
#        for refl in peak:
#        
#            sim_int[refl['hkl']] = ( (refl['multiplicity']/total_m) * inten ) / 100
#            
#    else:
#        
#        sim_int[peak[0]['hkl']] = inten / 100
#    
#
#""" check based off given hkl list """
#
#for hkl in hkls:
#    
#    if hkl in sim_int: pass
#    else: raise ValueError('hkl not in simulation.. did you use correct radiation?')









