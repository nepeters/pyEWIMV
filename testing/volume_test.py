#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 20:35:39 2020

@author: nate
"""

import numpy as _np

from pyTex import bunge
from pyTex.orientation import eu2om

crystalSym = 'm-3m'
sampleSym = '1'
cellSize = _np.deg2rad(5)

od = bunge(cellSize, crystalSym, sampleSym)

g, bungeList = eu2om((od.phi1,od.Phi,od.phi2),out='mdarray')
posEdge = _np.zeros_like(bungeList).astype(bool)
negEdge = _np.zeros_like(bungeList).astype(bool)

Phi_zero = (od.Phi == 0)
Phi_max = (od.Phi == _np.max(od.Phi))

phi1_zero = (od.phi1 == 0)
phi1_max = (od.phi1 == _np.max(od.phi1))

phi2_zero = (od.phi2 == 0)
phi2_max = (od.phi2 == _np.max(od.phi2))

dphi1_dphi2 = _np.ones_like(bungeList) * 5
#phi1 edge cases - 0.5*Δφ1 + Δφ2
dphi1_dphi2[phi1_zero+phi1_max] = 7
#phi2 edge cases - Δφ1 + 0.5*Δφ2
dphi1_dphi2[phi2_zero+phi2_max] = 7
#phi1 and phi2 edge case - 0.5*Δφ1 + 0.5*Δφ2
dphi1_dphi2[(phi2_zero+phi2_max)*(phi1_zero+phi1_max)] = 3

import mayavi.mlab as mlab

mlab.figure(bgcolor=(1,1,1))

pts = mlab.points3d(od.phi2,od.Phi,od.phi1,dphi1_dphi2,mode='point')
pts.actor.property.render_points_as_spheres = True
pts.actor.property.point_size = 6

mlab.show(stop=True)