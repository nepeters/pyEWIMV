#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 21:35:33 2020

@author: nate
"""

"""
MAUD ODF import 
"""

from pyTex import bunge
import numpy as np

crystalSym = 'm-3m'
sampleSym = '1'
cellSize = np.deg2rad(5)

od_file = '/mnt/c/Users/Nate/Dropbox/ORNL/EWIMVvsMTEX/MAUD EWIMV exports/NOMAD_Al_4Datasets_NoSSabs_odf_5res'

maud_od = bunge.loadMAUD(od_file,
                         np.deg2rad(5),
                         'm-3m',
                         '1')

cV = maud_od.res * maud_od.res * ( np.cos( maud_od.Phi - ( maud_od.res/2 ) ) - np.cos( maud_od.Phi + ( maud_od.res/2 ) ) )

print(maud_od.index(cellVolume=cV))
print(maud_od.entropy(cellVolume=cV))
# maud_od.plot3d()

tube_rad = np.deg2rad(8)
tube_exp = 1
hkls = [(1,1,1),(2,0,0),(2,2,0)]


recalc_pf = maud_od._calcPF( hkls, tube_rad, tube_exp, tube_proj=True )

cl = np.arange(0,7.5,0.5)
recalc_pf.plot(contourlevels=cl,proj='earea',cmap='magma')
recalc_pf._interpolate(np.deg2rad(5))
recalc_pf.export('/mnt/c/Users/Nate/Desktop',sampleName='maud_recalc')

# %%

from pyTex import poleFigure

fullPFgrid, alp, bet = poleFigure.grid(res=np.deg2rad(5),
                                       radians=True,
                                       cen=False,
                                       ret_ab=True)

#calculate pole figure y's
sph = np.array((np.ravel(alp),np.ravel(bet))).T
#offset (001) direction to prevent errors during path calculation
sph[:,0] = np.where(sph[:,0] == 0, np.deg2rad(0.25), sph[:,0])

#convert to xyz
xyz_pf = np.zeros((sph.shape[0],3))
xyz_pf[:,0] = np.sin( sph[:,0] ) * np.cos( sph[:,1] )
xyz_pf[:,1] = np.sin( sph[:,0] ) * np.sin( sph[:,1] )
xyz_pf[:,2] = np.cos( sph[:,0] )
