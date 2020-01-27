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

calc_od = bunge.loadMAUD(od_file,
                         np.deg2rad(5),
                         'm-3m',
                         '1')

### 3D ODF plot ###

import mayavi.mlab as mlab
from tvtk.util import ctf
from matplotlib.pyplot import cm

mlab.figure(figure='1',bgcolor=(0.75,0.75,0.75))

#reshape pts
data = calc_od.weights.reshape(calc_od.phi1.shape)
#round small values (<1E-5)
data[data < 1E-5] = 0

#needs work
# vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(data), vmin=0, vmax=0.8)
# vol.volume_mapper_type = 'FixedPointVolumeRayCastMapper'

cont = mlab.pipeline.contour_surface(mlab.pipeline.scalar_field(data),
                                      contours=list(np.linspace(2,np.max(data),10)),
                                      transparent=True)

out = mlab.outline()

ax = mlab.axes(color=(0,0,0),
                xlabel='phi2',
                ylabel='Phi',
                zlabel='phi1',
                ranges=[0, np.rad2deg(calc_od._phi2max),
                        0, np.rad2deg(calc_od._Phimax),
                        0, np.rad2deg(calc_od._phi1max)])  

ax.axes.number_of_labels = 5
ax.axes.corner_offset = 0.04
#font size doesn't work @ v4.7.1
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

cbar = mlab.scalarbar(cont)
cbar.shadow = True
# cbar.use_default_range = False
# cbar.data_range = np.array([ 5, 40.4024208 ])
cbar.number_of_labels = 10
#adjust label position
cbar.label_text_property.justification = 'centered'
cbar.label_text_property.font_family = 'arial'
cbar.scalar_bar.text_pad = 10
cbar.scalar_bar.unconstrained_font_size = True
cbar.label_text_property.italic = False
cbar.label_text_property.font_size = 20
#turn off parallel projection
mlab.gcf().scene.parallel_projection = False

# """ add fibre """
# tubePts = np.rad2deg( bungeAngs[nn_gridPts_full[0][1368].astype(int)] )
# fibrePts = np.rad2deg( fibre_full_e[0][1368] )

# gd3 = mlab.points3d(tubePts[:,2] / 5,
#                     tubePts[:,1] / 5,
#                     tubePts[:,0] / 5,
#                     mode='point',
#                     color=(0,0,0))

# gd3.actor.property.render_points_as_spheres = True
# gd3.actor.property.point_size = 5

# gd4 = mlab.points3d(fibrePts[:,2] / 5,
#                     fibrePts[:,1] / 5,
#                     fibrePts[:,0] / 5,
#                     mode='point',
#                     color=(1,0,0)) 

# gd4.actor.property.render_points_as_spheres = True
# gd4.actor.property.point_size = 9

mlab.show(stop=True)