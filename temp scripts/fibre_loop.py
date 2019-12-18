#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 10:19:24 2019

@author: nate
"""

"""
need to pass:
    OD (bunge box limits)
    rad (fibre limits)
"""

def calcFibre(hfam, yset, omega, h5file=None, h5gname=None):
    
    fibre_e = {}
    fibre_q = {}
    
    nn_gridPts = {}
    nn_gridDist = {}
    
    for yi,y in enumerate(yset):
        
        if h5file is None:
            
            axis = np.cross(hfam,y)
            angle = np.arccos(np.dot(hfam,y))
            
            q0 = quat.from_axis_angle(axis, angle)        
            q1 = [quat.from_axis_angle(h, omega) for h in hfam]
                    
            qfib = np.zeros((len(q1[0]),len(q0),4))
            
            for sym_eq,(qA,qB) in enumerate(zip(q0,q1)):
                
                temp = quat.multiply(qA, qB)
                qfib[:,sym_eq,:] = temp
              
            phi1, Phi, phi2 = quat2eu(qfib)
            
            phi1 = np.where(phi1 < 0, phi1 + 2*np.pi, phi1) #brnng back to 0 - 2pi
            Phi = np.where(Phi < 0, Phi + np.pi, Phi) #brnng back to 0 - pi
            phi2 = np.where(phi2 < 0, phi2 + 2*np.pi, phi2) #brnng back to 0 - 2pi
            
            eu_fib = np.stack( (phi1, Phi, phi2), axis=2 )
            eu_fib = np.reshape( eu_fib, (eu_fib.shape[0]*eu_fib.shape[1], eu_fib.shape[2]) ) #new method       
    
            fz = (eu_fib[:,0] < od._phi1max) & (eu_fib[:,1] < od._Phimax) & (eu_fib[:,2] < od._phi2max)
            fz_idx = np.nonzero(fz)
            
            fibre_e[yi] = eu_fib[fz]
    
            fib_idx = np.unravel_index(fz_idx[0], (qfib.shape[0],qfib.shape[1]))
            
            fibre_q[yi] = qfib[fib_idx]
            
            """ distance calc """
            temp = quatMetricNumba(qgrid,qfib[fib_idx])
        
        else:
            
            with h5.File(h5file,'r') as f:
                grp = f.get(h5gname) #load main group
                dist_grp = grp.get('dist')
                temp = dist_grp.get(str(yi))[()]
                
        """ find tube """
        tube = (temp <= rad)
        temp = np.column_stack((np.argwhere(tube)[:,0],temp[tube]))
        
        """ round very small values """
        temp = np.round(temp, decimals=7)
        
        """ move values at zero to very small (1E-5) """
        temp[:,1] = np.where(temp[:,1] == 0, 1E-5, temp[:,1])
        
        """ sort by min distance """
        temp = temp[np.argsort(temp[:,1],axis=0)]
        """ return unique pts (first in list) """
        uni_pts = np.unique(temp[:,0],return_index=True)
        
        nn_gridPts[yi] = uni_pts[0]
        nn_gridDist[yi] = temp[uni_pts[1],1]
        
    return nn_gridPts, nn_gridDist, fibre_e, fibre_q
        
""" 
multiprocessing using parmap 
TODO better optimized
"""

# import parmap as pm

inputs = []
inputs_full = []
outputs = {}
outputs_full = {}
hkl_loop_str = np.array(hkl_str)[uni_hkls_idx]

for hi, hfam in tqdm(enumerate(pf.symHKL)):
    # inputs.append((hfam,pf.y[hi]))
    outputs[hi] = calcFibre(hfam,pf.y[hi],omega)

for hi, hfam in tqdm(enumerate(symHKL_loop)):
    outputs_full[hi] = calcFibre(hfam,xyz_pf,omega,'fibres.h5',hkl_loop_str[hi]+'_'+str(int(round(np.rad2deg(cellSize)))))
    
# outputs = pm.starmap(calcFibre, inputs, omega, pm_pbar=True, pm_processes=4)
# outputs_full = pm.starmap(calcFibre, inputs_full, omega, pm_pbar=True, pm_processes=4)

nn_gridPts = {}
nn_gridDist = {}

nn_gridPts_full = {}
nn_gridDist_full = {}

fibre_e_full = {}
fibre_q_full = {}

for hi, h in tqdm(enumerate(hkls)):
    
    nn_gridPts[hi] = outputs[hi][0]
    nn_gridDist[hi] = outputs[hi][1]

for i,hi in tqdm(enumerate(hkls_loop_idx)):
    
    nn_gridPts_full[i] = outputs_full[hi][0]
    nn_gridDist_full[i] = outputs_full[hi][1]
    # fibre_e_full[i] = outputs_full[hi][2]
    # fibre_q_full[i] = outputs_full[hi][3]