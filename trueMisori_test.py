

import os

import numpy as _np 
from sklearn.neighbors import KDTree as _KDTree
import rowan as _quat

from tqdm.auto import tqdm as _tqdm

from pyTex import poleFigure, bunge
from pyTex.utils import genSymOps as _genSymOps
from pyTex.orientation import quat2eu as _quat2eu


def fast_mult(qi, qj):
    """
    used to calculate only the scalar portion of quaternion
    
    fast for misorient.
    
    """

    qi = _np.asarray(qi) 
    qj = _np.asarray(qj)
    
    output = qi[..., 0] * qj[..., 0] - _np.sum(qi[..., 1:] * qj[..., 1:], axis=-1)
    
    return output
                     

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

crystalSym = 'm-3m'
sampleSym = '1'
cellSize = _np.deg2rad(5)
od = bunge(cellSize, crystalSym, sampleSym)

hkls = []
files = []

# datadir = '/home/nate/projects/pyReducePF/pole figures/pole figures integ int Al absCorr 2ndFit/combined'
datadir = '/mnt/c/Users/Nate/pyReducePF/pole figures/pole figures integ int Al absCorr 2ndFit/combined'

for file in os.listdir(datadir):
    
    pfName = file.split(')')[0].split('(')[1]
    
    try:
        hkls.append(tuple([int(c) for c in pfName]))
        files.append(os.path.join(datadir,file))
    except: #not hkls
        continue
    
    sortby = [sum([c**2 for c in h]) for h in hkls]
    hkls = [x for _, x in sorted(zip(sortby,hkls), key=lambda pair: pair[0])]
    files = [x for _, x in sorted(zip(sortby,files), key=lambda pair: pair[0])]
    
pf = poleFigure(files,hkls,crystalSym,'sparse')

# rotations around y (integration variable along path)
phi = _np.linspace(0,2*_np.pi,73)

## tube radius
tube_rad = _np.deg2rad(8)

## tell numpy to shut up
_np.seterr(divide='ignore')

## set reflection weights (not used)
refl_wgt = _np.ones((len(pf.hkls))) #all ones

""" use sklearn KDTree for reduction of points for query (euclidean) """

#throw q_grid into positive hemisphere (SO3) for euclidean distance
qgrid_pos = _np.copy(od.q_grid)
qgrid_pos[qgrid_pos[:,0] < 0] *= -1
q_tree = _KDTree(qgrid_pos)

bungeAngs = _np.zeros(( _np.product(od.phi1cen.shape), 3 ))
for ii,i in enumerate(_np.ndindex(od.phi1cen.shape)):    
    bungeAngs[ii,:] = _np.array((od.phi1cen[i],od.Phicen[i],od.phi2cen[i]))

#gnomic rotation angle
rad = _np.sqrt( 2 * ( 1 - _np.cos(0.5*tube_rad) ) )
#euclidean rotation angle
euc_rad = _np.sqrt( 4 * _np.sin(0.25*tube_rad)**2 )

symOps = _genSymOps(crystalSym)
# symOps = _np.unique(_np.swapaxes(symOps,2,0),axis=0)

# proper = _np.where( _np.linalg.det(symOps) == 1 ) #proper orthogonal/no inversion
# quatSymOps = _quat.from_matrix(symOps[proper])
quatSymOps = _quat.from_matrix(symOps)
quatSymOps = _np.tile(quatSymOps[:,:,_np.newaxis],(1,1,len(phi)))
quatSymOps = quatSymOps.transpose((0,2,1))

crys_symOps = _quat.from_matrix(symOps)

tube_rad = _np.arccos( -( ( rad ** 2 ) / 2 ) - 1 ) / 0.5
print(tube_rad)
raise Exception()

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
    
yset = pf.y

od_cell_list = _np.array([i for i in range(qgrid_pos.shape[0])])
od_cell_list = _np.broadcast_to(od_cell_list[:,None],(qgrid_pos.shape[0],len(phi)))

for fi,fam in enumerate(_tqdm(pf._normHKLs[1:2],desc='Calculating paths')):
    
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
        # t0 = _time.time()
        #calculate path for single (hkl)
        axis[fi][yi] = _np.cross(fam,y)
        axis[fi][yi] = axis[fi][yi] / _np.linalg.norm(axis[fi][yi],axis=-1)
        omega[fi][yi] = _np.arccos(_np.dot(fam,y))

        q0[fi][yi] = _np.hstack( [ _np.cos(omega[fi][yi]/2), _np.sin(omega[fi][yi]/2) * axis[fi][yi] ] )
        q[fi][yi]  = _np.hstack( [ cphi[:, _np.newaxis], _np.tile( y, (len(cphi),1) ) * sphi[:, _np.newaxis] ] )
        qf[yi] = _quat.multiply(q[fi][yi], q0[fi][yi])
        
        for smpl_symOp in od.smpl_symOps: 

            #multiply by sym ops, first sample then crystal
            qf_smplSym = _quat.multiply(smpl_symOp, qf[yi])
            qfib = _quat.multiply(qf_smplSym, quatSymOps)
            
            # transpose to correct format for conversion
            qfib = qfib.transpose((1,0,2))
            
            #convert to bunge euler
            phi1, Phi, phi2 = _quat2eu(qfib)
            
            phi1 = _np.where(phi1 < 0, phi1 + 2*_np.pi, phi1) #brnng back to 0 - 2pi
            Phi = _np.where(Phi < 0, Phi + _np.pi, Phi) #brnng back to 0 - pi
            phi2 = _np.where(phi2 < 0, phi2 + 2*_np.pi, phi2) #brnng back to 0 - 2pi
            
            #fundamental zone calc (not true!)
            eu_fib = _np.stack( (phi1, Phi, phi2), axis=2 )
            eu_fib = _np.reshape( eu_fib, (eu_fib.shape[0]*eu_fib.shape[1], eu_fib.shape[2]) ) #new method       
    
            fz = (eu_fib[:,0] <= od._phi1max) & (eu_fib[:,1] <= od._Phimax) & (eu_fib[:,2] <= od._phi2max)
            fz_idx = _np.nonzero(fz)
            
            #pull only unique points? - not sure why there are repeated points, something with symmetry for certain hkls
            #should only be ~73 points per path, but three fold symmetry is also present
            path_e[fi][yi],uni_path_idx = _np.unique(eu_fib[fz],return_index=True,axis=0)
            fib_idx = _np.unravel_index(fz_idx[0], (qfib.shape[0],qfib.shape[1]))            
            path_q[fi][yi] = qfib[fib_idx][uni_path_idx]
    
            ## okay now we have the fiber for each y
            # now need to get the true misorientation as a "distance"
            crys_symOps = _np.tile(crys_symOps[:,:,_np.newaxis,_np.newaxis],(1,1,qgrid_pos.shape[0],len(path_q[fi][yi])))
            crys_symOps = crys_symOps.transpose((0,2,3,1))            
            
            ## create qgrid_pos for fast multiply
            qgrid_pos = _np.tile(qgrid_pos[:,:,_np.newaxis],(1,1,len(path_q[fi][yi])))
            qgrid_pos = qgrid_pos.transpose((0,2,1))
            
            ## tricky multi
            # q_fp_grid = _quat.multiply(qf[yi], qgrid_pos)
            q_fp_grid = _quat.multiply(qgrid_pos, path_q[fi][yi])
            
            ## another multi
            q_mis = fast_mult( q_fp_grid, crys_symOps )
            
            print('done with multi')
            
            ## get angle
            ang_mis = _np.arccos( q_mis ) * 2
            
            ## get min angle (along axis = 0 | across 24 equiv.)
            ang_min  = _np.min(ang_mis,axis=0)
        
            ## get mask (every point within tube)
            mask_tube  = (ang_min <= tube_rad)
            mask_tube2 = (ang_min <= tube_rad)
            
            ## get unique cell points and distances
            pts_in_tube = _np.argwhere(mask_tube)
            dist_pts_in_tube = ang_min[pts_in_tube[:,0],pts_in_tube[:,1]]
            
            ## combine together
            pts_in_tube = _np.hstack((pts_in_tube,dist_pts_in_tube[:,None]))
            
            ## sort by the distance (ascending)
            pts_in_tube = pts_in_tube[_np.argsort(pts_in_tube[:,2]),:]
            
            ## get unique indicies
            uniq_pts_in_tube,uniq_idx = _np.unique(pts_in_tube[:,0],return_index=True)
            
            ## get unique distances
            uniq_pts_dist = pts_in_tube[uniq_idx,2]
        
            newDist = uniq_pts_dist
            newPts  = uniq_pts_in_tube
            
            """ euclidean distance calculation - KDTree """
            
            qfib_pos = _np.copy(qfib[fib_idx])
            qfib_pos[qfib_pos[:,0] < 0] *= -1
            
            # returns tuple - first array are points, second array is distances
            query = q_tree.query_radius(qfib_pos,euc_rad,return_distance=True)
        
            # concatenate arrays
            query = _np.column_stack([_np.concatenate(ar) for ar in query])
            
            # round very small values
            query = _np.round(query, decimals=7)
            
            # move values at zero to very small (1E-5)
            query[:,1] = _np.where(query[:,1] == 0, 1E-5, query[:,1])            
            
            # sort by minimum distance - unique function takes first appearance of index
            query_sort = query[_np.argsort(query[:,1],axis=0)]
            
            # return unique points
            uni_pts = _np.unique(query_sort[:,0],return_index=True)
            
            if yi not in gridPts[fi]:

                gridPts[fi][yi] = [uni_pts[0].astype(int)]
                gridDist[fi][yi] = [query_sort[uni_pts[1],1]]
            
            else:
                
                gridPts[fi][yi].append(uni_pts[0].astype(int))
                gridDist[fi][yi].append(query_sort[uni_pts[1],1])

        gridDist[fi][yi] = _np.concatenate(gridDist[fi][yi])
        gridPts[fi][yi]  = _np.concatenate(gridPts[fi][yi])            
            
        
        break
        
# %%

""" tube plot """

pf_num = 0
yi     = 0

fibers = path_e[pf_num][yi]

import mayavi.mlab as mlab

mlab.figure(bgcolor=(1,1,1))

gd = mlab.points3d(od.phi1,od.Phi,od.phi2,mode='point',scale_factor=1,color=(0.25,0.25,0.25))
gd.actor.property.render_points_as_spheres = True
gd.actor.property.point_size = 2

mlab.axes(color=(0,0,0),ranges=[0,2*_np.pi,0,_np.pi/2,0,_np.pi/2])

od_cells = gridPts[pf_num][yi].astype(int)

od_cells_new = newPts.astype(int)

pts = mlab.points3d(bungeAngs[od_cells,0],bungeAngs[od_cells,1],bungeAngs[od_cells,2],mode='point',scale_factor=1,color=(0,1,0))
pts.actor.property.render_points_as_spheres = True
pts.actor.property.point_size = 5

pts2 = mlab.points3d(bungeAngs[od_cells_new,0],bungeAngs[od_cells_new,1],bungeAngs[od_cells_new,2],mode='point',scale_factor=1,color=(0,1,1))
pts2.actor.property.render_points_as_spheres = True
pts2.actor.property.point_size = 6

# centerTube = mlab.plot3d(fibers[:,0],fibers[:,1],fibers[:,2],tube_sides=36,opacity=0.25,tube_radius=tube_rad,color=(0,0,1))

fiber = mlab.points3d(fibers[:,0],fibers[:,1],fibers[:,2],mode='point',scale_factor=1,color=(1,0,0)) 
fiber.actor.property.render_points_as_spheres = True
fiber.actor.property.point_size = 5
        