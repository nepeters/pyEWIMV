## First attempt at julia function
#
#

using LinearAlgebra

function quat_mul(q1, q2; left=true)
    #quat multiplcation: q1 * q2
    #left = True: q1 is large, left = False: q2 is large

    if left == true
        q3 = zeros(size(q1)[1],4)
    
        q3[:,1] = q1[:,1]*q2[1] - q1[:,2]*q2[2] - q1[:,3]*q2[3] - q1[:,4]*q2[4]
        ##          a1      b2        a2      b1         c1     d2          c2    d1
        q3[:,2] = q1[:,1]*q2[2] + q2[1]*q1[:,2] + q1[:,3]*q2[4] - q2[3]*q1[:,4]
        ##          a1      c2        a2      c1         d1     b2          d2    b1           
        q3[:,3] = q1[:,1]*q2[3] + q2[1]*q1[:,3] + q1[:,4]*q2[2] - q2[4]*q1[:,2]
        ##          a1      d2        a2      d1         b1     c2          b2    c1 
        q3[:,4] = q1[:,1]*q2[4] + q2[1]*q1[:,4] + q1[:,2]*q2[3] - q2[2]*q1[:,3]
    else
        q3 = zeros(size(q2)[1],4)
    
        q3[:,1] = q1[1]*q2[:,1] - q1[2]*q2[:,2] - q1[3]*q2[:,3] - q1[4]*q2[:,4]
        ##          a1      b2        a2      b1         c1     d2          c2    d1
        q3[:,2] = q1[1]*q2[:,2] + q2[:,1]*q1[2] + q1[3]*q2[:,4] - q2[:,3]*q1[4]
        ##          a1      c2        a2      c1         d1     b2          d2    b1           
        q3[:,3] = q1[1]*q2[:,3] + q2[:,1]*q1[3] + q1[4]*q2[:,2] - q2[:,4]*q1[2]
        ##          a1      d2        a2      d1         b1     c2          b2    c1 
        q3[:,4] = q1[1]*q2[:,4] + q2[:,1]*q1[4] + q1[2]*q2[:,3] - q2[:,2]*q1[3]
    end
   
    return q3

end

function quat2eu(quat; P=-1)

    #quaternion to euler (bunge)
    #    - input is NxMx4 mdarray | N - along fibre. M - sym equiv.
    #    - output is 3 array(N) - (φ1,Φ,φ2)
    #
    #convention from https://doi.org/10.1088/0965-0393/23/8/083501
    

    q03 = quat[:,:,1].^2 + quat[:,:,4].^2 
    q12 = quat[:,:,2].^2 + quat[:,:,3].^2 
    chi = sqrt(q03*q12)

    q03 = round.(q03, decimals = 15)
    q12 = round.(q12, decimals = 15)
    chi = round.(chi, decimals = 15)
    
    case1 = (chi == 0) & (q12 == 0)
    case2 = (chi == 0) & (q03 == 0)
    case3 = (chi != 0)
    
    # phi1 = _np.zeros_like(q03)
    # Phi = _np.zeros_like(q03)
    # phi2 = _np.zeros_like(q03)
    
    # q0 = quat[:,:,0]
    # q1 = quat[:,:,1]
    # q2 = quat[:,:,2]
    # q3 = quat[:,:,3]
    
    # # saved_handler = _np.seterrcall(err_handler)
    # save_err = _np.seterr(all='raise')

    # if _np.any(case1): 
        
    #     phi1_t = _np.arctan2( -2*P*q0[case1]*q3[case1], q0[case1]**2 - q3[case1]**2 )
    #     Phi_t = _np.zeros_like(phi1_t)
    #     phi2_t = _np.zeros_like(phi1_t) 
        
    #     phi1[case1] = phi1_t
    #     Phi[case1] = Phi_t
    #     phi2[case1] = phi2_t            
        
    # # if len(case2[0]) != 0: 
    # if _np.any(case2):
        
    #     phi1_t = _np.arctan2( 2*q1[case2]*q2[case2], q1[case2]**2 - q2[case2]**2 )
    #     Phi_t = _np.ones_like(phi1_t)*_np.pi
    #     phi2_t = _np.zeros_like(phi1_t)

    #     phi1[case2] = phi1_t
    #     Phi[case2] = Phi_t
    #     phi2[case2] = phi2_t
        
    # # if len(case3[0]) != 0:
    # if _np.any(case3):
        
    #     phi1_t = _np.arctan2( (q1[case3]*q3[case3] - P*q0[case3]*q2[case3]) / chi[case3],
    #                             (-P*q0[case3]*q1[case3] - q2[case3]*q3[case3]) / chi[case3] )            
    #     Phi_t = _np.arctan2( 2*chi[case3], q03[case3] - q12[case3])            
    #     phi2_t = _np.arctan2( (P*q0[case3]*q2[case3] + q1[case3]*q3[case3]) / chi[case3],
    #                             (q2[case3]*q3[case3] - P*q0[case3]*q1[case3]) / chi[case3] )  
        
    #     phi1[case3] = phi1_t
    #     Phi[case3] = Phi_t
    #     phi2[case3] = phi2_t

    # phi1 = _np.round(phi1, decimals = 8)
    # Phi = _np.round(Phi, decimals = 8)
    # phi2 = _np.round(phi2, decimals = 8)

    # return phi1,Phi,phi2

end

fam = [1 1 1]
y = [4 5 6]
phi = collect(0:deg2rad(5):deg2rad(360))
crysSymOps = [1 0 0 0; 0 1 0 0;0 0 1 0;0 0 0 1;1 0 0 0; 0 1 0 0;0 0 1 0;0 0 0 1;1 0 0 0; 0 1 0 0;0 0 1 0;0 0 0 1;1 0 0 0; 0 1 0 0;0 0 1 0;0 0 0 1;1 0 0 0; 0 1 0 0;0 0 1 0;0 0 0 1;1 0 0 0; 0 1 0 0;0 0 1 0;0 0 0 1;]
# crysSymOps = [1 0 0 0]
smplSymOps = [1 0 0 0]

#initialize
C = zeros(3)
omega = zeros(size(fam)[1])
q0 = zeros(4)
q  = zeros(length(phi),4)
qf = zeros(length(phi),4)

cphi = cos.(phi ./ 2)
sphi = cos.(phi ./ 2)

#normalize
fam /= norm(fam)
y /= norm(y)

C = cross(vec(fam),vec(y))
C /= norm(C)

omega = acos(dot(vec(fam),vec(y)))

q0[1]  = cos(omega/2) 
q0[2:4] = sin(omega/2) .* C

q[:,1] = cphi
q[:,2:4] = repeat(y,length(cphi),1) .* sphi

#create qf
qf[:,:] = quat_mul(q, q0)
# # for k in 1:size(cphi)[1]

# #     qf[k,1] = q[k,1]*q0[1] - q[k,2]*q0[2] - q[k,3]*q0[3] - q[k,4]*q0[4]
# #     qf[k,2] = q[k,1]*q0[2] + q0[1]*q[k,2] + q[k,3]*q0[4] - q0[3]*q[k,4]
# #     qf[k,3] = q[k,1]*q0[3] + q0[1]*q[k,3] + q[k,4]*q0[2] - q0[4]*q[k,2]
# #     qf[k,4] = q[k,1]*q0[4] + q0[1]*q[k,4] + q[k,2]*q0[3] - q0[2]*q[k,3]

# # end

qf_crys = zeros(length(cphi),size(crysSymOps)[1],4)
qf_smpl = zeros(size(qf))

crysSymOps = repeat(crysSymOps,1,1,73)
crysSymOps = permutedims(crysSymOps,[3,1,2])

#perform smplSymOps
for n in 1:size(smplSymOps)[1]

    qf_smpl[:,:] = quat_mul(smplSymOps[n,:],qf,left=false)
    
    #perform crysSymOps
    #can we vectorize this?
    # for p in 1:size(crysSymOps)[1]

    #     qf_crys[p,:,:] = quat_mul(qf_smpl,crysSymOps[n,:],left=true)

    # end

    @fastmath qf_crys[:,:,1] = @. qf_smpl[:,1]*crysSymOps[:,:,1] - qf_smpl[:,2] * crysSymOps[:,:,2] - qf_smpl[:,3] * crysSymOps[:,:,3] - qf_smpl[:,4] * crysSymOps[:,:,4]
    #                a1      b2        a2      b1         c1     d2          c2    d1
    @fastmath qf_crys[:,:,2] = @. qf_smpl[:,1]*crysSymOps[:,:,2] + crysSymOps[:,:,1]*qf_smpl[:,2] + qf_smpl[:,3]*crysSymOps[:,:,4] - crysSymOps[:,:,3]*qf_smpl[:,4]
    ##                a1      c2        a2      c1         d1     b2          d2    b1           
    @fastmath qf_crys[:,:,3] = @. qf_smpl[:,1]*crysSymOps[:,:,3] + crysSymOps[:,:,1]*qf_smpl[:,3] + qf_smpl[:,4]*crysSymOps[:,:,2] - crysSymOps[:,:,4]*qf_smpl[:,2]
    ##                a1      d2        a2      d1         b1     c2          b2    c1 
    @fastmath qf_crys[:,:,4] = @. qf_smpl[:,1]*crysSymOps[:,:,4] + crysSymOps[:,:,1]*qf_smpl[:,4] + qf_smpl[:,2]*crysSymOps[:,:,3] - crysSymOps[:,:,2]*qf_smpl[:,3]

    # qf_crys = permutedims(qf_crys,[2,1,3])
    
    @fastmath q03 = @. qf_crys[:,:,1]^2 + qf_crys[:,:,4]^2 
    @fastmath q12 = @. qf_crys[:,:,2]^2 + qf_crys[:,:,3]^2 
    @fastmath chi = sqrt.(q03.*q12)

    # q03 = round.(q03, digits = 15)
    # q12 = round.(q12, digits = 15)
    # chi = round.(chi, digits = 15)
    
    # global case1 = (chi .== 0) .& (q12 .== 0)
    # global case2 = (chi .== 0) .& (q03 .== 0)
    # global case3 = (chi .!= 0)

end


# q0[fi][yi] = _np.hstack( [ _np.cos(omega[fi][yi]/2), _np.sin(omega[fi][yi]/2) * axis[fi][yi] ] )
# q[fi][yi]  = _np.hstack( [ cphi[:, _np.newaxis], _np.tile( y, (len(cphi),1) ) * sphi[:, _np.newaxis] ] )
# qf[yi] = _quat.multiply(q[fi][yi], q0[fi][yi])