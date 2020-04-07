## First attempt at julia function
#
#

using LinearAlgebra

function mul(q1, q2)
    #quat multiplication q1*q2, q1 being 2D, q2 being 1D

    q3::Array(Float64,length(q1),4)

    q11 = view(q1,:,1)
    q12 = view(q1,:,2)
    q13 = view(q1,:,3)
    q14 = view(q1,:,4)

    @fastmath q3[:, 1] =
        q11 * q2[1] - q12 * q2[2] - q13 * q2[3] -
        q14 * q2[4]
    ##          a1      b2        a2      b1         c1     d2          c2    d1
    @fastmath q3[:, 2] =
        q11 * q2[2] + q2[1] * q12 + q13 * q2[4] -
        q2[3] * q14
    ##          a1      c2        a2      c1         d1     b2          d2    b1
    @fastmath q3[:, 3] =
        q11 * q2[3] + q2[1] * q13 + q14 * q2[2] -
        q2[4] * q12
    ##          a1      d2        a2      d1         b1     c2          b2    c1
    @fastmath q3[:, 4] =
        q11 * q2[4] + q2[1] * q14 + q12 * q2[3] -
        q2[2] * q13

        return q3

end

const fam = [1 1 1]
const y = [4 5 6]
const phi = collect(0:deg2rad(5):deg2rad(360))
const crysSymOps = [
    1 0 0 0
    0 1 0 0
    0 0 1 0
    0 0 0 1
    1 0 0 0
    0 1 0 0
    0 0 1 0
    0 0 0 1
    1 0 0 0
    0 1 0 0
    0 0 1 0
    0 0 0 1
    1 0 0 0
    0 1 0 0
    0 0 1 0
    0 0 0 1
    1 0 0 0
    0 1 0 0
    0 0 1 0
    0 0 0 1
    1 0 0 0
    0 1 0 0
    0 0 1 0
    0 0 0 1
]

# crysSymOps = [1 0 0 0]
const smplSymOps = [1 0 0 0]

#initialize
C = zeros(3)
omega = zeros(size(fam)[1])
q0 = zeros(4)
q = zeros(length(phi), 4)
qf = zeros(length(phi), 4)

cphi = cos.(phi ./ 2)
sphi = cos.(phi ./ 2)

#normalize
fam /= norm(fam)
y /= norm(y)

C = cross(vec(fam), vec(y))
C /= norm(C)

omega = acos(dot(vec(fam), vec(y)))

q0[1] = cos(omega / 2)
q0[2:4] = sin(omega / 2) .* C

q[:, 1] = cphi
q[:, 2:4] = repeat(y, length(cphi), 1) .* sphi

#create qf
qf[:, :] = quat_mul(q, q0)
# # for k in 1:size(cphi)[1]

# #     qf[k,1] = q[k,1]*q0[1] - q[k,2]*q0[2] - q[k,3]*q0[3] - q[k,4]*q0[4]
# #     qf[k,2] = q[k,1]*q0[2] + q0[1]*q[k,2] + q[k,3]*q0[4] - q0[3]*q[k,4]
# #     qf[k,3] = q[k,1]*q0[3] + q0[1]*q[k,3] + q[k,4]*q0[2] - q0[4]*q[k,2]
# #     qf[k,4] = q[k,1]*q0[4] + q0[1]*q[k,4] + q[k,2]*q0[3] - q0[2]*q[k,3]

# # end

qf_crys = zeros(length(cphi), size(crysSymOps)[1], 4)
qf_smpl = zeros(size(qf))

crysSymOps = repeat(crysSymOps, 1, 1, 73)
crysSymOps = permutedims(crysSymOps, [3, 1, 2])

#perform smplSymOps
for n = 1:size(smplSymOps)[1]

    qf_smpl[:, :] = quat_mul(smplSymOps[n, :], qf)

    #perform crysSymOps
    #can we vectorize this?
    # for p in 1:size(crysSymOps)[1]

    #     qf_crys[p,:,:] = quat_mul(qf_smpl,crysSymOps[n,:],left=true)

    # end

    @fastmath qf_crys[:, :, 1] = @. qf_smpl[:, 1] * crysSymOps[:, :, 1] -
       qf_smpl[:, 2] * crysSymOps[:, :, 2] -
       qf_smpl[:, 3] * crysSymOps[:, :, 3] -
       qf_smpl[:, 4] * crysSymOps[:, :, 4]
    #                a1      b2        a2      b1         c1     d2          c2    d1
    @fastmath qf_crys[:, :, 2] = @. qf_smpl[:, 1] * crysSymOps[:, :, 2] +
       crysSymOps[:, :, 1] * qf_smpl[:, 2] +
       qf_smpl[:, 3] * crysSymOps[:, :, 4] -
       crysSymOps[:, :, 3] * qf_smpl[:, 4]
    ##                a1      c2        a2      c1         d1     b2          d2    b1
    @fastmath qf_crys[:, :, 3] = @. qf_smpl[:, 1] * crysSymOps[:, :, 3] +
       crysSymOps[:, :, 1] * qf_smpl[:, 3] +
       qf_smpl[:, 4] * crysSymOps[:, :, 2] -
       crysSymOps[:, :, 4] * qf_smpl[:, 2]
    ##                a1      d2        a2      d1         b1     c2          b2    c1
    @fastmath qf_crys[:, :, 4] = @. qf_smpl[:, 1] * crysSymOps[:, :, 4] +
       crysSymOps[:, :, 1] * qf_smpl[:, 4] +
       qf_smpl[:, 2] * crysSymOps[:, :, 3] -
       crysSymOps[:, :, 2] * qf_smpl[:, 3]

    # qf_crys = permutedims(qf_crys,[2,1,3])

    @fastmath q03 = @. qf_crys[:, :, 1]^2 + qf_crys[:, :, 4]^2
    @fastmath q12 = @. qf_crys[:, :, 2]^2 + qf_crys[:, :, 3]^2
    @fastmath chi = sqrt.(q03 .* q12)

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
