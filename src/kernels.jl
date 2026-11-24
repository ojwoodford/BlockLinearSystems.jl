# Relative tolerance used to declare a damped block numerically singular: 1e-12 for Float64, scaled
# with the precision of T.
@inline singular_tol(::Type{T}) where T = 4504 * eps(T)

"""
    inv_sym_block(H::SMatrix{P, P, T}, lambda::T, scale::SVector{P, T}) where {P, T}

Invert a symmetric positive semi-definite block `H` with diagonal Levenberg-Marquardt damping
`H + lambda * diag(scale)` (`scale` is the damping diagonal itself, e.g. `|diag(H)|` for Marquardt scaling) in registers.
Returns `(invH, success)` where `success` is false if the damped block is not numerically positive definite: every
Cholesky pivot must be positive and, squared, more than `singular_tol(T)` times its damped diagonal entry. The closed
forms for sizes 1-3 apply the same test through the leading principal minors.
"""
@inline function inv_sym_block(H::SMatrix{1, 1, T, 1}, lambda::T, scale::SVector{1, T}) where T
    h11 = H[1, 1] + lambda * scale[1]
    # The Cholesky path's test at size 1: the one pivot is h11 itself, so its ratio to the damped diagonal is 1
    if !(h11 > 0) || !isfinite(h11)
        return (invH = zeros(SMatrix{1, 1, T, 1}), success = false)
    end
    return (invH = SMatrix{1, 1, T, 1}((inv(h11),)), success = true)
end

@inline function inv_sym_block(H::SMatrix{2, 2, T, 4}, lambda::T, scale::SVector{2, T}) where T
    H11 = H[1, 1] + lambda * scale[1]
    H22 = H[2, 2] + lambda * scale[2]
    H12 = H[1, 2]

    # The Cholesky path's test: the pivots are H11 and det / H11, each relative to its damped diagonal entry
    det = H11 * H22 - H12 * H12
    if !(H11 > 0) || !(det > singular_tol(T) * (H11 * H22)) || !isfinite(det)
        return (invH = zeros(SMatrix{2, 2, T, 4}), success = false)
    end
    inv_det = inv(det)
    invH = SMatrix{2, 2, T, 4}((H22 * inv_det, -H12 * inv_det, -H12 * inv_det, H11 * inv_det))
    return (invH = invH, success = true)
end

@inline function inv_sym_block(H::SMatrix{3, 3, T, 9}, lambda::T, scale::SVector{3, T}) where T
    H11 = H[1, 1] + lambda * scale[1]
    H22 = H[2, 2] + lambda * scale[2]
    H33 = H[3, 3] + lambda * scale[3]
    H12 = H[1, 2]; H13 = H[1, 3]; H23 = H[2, 3]

    c11 = H22 * H33 - H23 * H23
    c12 = H13 * H23 - H12 * H33
    c13 = H12 * H23 - H13 * H22
    c22 = H11 * H33 - H13 * H13
    c23 = H12 * H13 - H11 * H23
    c33 = H11 * H22 - H12 * H12

    det = H11 * c11 + H12 * c12 + H13 * c13

    # The Cholesky path's test: the pivots are H11, c33 / H11 and det / c33 (c33 is the leading 2x2 minor),
    # each relative to its damped diagonal entry
    if !(H11 > 0) || !(c33 > singular_tol(T) * (H11 * H22)) || !(det > singular_tol(T) * (c33 * H33)) ||
       !isfinite(det)
        return (invH = zeros(SMatrix{3, 3, T, 9}), success = false)
    end

    inv_det = inv(det)
    invH = SMatrix{3, 3, T, 9}((
        c11 * inv_det, c12 * inv_det, c13 * inv_det,
        c12 * inv_det, c22 * inv_det, c23 * inv_det,
        c13 * inv_det, c23 * inv_det, c33 * inv_det
    ))
    return (invH = invH, success = true)
end

@inline function inv_sym_block(H::SMatrix{D, D, T}, lambda::T, scale::SVector{D, T}) where {D, T}
    Hd = H + lambda * SMatrix{D, D, T}(Diagonal(scale))
    chol = cholesky(Symmetric(Hd, :U); check=false)
    ok = issuccess(chol)
    if ok
        # Every pivot must be safely positive relative to its damped diagonal entry. (A determinant
        # test, as for the closed-form 1-3 dimensional blocks, would underflow for large blocks.)
        U = chol.U
        min_ratio = minimum(i -> abs2(U[i, i]) / Hd[i, i], ntuple(identity, Val(D)))
        ok = min_ratio > singular_tol(T) && isfinite(min_ratio)
    end
    ok || return (invH = zeros(SMatrix{D, D, T, D * D}), success = false)
    # inv(chol) is U⁻¹U⁻ᵀ: one triangular solve and a product, exactly symmetric; `chol \ I` is two solves
    return (invH = SMatrix{D, D, T, D * D}(inv(chol)), success = true)
end

# Block dimensions up to D_STATIC use statically sized (register) arithmetic, dispatched on StaticInt;
# larger ones use a dynamic path with a per-chunk scratch matrix. StaticArrays' Cholesky stops beating
# LAPACK at around this size, and every extra static size costs compile time.
const D_STATIC = 8

# Block dimension as an Int, from either a StaticInt or an Int
@inline dimval(::StaticInt{D}) where D = D
@inline dimval(d::Int) = d

@inline make_svec(arr::AbstractVector{T}, off::Int, ::StaticInt{D}) where {T, D} =
    SVector{D, T}(ntuple(i -> @inbounds(arr[off + i - 1]), Val(D)))

@inline make_smat(arr::AbstractVector{T}, off::Int, ::StaticInt{D}) where {T, D} =
    SMatrix{D, D, T, D * D}(ntuple(i -> @inbounds(arr[off + i - 1]), Val(D * D)))

# Call `f(s_de)` with the block dimension `de` as a StaticInt when `1 <= de <= D_STATIC`, else as an Int.
# sirbsdispatch clamps a value outside its range to the nearest bound, so the range is tested here.
@inline dispatch_dim(f::F, de::Int) where F =
    1 <= de <= D_STATIC ? sirbsdispatch(static(1), static(D_STATIC), de, f) : f(de)
