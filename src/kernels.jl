# Relative tolerance used to declare a damped block numerically singular: 1e-12 for Float64, scaled
# with the precision of T.
@inline singular_tol(::Type{T}) where T = 4504 * eps(T)

"""
    inv_sym_block(H::SMatrix{P, P, T}, lambda::T, scale::SVector{P, T}) where {P, T}

Invert a symmetric positive semi-definite block `H` with diagonal Levenberg-Marquardt damping
`H + lambda * diag(scale)` (`scale` is the damping diagonal itself, e.g. `|diag(H)|` for Marquardt scaling) in registers.
Returns `(invH, success)` where `success` is false if the block is numerically rank-deficient.
"""
@inline function inv_sym_block(H::SMatrix{1, 1, T, 1}, lambda::T, scale::SVector{1, T}) where T
    h11 = H[1, 1] + lambda * scale[1]
    if h11 <= eps(T) * abs(H[1, 1]) || !isfinite(h11)
        return (invH = zeros(SMatrix{1, 1, T, 1}), success = false)
    end
    return (invH = SMatrix{1, 1, T, 1}((inv(h11),)), success = true)
end

@inline function inv_sym_block(H::SMatrix{2, 2, T, 4}, lambda::T, scale::SVector{2, T}) where T
    H11 = H[1, 1] + lambda * scale[1]
    H22 = H[2, 2] + lambda * scale[2]
    H12 = H[1, 2]

    det = H11 * H22 - H12 * H12
    if det <= singular_tol(T) * (H11 * H22) || !isfinite(det)
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
    diag_prod = H11 * H22 * H33

    if det <= singular_tol(T) * diag_prod || !isfinite(det)
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
    return (invH = SMatrix{D, D, T, D * D}(chol \ one(SMatrix{D, D, T})), success = true)
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

# dot(a[ia:ia+de-1], b[ib:ib+de-1]) with a static or run-time dimension
@inline function dot_dim(a::Vector{T}, ia::Int, b::Vector{T}, ib::Int, s_de::Union{StaticInt, Int}) where T
    acc = zero(T)
    @inbounds for i in 0:dimval(s_de) - 1
        acc += a[ia + i] * b[ib + i]
    end
    return acc
end

# Call `f(s_de)` with the block dimension `de` as a StaticInt when `de <= D_STATIC`, else as an Int
@inline function dispatch_dim(f::F, de::Int) where F
    de == 1 && return f(static(1))
    de == 2 && return f(static(2))
    de == 3 && return f(static(3))
    de == 4 && return f(static(4))
    de == 5 && return f(static(5))
    de == 6 && return f(static(6))
    de == 7 && return f(static(7))
    de == 8 && return f(static(8))
    return f(de)
end
