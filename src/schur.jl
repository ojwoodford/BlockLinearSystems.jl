using Static, StaticArrays, SparseArrays, LinearAlgebra

"""
    inv_sym_block(H::SMatrix{P, P, T}, lambda::T, scale::SVector{P, T}) where {P, T}

Invert a symmetric positive semi-definite block `H` with diagonal Levenberg-Marquardt damping
`H + lambda * diag(scale)` (`scale` is the damping diagonal itself, e.g. `|diag(H)|` for Marquardt scaling) in registers.
Returns `(invH, success)` where `success` is false if the block is numerically rank-deficient.
"""
# Relative tolerance used to declare a damped block numerically singular: 1e-12 for Float64, scaled
# with the precision of T.
@inline singular_tol(::Type{T}) where T = 4504 * eps(T)

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

"""
    SchurStructure

Metadata describing bipartite variable partitioning, block offsets, and coupling sparsity.
"""
struct SchurStructure
    nc::Int                          # Total scalar dimension of retained variables
    ncblocks::Int                    # Number of retained blocks
    npblocks::Int                    # Number of eliminable blocks
    c_dims::Vector{UInt8}            # Dimension of each retained block
    p_dims::Vector{UInt8}            # Dimension of each eliminable block
    boffsets_c::Vector{Int}          # Scalar start index of each camera block in x_c
    boffsets_p::Vector{Int}          # Scalar start index of each point block in x_p
    ppoffsets::Vector{Int}           # Start offset in flat C_data for each C_l block
    cpptr::Vector{Int}               # Landmark-major CSR pointer for coupling entries
    cpblock::Vector{Int}             # Retained camera block index for each coupling entry (sorted ascending per landmark!)
    cpoffsets::Vector{Int}           # Start offset in flat E_data for each coupling block
    max_obs_dim::Int                 # Maximum sum_{c in obs(l)} d_c across all landmarks
    p_groups::Vector{UnitRange{Int}} # Groups of landmarks by uniform dimension
    p_sizes::Vector{Int}             # Static landmark dimension for each group
    chunks::Vector{Vector{UnitRange{Int}}} # chunks[group_idx][chunk_id]
end

# Largest supported dimension of an eliminated block
const MAX_ELIMINATED_DIM = 255

function SchurStructure(
    c_dims::AbstractVector{<:Integer},
    p_dims::AbstractVector{<:Integer},
    observations::AbstractVector{<:AbstractVector{<:Integer}};
    nchunks::Int=1
)
    ncblocks = length(c_dims)
    npblocks = length(p_dims)
    length(observations) == npblocks || throw(DimensionMismatch("Number of landmark observations ($(length(observations))) must match the number of eliminated blocks ($npblocks)"))
    nchunks >= 1 || throw(ArgumentError("nchunks must be at least 1, got $nchunks"))

    # Validate inputs here, rather than failing (or silently going wrong) deep inside the reduction
    for (i, d) in enumerate(c_dims)
        1 <= d <= 255 || throw(ArgumentError("Retained block $i has dimension $d; dimensions must be between 1 and 255"))
    end
    for (l, d) in enumerate(p_dims)
        1 <= d <= MAX_ELIMINATED_DIM || throw(ArgumentError("Eliminated block $l has dimension $d; supported dimensions are 1 to $MAX_ELIMINATED_DIM"))
    end
    seen = zeros(Int, ncblocks)   # seen[c] == l once landmark l has been observed by retained block c
    for (l, obs) in enumerate(observations)
        for c in obs
            1 <= c <= ncblocks || throw(ArgumentError("Eliminated block $l references retained block $c, but there are only $ncblocks retained blocks"))
            seen[c] == l && throw(ArgumentError("Retained block $c is listed more than once in the observations of eliminated block $l"))
            seen[c] = l
        end
    end

    c_dims_u8 = UInt8.(c_dims)
    p_dims_u8 = UInt8.(p_dims)

    nc = sum(Int, c_dims; init=0)

    boffsets_c = Vector{Int}(undef, ncblocks)
    if ncblocks > 0
        boffsets_c[1] = 1
        for i in 2:ncblocks
            boffsets_c[i] = boffsets_c[i-1] + Int(c_dims[i-1])
        end
    end

    boffsets_p = Vector{Int}(undef, npblocks)
    if npblocks > 0
        boffsets_p[1] = 1
        for i in 2:npblocks
            boffsets_p[i] = boffsets_p[i-1] + Int(p_dims[i-1])
        end
    end

    ppoffsets = Vector{Int}(undef, npblocks)
    if npblocks > 0
        ppoffsets[1] = 1
        for i in 2:npblocks
            ppoffsets[i] = ppoffsets[i-1] + Int(p_dims[i-1])^2
        end
    end

    cpptr = Vector{Int}(undef, npblocks + 1)
    cpptr[1] = 1
    for l in 1:npblocks
        cpptr[l+1] = cpptr[l] + length(observations[l])
    end
    num_couplings = cpptr[end] - 1

    cpblock = Vector{Int}(undef, num_couplings)
    cpoffsets = Vector{Int}(undef, num_couplings)
    curr_e_off = 1
    idx = 1
    for l in 1:npblocks
        dp = Int(p_dims[l])
        sorted_cams = sort(observations[l])
        for c in sorted_cams
            cpblock[idx] = c
            cpoffsets[idx] = curr_e_off
            curr_e_off += Int(c_dims[c]) * dp
            idx += 1
        end
    end

    max_obs_dim = 0
    for l in 1:npblocks
        obs_dim = sum(Int(c_dims[c]) for c in observations[l]; init=0)
        max_obs_dim = max(max_obs_dim, obs_dim)
    end

    p_groups = UnitRange{Int}[]
    p_sizes = Int[]
    if npblocks > 0
        curr_dim = Int(p_dims[1])
        start_idx = 1
        for l in 2:npblocks
            d = Int(p_dims[l])
            if d != curr_dim
                push!(p_groups, start_idx:l-1)
                push!(p_sizes, curr_dim)
                curr_dim = d
                start_idx = l
            end
        end
        push!(p_groups, start_idx:npblocks)
        push!(p_sizes, curr_dim)
    end

    chunks = Vector{Vector{UnitRange{Int}}}(undef, length(p_groups))
    for (g_idx, grp) in enumerate(p_groups)
        len = length(grp)
        group_chunks = Vector{UnitRange{Int}}(undef, nchunks)
        chunk_sz = cld(len, nchunks)
        for k in 1:nchunks
            first_idx = grp.start + (k - 1) * chunk_sz
            last_idx = min(grp.stop, first_idx + chunk_sz - 1)
            group_chunks[k] = first_idx <= last_idx ? (first_idx:last_idx) : (first_idx:first_idx-1)
        end
        chunks[g_idx] = group_chunks
    end

    return SchurStructure(
        nc, ncblocks, npblocks,
        c_dims_u8, p_dims_u8, boffsets_c, boffsets_p,
        ppoffsets, cpptr, cpblock, cpoffsets,
        max_obs_dim, p_groups, p_sizes, chunks
    )
end

"""
    SchurReductionWorkspace{T}

Pre-allocated scratch memory and caches for parallel Schur reduction and back-substitution.

The reduction runs in two phases. Phase 1 (parallel over chunks of eliminated blocks) factors every
`C_l`, caching `C_inv`, `z_l` and `W_cl = C_inv * E_cl'` (`W_data`, laid out like `E_data`). Phase 2
(parallel over `col_chunks`, disjoint ranges of retained blocks) has each chunk write straight into the
columns of `S` (and entries of `g_S`) that it owns, so no per-chunk copies of `S` are needed and nothing
has to be merged afterwards.
"""
struct SchurReductionWorkspace{T}
    W_scratch::Vector{Matrix{T}}          # per phase-1 chunk: max_Dp x max_Dp scratch for blocks above D_STATIC (0 x 0 otherwise)
    W_data::Vector{T}                     # W_cl = C_l^-1 E_cl' for every coupling, same layout as E_data
    C_inv_data::Vector{T}                 # Cached C_l(lambda)^-1: length sum_l d_l^2
    z_data::Vector{T}                     # Cached z_l = C_inv * g_p,l: length sum_l d_l
    cam_ptr::Vector{Int}                  # Retained-block-major CSR pointer into cam_entry
    cam_entry::Vector{Int}                # Coupling entries (indices into cpblock) of each retained block, ascending
    entry_landmark::Vector{Int}           # Eliminated block of each coupling entry
    col_chunks::Vector{UnitRange{Int}}    # Disjoint ranges of retained blocks, one per phase-2 chunk
    failed_landmark::Base.Threads.Atomic{Int}
end

"Split the retained blocks `1:n` into `K` contiguous ranges of roughly equal work (couplings plus the diagonal)."
function partition_columns(cam_ptr::Vector{Int}, K::Int)
    n = length(cam_ptr) - 1
    total = cam_ptr[end] - cam_ptr[1] + n
    ranges = UnitRange{Int}[]
    start = 1
    for k in 1:K
        stop = start - 1
        if k == K
            stop = n
        else
            target = (total * k) / K
            while stop < n && (cam_ptr[stop + 2] - cam_ptr[1] + (stop + 1)) <= target
                stop += 1
            end
        end
        push!(ranges, start:stop)
        start = stop + 1
    end
    return ranges
end

function SchurReductionWorkspace{T}(
    structure::SchurStructure, nchunks::Int;
    is_dense::Bool=true, nnz_S::Int=0   # accepted for compatibility: no per-chunk copies of S are needed any more
) where T
    max_Dp = maximum(structure.p_sizes; init=3)

    # One scratch matrix per phase-1 chunk, only used for blocks above D_STATIC
    nscratch = max(nchunks, maximum(length, structure.chunks; init=1))
    W_scratch = [Matrix{T}(undef, max_Dp > D_STATIC ? max_Dp : 0, max_Dp > D_STATIC ? max_Dp : 0) for _ in 1:nscratch]

    total_c_size = sum(Int(d)^2 for d in structure.p_dims; init=0)
    total_p_scalars = sum(Int(d) for d in structure.p_dims; init=0)
    C_inv_data = Vector{T}(undef, total_c_size)
    z_data = Vector{T}(undef, total_p_scalars)

    ncoup = length(structure.cpblock)
    total_e = 0
    for l in 1:structure.npblocks, e in structure.cpptr[l]:(structure.cpptr[l + 1] - 1)
        total_e += Int(structure.c_dims[structure.cpblock[e]]) * Int(structure.p_dims[l])
    end
    W_data = Vector{T}(undef, total_e)

    # Retained-block-major index of the couplings
    entry_landmark = Vector{Int}(undef, ncoup)
    counts = zeros(Int, structure.ncblocks + 1)
    for l in 1:structure.npblocks, e in structure.cpptr[l]:(structure.cpptr[l + 1] - 1)
        entry_landmark[e] = l
        counts[structure.cpblock[e] + 1] += 1
    end
    cam_ptr = cumsum(vcat(1, counts[2:end]))
    cam_entry = Vector{Int}(undef, ncoup)
    fill_pos = copy(cam_ptr)
    for e in 1:ncoup   # ascending e means ascending landmark within each retained block
        c = structure.cpblock[e]
        cam_entry[fill_pos[c]] = e
        fill_pos[c] += 1
    end

    return SchurReductionWorkspace{T}(
        W_scratch, W_data, C_inv_data, z_data,
        cam_ptr, cam_entry, entry_landmark, partition_columns(cam_ptr, max(nchunks, 1)),
        Base.Threads.Atomic{Int}(0)
    )
end

"""
    trial_workspace(ws)

A workspace for a second linear system with the same structure: own `C_inv`/`z` caches and failure flag;
the scratch (`W_data`, `W_scratch`) is only live within one solve and is shared.
"""
trial_workspace(ws::SchurReductionWorkspace{T}) where T = SchurReductionWorkspace{T}(
    ws.W_scratch, ws.W_data, similar(ws.C_inv_data), similar(ws.z_data),
    ws.cam_ptr, ws.cam_entry, ws.entry_landmark, ws.col_chunks, Base.Threads.Atomic{Int}(0)
)

const SchurBlockStructure = SchurStructure

"""
    SchurBlockMatrix{T, BType}

Block Hessian structure partitioned into retained camera block `B`,
eliminable landmark blocks `C_data`, coupling blocks `E_data`,
and bipartite indexing `structure::SchurStructure`.
"""
struct SchurBlockMatrix{T, BType}
    B::BType                         # Matrix{T} (dense) or SparseBlockLinearSystem (sparse)
    C_data::Vector{T}                # Contiguous storage for block-diagonal C: sum_l d_l^2
    E_data::Vector{T}                # Contiguous storage for block-sparse E: sum_e (d_c * d_l)
    structure::SchurStructure
end

SchurBlockMatrix(structure::SchurStructure, B::BType, C_data::Vector{T}, E_data::Vector{T}) where {T, BType} =
    SchurBlockMatrix{T, BType}(B, C_data, E_data, structure)

function zero!(H::SchurBlockMatrix)
    if H.B isa Matrix
        fill!(H.B, zero(eltype(H.B)))
    else
        zero!(H.B)
    end
    fill!(H.C_data, zero(eltype(H.C_data)))
    fill!(H.E_data, zero(eltype(H.E_data)))
    return H
end

"""
    copy_structure(H::SchurBlockMatrix)

Return a new `SchurBlockMatrix` with the same structure but freshly allocated, zeroed storage for
`B`, `C_data` and `E_data`. Nothing is shared with `H` except the immutable `SchurStructure`.
"""
function copy_structure(H::SchurBlockMatrix{T}) where T
    B = H.B isa AbstractMatrix ? zeros(T, size(H.B)) : copy_structure(H.B)
    return SchurBlockMatrix(H.structure, B, zeros(T, length(H.C_data)), zeros(T, length(H.E_data)))
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

# dot(a[ia:ia+dp-1], b[ib:ib+dp-1]) with a static or run-time dimension
@inline function dot_dim(a::Vector{T}, ia::Int, b::Vector{T}, ib::Int, s_dp::Union{StaticInt, Int}) where T
    acc = zero(T)
    @inbounds for i in 0:dimval(s_dp) - 1
        acc += a[ia + i] * b[ib + i]
    end
    return acc
end

# Call `f(s_dp)` with the block dimension `dp` as a StaticInt when `dp <= D_STATIC`, else as an Int
@inline function dispatch_dim(f::F, dp::Int) where F
    dp == 1 && return f(static(1))
    dp == 2 && return f(static(2))
    dp == 3 && return f(static(3))
    dp == 4 && return f(static(4))
    dp == 5 && return f(static(5))
    dp == 6 && return f(static(6))
    dp == 7 && return f(static(7))
    dp == 8 && return f(static(8))
    return f(dp)
end

# Threading helper. Polyester (the same mechanism NLLSModel.jl uses for assembly, so the two never compete
# for threads) has low per-call overhead, but loops still run serially unless more than one thread is
# available *and* there is enough work to be worth splitting.
const SCHUR_THREAD_MIN_WORK = Ref(4096)

@inline use_threads(nitems::Integer, work::Integer=nitems) = nitems > 1 && work >= SCHUR_THREAD_MIN_WORK[] && Threads.nthreads() > 1

# Chunk 1 accumulates straight into the reduced system (`S` or `S.nzval`); only chunks 2..K need their
# own buffers, which are merged in afterwards. With a single chunk no extra memory is needed at all.
@inline chunk_buffer(target::AbstractArray, extra::Vector, k::Int) = k == 1 ? target : @inbounds(extra[k - 1])

# Kept out of line so that the serial path does not pay for the closure Polyester builds
@noinline function batch_foreach(f::F, n::Int) where F
    Polyester.@batch for i in 1:n
        f(i)
    end
    return nothing
end

@inline function foreach_index(f::F, n::Int, threaded::Bool) where F
    if threaded
        batch_foreach(f, n)
    else
        for i in 1:n
            f(i)
        end
    end
    return nothing
end

"""
    factor_landmark!(C_inv_data, z_data, W_data, scratch, structure, C_data, E_data, b_p, scaling_p, lambda, l, s_dp)

Phase 1 for eliminated block `l`: invert `C_l + lambda * diag(scaling)`, cache `C_inv` and
`z_l = C_inv * g_p,l`, and store `W_cl = C_inv * E_cl'` for every observing retained block in `W_data`.
Returns `false` if the damped block is numerically singular or not positive definite.

`s_dp` is a `StaticInt` for blocks of up to `D_STATIC` dimensions (register arithmetic), otherwise the
`Int` dimension, which uses a LAPACK Cholesky on the `scratch` matrix (O(d^3) per eliminated block plus
O(d^2) per coupling row; no allocation).
"""
@inline function factor_landmark!(
    C_inv_data::Vector{T}, z_data::Vector{T}, W_data::Vector{T}, ::Matrix{T},
    structure::SchurStructure, C_data::Vector{T}, E_data::Vector{T},
    b_p::AbstractVector{T}, scaling_p::AbstractVector{T}, lambda::T, l::Int, s_dp::StaticInt{Dp}
) where {T, Dp}
    po = structure.ppoffsets[l]
    sc_off = structure.boffsets_p[l]

    res = inv_sym_block(make_smat(C_data, po, s_dp), lambda, make_svec(scaling_p, sc_off, s_dp))
    res.success || return false
    C_inv = res.invH
    @inbounds for i in 1:(Dp * Dp)
        C_inv_data[po + i - 1] = C_inv[i]
    end

    z_l = C_inv * make_svec(b_p, sc_off, s_dp)
    @inbounds for i in 1:Dp
        z_data[sc_off + i - 1] = z_l[i]
    end

    @inbounds for e in structure.cpptr[l]:(structure.cpptr[l + 1] - 1)
        eo = structure.cpoffsets[e]
        dc = Int(structure.c_dims[structure.cpblock[e]])
        for r in 1:dc
            w_r = C_inv * make_svec(E_data, eo + (r - 1) * Dp, s_dp)
            for i in 1:Dp
                W_data[eo + (r - 1) * Dp + i - 1] = w_r[i]
            end
        end
    end
    return true
end

@inline function factor_landmark!(
    C_inv_data::Vector{T}, z_data::Vector{T}, W_data::Vector{T}, scratch::Matrix{T},
    structure::SchurStructure, C_data::Vector{T}, E_data::Vector{T},
    b_p::AbstractVector{T}, scaling_p::AbstractVector{T}, lambda::T, l::Int, dp::Int
) where T
    po = structure.ppoffsets[l]
    sc_off = structure.boffsets_p[l]

    M = view(scratch, 1:dp, 1:dp)
    @inbounds for j in 1:dp, i in 1:dp
        M[i, j] = C_data[po + (j - 1) * dp + i - 1]
    end
    @inbounds for i in 1:dp
        M[i, i] += lambda * scaling_p[sc_off + i - 1]
    end
    # Cholesky; every pivot must be safely positive relative to its (damped) diagonal entry. The damped
    # diagonal entry is recomputed from the inputs.
    _, info = LinearAlgebra.LAPACK.potrf!('U', M)
    info == 0 || return false
    @inbounds for i in 1:dp
        hii = C_data[po + (i - 1) * (dp + 1)] + lambda * scaling_p[sc_off + i - 1]
        ratio = abs2(M[i, i]) / hii
        (isfinite(ratio) && ratio > singular_tol(T)) || return false
    end
    LinearAlgebra.LAPACK.potri!('U', M)

    @inbounds for j in 1:dp, i in 1:j
        v = M[i, j]
        C_inv_data[po + (j - 1) * dp + i - 1] = v
        C_inv_data[po + (i - 1) * dp + j - 1] = v
    end

    @inbounds for i in 1:dp
        acc = zero(T)
        for j in 1:dp
            acc += C_inv_data[po + (j - 1) * dp + i - 1] * b_p[sc_off + j - 1]
        end
        z_data[sc_off + i - 1] = acc
    end

    @inbounds for e in structure.cpptr[l]:(structure.cpptr[l + 1] - 1)
        eo = structure.cpoffsets[e]
        dc = Int(structure.c_dims[structure.cpblock[e]])
        for r in 1:dc
            row_off = eo + (r - 1) * dp
            for i in 1:dp
                acc = zero(T)
                for j in 1:dp
                    acc += C_inv_data[po + (j - 1) * dp + i - 1] * E_data[row_off + j - 1]
                end
                W_data[row_off + i - 1] = acc
            end
        end
    end
    return true
end

"""
    factor_landmarks_group!(ws..., p_chunk, s_dp) -> Bool

Phase 1 over a chunk of eliminated blocks of uniform dimension. On failure the lowest failing
eliminated block index is recorded in `failed_landmark`.
"""
function factor_landmarks_group!(
    C_inv_data::Vector{T}, z_data::Vector{T}, W_data::Vector{T}, scratch::Matrix{T},
    failed_landmark::Base.Threads.Atomic{Int},
    structure::SchurStructure, C_data::Vector{T}, E_data::Vector{T},
    b_p::AbstractVector{T}, scaling_p::AbstractVector{T}, lambda::T,
    p_chunk::UnitRange{Int}, s_dp::Union{StaticInt, Int}
) where T
    @inbounds for l in p_chunk
        failed_landmark[] != 0 && return false
        if !factor_landmark!(C_inv_data, z_data, W_data, scratch, structure, C_data, E_data, b_p, scaling_p, lambda, l, s_dp)
            old = failed_landmark[]
            while (old == 0 || l < old)
                old_val = Threads.atomic_cas!(failed_landmark, old, l)
                old_val == old && break
                old = failed_landmark[]
            end
            return false
        end
    end
    return true
end

# Contribution of coupling entry e2 (eliminated block l, retained block c2) to g_S and to the column
# block c2 of S: g_S[c2] -= E_l,c2 z_l and S[c1, c2] -= W_l,c1' E_l,c2 for every c1 <= c2 observing l.
@inline function accumulate_entry_dense!(
    S::Matrix{T}, g_S::Vector{T}, structure::SchurStructure, E_data::Vector{T}, W_data::Vector{T}, z_data::Vector{T},
    e2::Int, l::Int, s_dp::Union{StaticInt, Int}
) where T
    dp = dimval(s_dp)
    c2 = structure.cpblock[e2]
    dc2 = Int(structure.c_dims[c2])
    co2 = structure.boffsets_c[c2]
    eo2 = structure.cpoffsets[e2]
    sc_off = structure.boffsets_p[l]
    @inbounds for r in 1:dc2
        g_S[co2 + r - 1] -= dot_dim(E_data, eo2 + (r - 1) * dp, z_data, sc_off, s_dp)
    end
    @inbounds for e1 in structure.cpptr[l]:e2
        c1 = structure.cpblock[e1]
        dc1 = Int(structure.c_dims[c1])
        co1 = structure.boffsets_c[c1]
        wo1 = structure.cpoffsets[e1]
        for col in 1:dc2
            e_off = eo2 + (col - 1) * dp
            nrow = c1 == c2 ? col : dc1      # diagonal block: upper triangle only
            for row in 1:nrow
                S[co1 + row - 1, co2 + col - 1] -= dot_dim(W_data, wo1 + (row - 1) * dp, E_data, e_off, s_dp)
            end
        end
    end
    return nothing
end

"""
    accumulate_dense_columns!(S, g_S, structure, E_data, W_data, z_data, cam_ptr, cam_entry, entry_landmark, c_range)

Phase 2 for the retained blocks in `c_range`: write all contributions to their columns of the dense upper
triangle of `S` and to `g_S` directly. Chunks with disjoint `c_range` never write the same entry.
"""
function accumulate_dense_columns!(
    S::Matrix{T}, g_S::Vector{T}, structure::SchurStructure, E_data::Vector{T}, W_data::Vector{T}, z_data::Vector{T},
    cam_ptr::Vector{Int}, cam_entry::Vector{Int}, entry_landmark::Vector{Int}, c_range::UnitRange{Int}
) where T
    @inbounds for c2 in c_range
        for k in cam_ptr[c2]:(cam_ptr[c2 + 1] - 1)
            e2 = cam_entry[k]
            l = entry_landmark[e2]
            dispatch_dim(Int(structure.p_dims[l])) do s_dp
                accumulate_entry_dense!(S, g_S, structure, E_data, W_data, z_data, e2, l, s_dp)
            end
        end
    end
    return nothing
end

@inline function accumulate_entry_sparse!(
    nzval::Vector{T}, g_S::Vector{T}, structure::SchurStructure, E_data::Vector{T}, W_data::Vector{T}, z_data::Vector{T},
    S_colptr::Vector{Ti}, block_colptr::Vector{Ti}, block_rowval::Vector{Ti}, block_row_offsets::Vector{Int}, col_heights::Vector{Int},
    e2::Int, l::Int, s_dp::Union{StaticInt, Int}
) where {T, Ti}
    dp = dimval(s_dp)
    c2 = structure.cpblock[e2]
    dc2 = Int(structure.c_dims[c2])
    co2 = structure.boffsets_c[c2]
    eo2 = structure.cpoffsets[e2]
    sc_off = structure.boffsets_p[l]
    @inbounds for r in 1:dc2
        g_S[co2 + r - 1] -= dot_dim(E_data, eo2 + (r - 1) * dp, z_data, sc_off, s_dp)
    end
    col_base = S_colptr[co2]
    stride = col_heights[c2]
    b_row_idx = block_colptr[c2]
    b_row_end = block_colptr[c2 + 1] - 1
    @inbounds for e1 in structure.cpptr[l]:e2
        c1 = structure.cpblock[e1]
        dc1 = Int(structure.c_dims[c1])
        wo1 = structure.cpoffsets[e1]
        # monotone merge-walk down the block column
        while b_row_idx <= b_row_end && block_rowval[b_row_idx] < c1
            b_row_idx += 1
        end
        @assert b_row_idx <= b_row_end && block_rowval[b_row_idx] == c1 "Sparse pattern missing entry ($c1, $c2)"
        row_offset = block_row_offsets[b_row_idx]
        for col in 1:dc2
            e_off = eo2 + (col - 1) * dp
            col_ptr = col_base + (col - 1) * stride + row_offset
            nrow = c1 == c2 ? col : dc1
            for row in 1:nrow
                nzval[col_ptr + row - 1] -= dot_dim(W_data, wo1 + (row - 1) * dp, E_data, e_off, s_dp)
            end
        end
    end
    return nothing
end

"""
    accumulate_sparse_columns!(nzval, g_S, structure, E_data, W_data, z_data, cam_ptr, cam_entry, entry_landmark,
                               S_colptr, block_colptr, block_rowval, block_row_offsets, col_heights, c_range)

Sparse counterpart of [`accumulate_dense_columns!`](@ref), writing into `S.nzval`.
"""
function accumulate_sparse_columns!(
    nzval::Vector{T}, g_S::Vector{T}, structure::SchurStructure, E_data::Vector{T}, W_data::Vector{T}, z_data::Vector{T},
    cam_ptr::Vector{Int}, cam_entry::Vector{Int}, entry_landmark::Vector{Int},
    S_colptr::Vector{Ti}, block_colptr::Vector{Ti}, block_rowval::Vector{Ti},
    block_row_offsets::Vector{Int}, col_heights::Vector{Int}, c_range::UnitRange{Int}
) where {T, Ti}
    @inbounds for c2 in c_range
        for k in cam_ptr[c2]:(cam_ptr[c2 + 1] - 1)
            e2 = cam_entry[k]
            l = entry_landmark[e2]
            dispatch_dim(Int(structure.p_dims[l])) do s_dp
                accumulate_entry_sparse!(nzval, g_S, structure, E_data, W_data, z_data, S_colptr, block_colptr, block_rowval,
                                         block_row_offsets, col_heights, e2, l, s_dp)
            end
        end
    end
    return nothing
end

"""
    reset_schur_dense!(S, B, scaling_c, lambda, ws)

Reset dense reduced system S = B + lambda * diag(scaling_c) and the failure flag.
"""
function reset_schur_dense!(
    S::Matrix{T}, B::Matrix{T}, scaling_c::AbstractVector{T},
    lambda::T, ws::SchurReductionWorkspace{T}
) where T
    copyto!(S, B)
    @inbounds for (i, sc) in enumerate(scaling_c)
        S[i, i] += lambda * sc
    end
    ws.failed_landmark[] = 0
    return nothing
end

"""
    reset_schur_sparse!(nzval, B_nzval, b_to_s_map, diag_inds, scaling_c, lambda, ws)

Reset sparse reduced system S = B + lambda * diag(scaling_c) and the failure flag.
"""
function reset_schur_sparse!(
    nzval::Vector{T}, B_nzval::Vector{T}, b_to_s_map::Vector{Int},
    diag_inds::Vector{Int}, scaling_c::AbstractVector{T},
    lambda::T, ws::SchurReductionWorkspace{T}
) where T
    fill!(nzval, zero(T))
    @inbounds for i in eachindex(b_to_s_map)
        idx = b_to_s_map[i]
        if idx > 0
            nzval[idx] = B_nzval[i]
        end
    end
    @inbounds for (i, idx) in enumerate(diag_inds)
        nzval[idx] += lambda * scaling_c[i]
    end
    ws.failed_landmark[] = 0
    return nothing
end

# Phase 1 for all groups of eliminated blocks; false if any block failed
function factor_all_landmarks!(
    structure::SchurStructure, C_data::Vector{T}, E_data::Vector{T}, b_p::AbstractVector{T},
    scaling_p::AbstractVector{T}, lambda::T, ws::SchurReductionWorkspace{T}
) where T
    for (g_idx, p_range) in enumerate(structure.p_groups)
        chunks = structure.chunks[g_idx]
        threaded = use_threads(length(chunks), length(structure.cpblock))
        dispatch_dim(structure.p_sizes[g_idx]) do s_dp
            foreach_index(length(chunks), threaded) do k
                factor_landmarks_group!(ws.C_inv_data, ws.z_data, ws.W_data, ws.W_scratch[k], ws.failed_landmark,
                                        structure, C_data, E_data, b_p, scaling_p, lambda, chunks[k], s_dp)
            end
        end
        ws.failed_landmark[] != 0 && return false
    end
    return true
end

"""
    reduce_schur_dense!(S, g_S, b_c, b_p, scaling_p, lambda, structure, C_data, E_data, ws)

Two-phase Schur reduction into the dense upper triangle of `S` (already holding `B` plus damping) and `g_S`.
"""
function reduce_schur_dense!(
    S::Matrix{T}, g_S::Vector{T}, b_c::AbstractVector{T}, b_p::AbstractVector{T},
    scaling_p::AbstractVector{T}, lambda::T, structure::SchurStructure,
    C_data::Vector{T}, E_data::Vector{T}, ws::SchurReductionWorkspace{T}
) where T
    factor_all_landmarks!(structure, C_data, E_data, b_p, scaling_p, lambda, ws) || return false
    copyto!(g_S, b_c)
    nk = length(ws.col_chunks)
    foreach_index(nk, use_threads(nk, length(structure.cpblock))) do k
        accumulate_dense_columns!(S, g_S, structure, E_data, ws.W_data, ws.z_data, ws.cam_ptr, ws.cam_entry, ws.entry_landmark, ws.col_chunks[k])
    end
    return true
end

"""
    reduce_schur_sparse!(S, g_S, b_c, b_p, scaling_p, lambda, structure, C_data, E_data,
                         block_colptr, block_rowval, block_row_offsets, col_heights, ws)

Two-phase Schur reduction into `S.nzval` (already holding `B` plus damping) and `g_S`.
"""
function reduce_schur_sparse!(
    S::SparseMatrixCSC{T, Ti}, g_S::Vector{T}, b_c::AbstractVector{T}, b_p::AbstractVector{T},
    scaling_p::AbstractVector{T}, lambda::T, structure::SchurStructure,
    C_data::Vector{T}, E_data::Vector{T},
    block_colptr::Vector{Ti}, block_rowval::Vector{Ti},
    block_row_offsets::Vector{Int}, col_heights::Vector{Int},
    ws::SchurReductionWorkspace{T}
) where {T, Ti}
    factor_all_landmarks!(structure, C_data, E_data, b_p, scaling_p, lambda, ws) || return false
    copyto!(g_S, b_c)
    nk = length(ws.col_chunks)
    foreach_index(nk, use_threads(nk, length(structure.cpblock))) do k
        accumulate_sparse_columns!(S.nzval, g_S, structure, E_data, ws.W_data, ws.z_data, ws.cam_ptr, ws.cam_entry, ws.entry_landmark,
                                   S.colptr, block_colptr, block_rowval, block_row_offsets, col_heights, ws.col_chunks[k])
    end
    return true
end

"""
    backsubstitute!(x_p, x_c, structure, E_data, C_inv_data, z_data)

Recover step Delta x_p for all eliminable blocks independently using cached C_inv and z:
`x_p,l = z_l - C_inv,l * (sum_c E_cl' x_c,c)`. The coupling sum is accumulated first and `C_inv` is
applied once per eliminable block.
"""
function backsubstitute!(
    x_p::AbstractVector{T}, x_c::AbstractVector{T},
    structure::SchurStructure, E_data::Vector{T},
    C_inv_data::Vector{T}, z_data::Vector{T}
) where T
    threaded = use_threads(structure.npblocks, length(structure.cpblock))
    for (g_idx, p_range) in enumerate(structure.p_groups)
        dp = structure.p_sizes[g_idx]
        chunks = structure.chunks[g_idx]
        # Function barrier on the (static) block dimension
        dispatch_dim(dp) do s_dp
            foreach_index(length(chunks), threaded) do k
                backsubstitute_group!(x_p, x_c, structure, E_data, C_inv_data, z_data, chunks[k], s_dp)
            end
        end
    end
    return nothing
end

function backsubstitute_group!(
    x_p::AbstractVector{T}, x_c::AbstractVector{T},
    structure::SchurStructure, E_data::Vector{T},
    C_inv_data::Vector{T}, z_data::Vector{T},
    p_chunk::UnitRange{Int}, s_dp::StaticInt{Dp}
) where {T, Dp}
    cpptr = structure.cpptr
    cpblock = structure.cpblock
    cpoffsets = structure.cpoffsets
    boffsets_c = structure.boffsets_c
    boffsets_p = structure.boffsets_p
    ppoffsets = structure.ppoffsets
    c_dims = structure.c_dims

    @inbounds for l in p_chunk
        p_off = boffsets_p[l]

        # t = sum over observing retained blocks of E_cl' * x_c,c
        t = zero(SVector{Dp, T})
        for e in cpptr[l]:(cpptr[l + 1] - 1)
            c = cpblock[e]
            dc = Int(c_dims[c])
            co = boffsets_c[c]
            eo = cpoffsets[e]
            for r in 1:dc
                t += make_svec(E_data, eo + (r - 1) * Dp, s_dp) * x_c[co + r - 1]
            end
        end

        x_l = make_svec(z_data, p_off, s_dp) - make_smat(C_inv_data, ppoffsets[l], s_dp) * t
        for i in 1:Dp
            x_p[p_off + i - 1] = x_l[i]
        end
    end
    return nothing
end

# Blocks above D_STATIC dimensions: same computation with run-time loops and one small temporary per call
function backsubstitute_group!(
    x_p::AbstractVector{T}, x_c::AbstractVector{T},
    structure::SchurStructure, E_data::Vector{T},
    C_inv_data::Vector{T}, z_data::Vector{T},
    p_chunk::UnitRange{Int}, dp::Int
) where T
    t = Vector{T}(undef, dp)
    @inbounds for l in p_chunk
        p_off = structure.boffsets_p[l]
        po = structure.ppoffsets[l]
        fill!(t, zero(T))
        for e in structure.cpptr[l]:(structure.cpptr[l + 1] - 1)
            c = structure.cpblock[e]
            dc = Int(structure.c_dims[c])
            co = structure.boffsets_c[c]
            eo = structure.cpoffsets[e]
            for r in 1:dc
                xc = x_c[co + r - 1]
                for i in 1:dp
                    t[i] += E_data[eo + (r - 1) * dp + i - 1] * xc
                end
            end
        end
        for i in 1:dp
            acc = z_data[p_off + i - 1]
            for j in 1:dp
                acc -= C_inv_data[po + (j - 1) * dp + i - 1] * t[j]
            end
            x_p[p_off + i - 1] = acc
        end
    end
    return nothing
end

"""
    expand_block_perm(block_perm::AbstractVector{<:Integer}, block_sizes::AbstractVector{<:Integer}) -> Vector{Int}

Expand a block-level permutation into a scalar-level permutation vector.
"""
function expand_block_perm(block_perm::AbstractVector{<:Integer}, block_sizes::AbstractVector{<:Integer})
    nblocks = length(block_sizes)
    @assert length(block_perm) == nblocks "block_perm and block_sizes must have equal length"
    total_dim = sum(Int, block_sizes; init=0)
    scalar_perm = Vector{Int}(undef, total_dim)
    boffsets = Vector{Int}(undef, nblocks)
    boffsets[1] = 1
    for i in 2:nblocks
        boffsets[i] = boffsets[i - 1] + Int(block_sizes[i - 1])
    end
    cursor = 1
    for b in block_perm
        bo = boffsets[b]
        bs = Int(block_sizes[b])
        for r in 1:bs
            scalar_perm[cursor] = bo + r - 1
            cursor += 1
        end
    end
    return scalar_perm
end

"""
    sort_schur_blocks(is_eliminable::AbstractVector{Bool}, block_sizes::AbstractVector{<:Integer}, observations::AbstractVector{<:AbstractVector{<:Integer}}) -> Vector{Int}

Two-pass variable ordering:
- Pass 1: Retained variables (role 0) ordered by dimension descending; eliminable variables (role 1) grouped by dimension descending.
- Pass 2: Within each eliminable dimension group, sort landmarks by primary observing camera index c_min(l) = min_{c in obs(l)} c.
Returns a block permutation vector mapping old block indices to new positions.
"""
function sort_schur_blocks(
    is_eliminable::AbstractVector{Bool},
    block_sizes::AbstractVector{<:Integer},
    observations::AbstractVector{<:AbstractVector}
)
    nblocks = length(is_eliminable)
    @assert length(block_sizes) == nblocks "is_eliminable and block_sizes must have equal length"

    # Retained blocks (sorted by dimension descending)
    c_indices = findall(!, is_eliminable)
    sort!(c_indices, by = i -> -Int(block_sizes[i]))

    # Eliminable blocks
    p_indices = findall(identity, is_eliminable)
    # Group by dimension descending, then within dimension by c_min
    sort!(p_indices, by = function(i)
        dp = Int(block_sizes[i])
        obs = observations[i]
        c_min = isempty(obs) ? typemax(Int) : minimum(obs)
        return (-dp, c_min)
    end)

    return [c_indices; p_indices]
end

"""
    build_schur_covisibility(structure::SchurStructure, [T=Float64]; extra_blocks=()) -> NamedTuple

Compute block-level and scalar-level co-visibility sparsity patterns for the reduced camera system S.
Returns `(block_covis, block_colptr, block_rowval, block_row_offsets, col_heights, S_sparse, diag_inds)`;
`S_sparse` is a zero-valued `SparseMatrixCSC{T}`.

The pattern contains every pair of retained blocks that share an eliminated block, plus the diagonal.
Retained-retained couplings that do not go through an eliminated block (e.g. odometry factors or loop
closures between poses with no common landmark) have no such slot, so list them in `extra_blocks`, an
iterable of `(i, j)` retained block index pairs (in either order; `i == j` is allowed and ignored).
Use [`retained_block_pairs`](@ref) to obtain them from the block pattern of `B`.
"""
function build_schur_covisibility(structure::SchurStructure, ::Type{T}=Float64; extra_blocks=()) where T
    ncblocks = structure.ncblocks
    npblocks = structure.npblocks
    c_dims = structure.c_dims
    cpptr = structure.cpptr
    cpblock = structure.cpblock
    boffsets_c = structure.boffsets_c
    nc = structure.nc

    # 1. Build camera block co-visibility graph (upper triangle). Collect (I, J) pairs and build the
    # sparse matrix once: inserting entries one at a time into a SparseMatrixCSC is O(nnz^2).
    I_cov = Int[]
    J_cov = Int[]
    for l in 1:npblocks
        e_start = cpptr[l]
        e_end = cpptr[l + 1] - 1
        for idx2 in e_start:e_end
            c2 = cpblock[idx2]
            for idx1 in e_start:idx2
                push!(I_cov, cpblock[idx1])
                push!(J_cov, c2)
            end
        end
    end
    for (i, j) in extra_blocks
        (1 <= i <= ncblocks && 1 <= j <= ncblocks) || throw(ArgumentError("Extra block pair ($i, $j) is outside the $ncblocks retained blocks"))
        push!(I_cov, min(i, j))
        push!(J_cov, max(i, j))
    end
    for c in 1:ncblocks
        push!(I_cov, c)
        push!(J_cov, c)
    end
    block_covis = sparse(I_cov, J_cov, true, ncblocks, ncblocks, |)

    block_colptr = Vector{Int}(block_covis.colptr)
    block_rowval = Vector{Int}(block_covis.rowval)
    nnz_blocks = nnz(block_covis)

    # 2. Compute scalar row offsets and column heights
    col_heights = zeros(Int, ncblocks)
    block_row_offsets = zeros(Int, nnz_blocks)
    for c2 in 1:ncblocks
        h = 0
        for idx in block_colptr[c2]:(block_colptr[c2 + 1] - 1)
            c1 = block_rowval[idx]
            block_row_offsets[idx] = h
            h += Int(c_dims[c1])
        end
        col_heights[c2] = h
    end

    # 3. Build scalar SparseMatrixCSC S pattern
    I_list = Int[]
    J_list = Int[]
    for c2 in 1:ncblocks
        for idx in block_colptr[c2]:(block_colptr[c2 + 1] - 1)
            c1 = block_rowval[idx]
            for c_col in 1:c_dims[c2], c_row in 1:c_dims[c1]
                push!(I_list, boffsets_c[c1] + c_row - 1)
                push!(J_list, boffsets_c[c2] + c_col - 1)
            end
        end
    end
    S_sparse = sparse(I_list, J_list, zeros(T, length(I_list)), nc, nc)

    # 4. Compute diagonal indices in S_sparse.nzval
    diag_inds = zeros(Int, nc)
    for j in 1:nc
        for idx in S_sparse.colptr[j]:(S_sparse.colptr[j + 1] - 1)
            if S_sparse.rowval[idx] == j
                diag_inds[j] = idx
                break
            end
        end
        @assert diag_inds[j] != 0 "Missing diagonal entry for camera scalar $j"
    end

    return (
        block_covis = block_covis,
        block_colptr = block_colptr,
        block_rowval = block_rowval,
        block_row_offsets = block_row_offsets,
        col_heights = col_heights,
        S_sparse = S_sparse,
        diag_inds = diag_inds
    )
end

"""
    retained_block_pairs(B::SparseBlockLinearSystem) -> Vector{Tuple{Int, Int}}

Off-diagonal stored blocks of `B` as `(row_block, col_block)` pairs. Pass them as `extra_blocks` to
[`build_schur_covisibility`](@ref) so the reduced system has a slot for every coupling in `B`.
"""
function retained_block_pairs(B::SparseBlockLinearSystem)
    cols, rows, _ = SparseArrays.findnz(B.indicestransposed)
    return [(Int(r), Int(c)) for (c, r) in zip(cols, rows) if r != c]
end

"""
    build_b_to_s_map(B_sparse::SparseMatrixCSC, S_sparse::SparseMatrixCSC) -> Vector{Int}

Build mapping vector from B_sparse.nzval to S_sparse.nzval.
"""
function build_b_to_s_map(B_sparse::SparseMatrixCSC, S_sparse::SparseMatrixCSC)
    b_to_s = Int[]
    for j in 1:B_sparse.n
        for k in B_sparse.colptr[j]:(B_sparse.colptr[j + 1] - 1)
            i = B_sparse.rowval[k]
            s_k = find_in_column(S_sparse, i, j)
            @assert s_k != 0 "B entry ($i, $j) not present in S sparsity pattern"
            push!(b_to_s, s_k)
        end
    end
    return b_to_s
end

# Index of `row` in column `j` of the sorted-pattern CSC matrix `S` (0 if absent), by binary search
@inline function find_in_column(S::SparseMatrixCSC, row::Integer, j::Integer)
    lo = Int(S.colptr[j])
    hi = Int(S.colptr[j + 1]) - 1
    idx = lo + searchsortedfirst(view(S.rowval, lo:hi), row) - 1
    return (idx <= hi && S.rowval[idx] == row) ? idx : 0
end

"""
    build_b_to_s_map(B_sys::SparseBlockLinearSystem, S_sparse::SparseMatrixCSC, boffsets_c::Vector{Int}) -> Vector{Int}

Build mapping vector from B_sys.data to S_sparse.nzval for symmetric upper-triangle reduction.
Entries of `B_sys.data` that are not mapped (the strict lower triangle of diagonal blocks) are 0.
"""
function build_b_to_s_map(B_sys::SparseBlockLinearSystem, S_sparse::SparseMatrixCSC, boffsets_c::Vector{Int})
    b_to_s = zeros(Int, length(B_sys.data))
    for (c, r, data_start) in zip(SparseArrays.findnz(B_sys.indicestransposed)...)
        dr = Int(B_sys.rowblocksizes[r])
        dc = Int(B_sys.columnblocksizes[c])

        if r == c
            for col in 1:dc
                scol = boffsets_c[r] + col - 1
                for row in 1:col
                    srow = boffsets_c[r] + row - 1
                    s_k = find_in_column(S_sparse, srow, scol)
                    @assert s_k != 0 "Missing diagonal entry ($srow, $scol) in S"
                    b_to_s[data_start + (col - 1)*dr + row - 1] = s_k
                end
            end
        elseif r > c
            for col in 1:dc
                srow = boffsets_c[c] + col - 1
                for row in 1:dr
                    scol = boffsets_c[r] + row - 1
                    s_k = find_in_column(S_sparse, srow, scol)
                    @assert s_k != 0 "Missing entry ($srow, $scol) in S"
                    b_to_s[data_start + (col - 1)*dr + row - 1] = s_k
                end
            end
        end
    end
    return b_to_s
end

"""
    build_b_to_s_map(B_sys::SparseBlockLinearSystem, covis, boffsets_c::Vector{Int}) -> Vector{Int}

As above, but using the block-level pattern in `covis` (the result of [`build_schur_covisibility`](@ref)):
one block-level search per stored block, then the scalar position follows from `block_row_offsets` and
`col_heights` with integer arithmetic.
"""
function build_b_to_s_map(B_sys::SparseBlockLinearSystem, covis::NamedTuple, boffsets_c::Vector{Int})
    S_colptr = covis.S_sparse.colptr
    b_to_s = zeros(Int, length(B_sys.data))
    for (c, r, data_start) in zip(SparseArrays.findnz(B_sys.indicestransposed)...)
        r >= c || continue   # only lower-triangle blocks are stored in B
        dr = Int(B_sys.rowblocksizes[r])
        dc = Int(B_sys.columnblocksizes[c])
        # Block (c, r) of the upper-triangle pattern: block column r, block row c
        rng = covis.block_colptr[r]:(covis.block_colptr[r + 1] - 1)
        k = searchsortedfirst(view(covis.block_rowval, rng), c)
        (k <= length(rng) && covis.block_rowval[rng[k]] == c) || throw(AssertionError("Block ($c, $r) of B is not present in the pattern of S"))
        row_offset = covis.block_row_offsets[rng[k]]
        base = S_colptr[boffsets_c[r]] + row_offset
        stride = covis.col_heights[r]
        if r == c
            for col in 1:dc, row in 1:col
                b_to_s[data_start + (col - 1)*dr + row - 1] = base + (col - 1) * stride + row - 1
            end
        else
            for col in 1:dc, row in 1:dr
                b_to_s[data_start + (col - 1)*dr + row - 1] = base + (row - 1) * stride + col - 1
            end
        end
    end
    return b_to_s
end

"""
    fast_bAb(H::SchurBlockMatrix, dx::AbstractVector) -> Real

Compute quadratic form dx' * H * dx in O(N) operations without materializing the full Hessian.
`dx` may have any element type; the result has the promoted type.

The retained part (`B`) is walked block by block. The eliminated part (`C` and the couplings `E`) is where
nearly all the work is, and is processed per group of eliminated blocks of uniform dimension with that
dimension known statically (3.4x faster than per-block run-time sizes on a 200-camera, 20000-landmark
problem). Specialising the retained part by run of retained block sizes was measured too: it saved nothing
measurable (that part takes ~0.01 ms) and cost 0.2-2 s of extra compilation, so it is not done.
"""
function fast_bAb(H::SchurBlockMatrix{T}, dx::AbstractVector) where T
    R = promote_type(T, eltype(dx))
    structure = H.structure
    nc = structure.nc
    dx_c = view(dx, 1:nc)
    dx_p = view(dx, nc+1:length(dx))
    
    # 1. dx_c' * B * dx_c
    total = zero(R)
    if H.B isa Matrix{T}
        B = H.B
        @inbounds for j in 1:nc
            val = zero(T)
            for i in 1:j
                b_val = B[i, j]
                if i == j
                    val += b_val * dx_c[i]
                else
                    val += b_val * dx_c[i]
                    total += b_val * dx_c[j] * dx_c[i]
                end
            end
            total += dx_c[j] * val
        end
    elseif H.B isa SparseBlockLinearSystem{T}
        # Walk the stored block pattern directly (column i of indicestransposed lists the stored
        # column-blocks j of block row i); each block is column-major at data[ptr:ptr+dr*dc-1].
        B_sys = H.B
        boffsets = structure.boffsets_c
        data = B_sys.data
        colptr = B_sys.indicestransposed.colptr
        rowval = B_sys.indicestransposed.rowval
        nzval = B_sys.indicestransposed.nzval
        @inbounds for i in 1:length(B_sys.rowblocksizes)
            r_start = boffsets[i]
            dr = B_sys.rowblocksizes[i]
            for k in colptr[i]:(colptr[i + 1] - 1)
                j = rowval[k]
                ptr = nzval[k]
                c_start = boffsets[j]
                dc = B_sys.columnblocksizes[j]
                blk_total = zero(T)
                for c in 1:dc
                    dxc = dx_c[c_start + c - 1]
                    col_ptr = ptr + (c - 1) * dr - 1
                    col_total = zero(T)
                    for r in 1:dr
                        col_total += data[col_ptr + r] * dx_c[r_start + r - 1]
                    end
                    blk_total += col_total * dxc
                end
                total += (i == j ? blk_total : 2 * blk_total)
            end
        end
    end

    # 2. 2 * dx_c' * E * dx_p and 3. dx_p' * C * dx_p, per group of eliminated blocks of uniform dimension
    for (g_idx, p_range) in enumerate(structure.p_groups)
        total += dispatch_dim(structure.p_sizes[g_idx]) do s_dp
            fast_bAb_eliminated(H, dx_c, dx_p, p_range, s_dp, R)
        end
    end

    return total
end

# E and C contributions of the eliminated blocks in `p_range`, all of dimension `s_dp` (static up to D_STATIC)
function fast_bAb_eliminated(H::SchurBlockMatrix{T}, dx_c, dx_p, p_range::UnitRange{Int}, s_dp::StaticInt, ::Type{R}) where {T, R}
    structure = H.structure
    D = dimval(s_dp)
    part = zero(R)
    @inbounds for l in p_range
        p_off = structure.boffsets_p[l]
        xp = make_svec(dx_p, p_off, s_dp)
        part += dot(xp, make_smat(H.C_data, structure.ppoffsets[l], s_dp) * xp)
        for e in structure.cpptr[l]:(structure.cpptr[l + 1] - 1)
            c = structure.cpblock[e]
            dc = Int(structure.c_dims[c])
            co = structure.boffsets_c[c]
            eo = structure.cpoffsets[e]
            for r in 1:dc
                part += 2 * dx_c[co + r - 1] * dot(make_svec(H.E_data, eo + (r - 1) * D, s_dp), xp)
            end
        end
    end
    return part
end

# Blocks above D_STATIC dimensions: the same computation with run-time loops
function fast_bAb_eliminated(H::SchurBlockMatrix{T}, dx_c, dx_p, p_range::UnitRange{Int}, dp::Int, ::Type{R}) where {T, R}
    structure = H.structure
    part = zero(R)
    @inbounds for l in p_range
        p_off = structure.boffsets_p[l]
        po = structure.ppoffsets[l]
        for col in 1:dp
            xcol = dx_p[p_off + col - 1]
            for row in 1:dp
                part += H.C_data[po + (col - 1) * dp + row - 1] * xcol * dx_p[p_off + row - 1]
            end
        end
        for e in structure.cpptr[l]:(structure.cpptr[l + 1] - 1)
            c = structure.cpblock[e]
            dc = Int(structure.c_dims[c])
            co = structure.boffsets_c[c]
            eo = structure.cpoffsets[e]
            for r in 1:dc
                acc = zero(R)
                for i in 1:dp
                    acc += H.E_data[eo + (r - 1) * dp + i - 1] * dx_p[p_off + i - 1]
                end
                part += 2 * dx_c[co + r - 1] * acc
            end
        end
    end
    return part
end
