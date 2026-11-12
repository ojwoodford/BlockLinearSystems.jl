using Static, StaticArrays, SparseArrays, LinearAlgebra

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
    if det <= 1e-12 * (H11 * H22) || !isfinite(det)
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

    if det <= 1e-12 * diag_prod || !isfinite(det)
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
const MAX_ELIMINATED_DIM = 3

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
"""
struct SchurReductionWorkspace{T}
    W_scratch::Vector{Matrix{T}}          # length nchunks, each (max_Dp, max_obs_dim)
    C_inv_data::Vector{T}                 # Cached C_l(lambda)^-1: length sum_l d_l^2
    z_data::Vector{T}                     # Cached z_l = C_inv * g_p,l: length sum_l d_l
    thread_g_S::Vector{Vector{T}}         # length nchunks, each length nc
    thread_dense_S::Vector{Matrix{T}}     # length nchunks, each nc x nc (empty if sparse)
    thread_nzvals::Vector{Vector{T}}      # length nchunks, each length nnz(S) (empty if dense)
    failed_landmark::Base.Threads.Atomic{Int}
end

function SchurReductionWorkspace{T}(
    structure::SchurStructure, nchunks::Int;
    is_dense::Bool=true, nnz_S::Int=0
) where T
    max_Dp = maximum(structure.p_sizes; init=3)
    max_obs = max(structure.max_obs_dim, 1)

    W_scratch = [Matrix{T}(undef, max_Dp, max_obs) for _ in 1:nchunks]
    
    total_c_size = sum(Int(d)^2 for d in structure.p_dims; init=0)
    total_p_scalars = sum(Int(d) for d in structure.p_dims; init=0)
    C_inv_data = Vector{T}(undef, total_c_size)
    z_data = Vector{T}(undef, total_p_scalars)

    thread_g_S = [Vector{T}(undef, structure.nc) for _ in 1:nchunks]
    
    if is_dense
        thread_dense_S = [Matrix{T}(undef, structure.nc, structure.nc) for _ in 1:nchunks]
        thread_nzvals = Vector{Vector{T}}()
    else
        thread_dense_S = Vector{Matrix{T}}()
        thread_nzvals = [Vector{T}(undef, nnz_S) for _ in 1:nchunks]
    end

    return SchurReductionWorkspace{T}(
        W_scratch, C_inv_data, z_data,
        thread_g_S, thread_dense_S, thread_nzvals,
        Base.Threads.Atomic{Int}(0)
    )
end

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

@inline make_svec(arr::AbstractVector{T}, off::Int, ::StaticInt{1}) where T = SVector{1, T}(arr[off])
@inline make_svec(arr::AbstractVector{T}, off::Int, ::StaticInt{2}) where T = SVector{2, T}(arr[off], arr[off+1])
@inline make_svec(arr::AbstractVector{T}, off::Int, ::StaticInt{3}) where T = SVector{3, T}(arr[off], arr[off+1], arr[off+2])

@inline make_smat(arr::AbstractVector{T}, off::Int, ::StaticInt{1}) where T = SMatrix{1, 1, T, 1}(arr[off])
@inline make_smat(arr::AbstractVector{T}, off::Int, ::StaticInt{2}) where T = SMatrix{2, 2, T, 4}(arr[off], arr[off+1], arr[off+2], arr[off+3])
@inline make_smat(arr::AbstractVector{T}, off::Int, ::StaticInt{3}) where T = SMatrix{3, 3, T, 9}(
    arr[off], arr[off+1], arr[off+2],
    arr[off+3], arr[off+4], arr[off+5],
    arr[off+6], arr[off+7], arr[off+8]
)

@inline function dot_W_E(W_scratch::Matrix{T}, r_idx::Int, E_data::Vector{T}, e_off::Int, ::StaticInt{1}) where T
    @inbounds W_scratch[1, r_idx] * E_data[e_off]
end
@inline function dot_W_E(W_scratch::Matrix{T}, r_idx::Int, E_data::Vector{T}, e_off::Int, ::StaticInt{2}) where T
    @inbounds W_scratch[1, r_idx] * E_data[e_off] + W_scratch[2, r_idx] * E_data[e_off + 1]
end
@inline function dot_W_E(W_scratch::Matrix{T}, r_idx::Int, E_data::Vector{T}, e_off::Int, ::StaticInt{3}) where T
    @inbounds W_scratch[1, r_idx] * E_data[e_off] + W_scratch[2, r_idx] * E_data[e_off + 1] + W_scratch[3, r_idx] * E_data[e_off + 2]
end

# Threading helper. `@threads` allocates tasks on every call and has scheduling overhead, so loops run
# serially unless there is more than one thread available *and* enough work to be worth splitting.
const SCHUR_THREAD_MIN_WORK = Ref(4096)

@inline use_threads(nitems::Integer, work::Integer=nitems) = nitems > 1 && work >= SCHUR_THREAD_MIN_WORK[] && Threads.nthreads() > 1

@inline function foreach_index(f::F, n::Int, threaded::Bool) where F
    if threaded
        Threads.@threads for i in 1:n
            f(i)
        end
    else
        for i in 1:n
            f(i)
        end
    end
    return nothing
end

"""
    reduce_schur_dense_group!(S_chunk, gS_chunk, C_inv_data, z_data, W_scratch, failed_landmark,
                              structure, C_data, E_data, b_p, scaling_p, lambda, p_chunk, ::StaticInt{Dp})

Pure-data reduction kernel accumulating landmark contributions into dense upper-triangle chunk buffers.
"""
function reduce_schur_dense_group!(
    S_chunk::Matrix{T}, gS_chunk::Vector{T},
    C_inv_data::Vector{T}, z_data::Vector{T},
    W_scratch::Matrix{T}, failed_landmark::Base.Threads.Atomic{Int},
    structure::SchurStructure, C_data::Vector{T}, E_data::Vector{T},
    b_p::V, scaling_p::V, lambda::T,
    p_chunk::UnitRange{Int}, s_dp::StaticInt{Dp}
) where {T, Dp, V<:AbstractVector{T}}
    ppoffsets = structure.ppoffsets
    boffsets_p = structure.boffsets_p
    boffsets_c = structure.boffsets_c
    c_dims = structure.c_dims
    cpptr = structure.cpptr
    cpblock = structure.cpblock
    cpoffsets = structure.cpoffsets

    @inbounds for l in p_chunk
        failed_landmark[] != 0 && return false

        # 1. Invert C_l(lambda) in registers
        po = ppoffsets[l]
        H_blk = make_smat(C_data, po, s_dp)
        
        sc_off = boffsets_p[l]
        scale_blk = make_svec(scaling_p, sc_off, s_dp)
        
        res = inv_sym_block(H_blk, lambda, scale_blk)
        if !res.success
            old = failed_landmark[]
            while (old == 0 || l < old)
                old_val = Threads.atomic_cas!(failed_landmark, old, l)
                old_val == old && break
                old = failed_landmark[]
            end
            return false
        end
        C_inv = res.invH

        for i in 1:(Dp * Dp)
            C_inv_data[po + i - 1] = C_inv[i]
        end

        # 2. Compute and cache z_l = C_inv * g_p,l
        g_pl = make_svec(b_p, sc_off, s_dp)
        z_l = C_inv * g_pl
        for i in 1:Dp
            z_data[sc_off + i - 1] = z_l[i]
        end

        # 3. Process observing cameras: compute W_cl and update thread g_S
        e_start = cpptr[l]
        e_end = cpptr[l + 1] - 1
        
        row_cursor = 0
        for e in e_start:e_end
            c = cpblock[e]
            eo = cpoffsets[e]
            dc = Int(c_dims[c])
            co = boffsets_c[c]

            for r in 1:dc
                row_E = make_svec(E_data, eo + (r - 1)*Dp, s_dp)
                w_r = C_inv * row_E
                for i in 1:Dp
                    W_scratch[i, row_cursor + r] = w_r[i]
                end
                
                # Accumulate into chunk-local g_S: += E_cl * z_l
                gS_chunk[co + r - 1] += dot(row_E, z_l)
            end
            row_cursor += dc
        end

        # 4. Upper-Triangle S Accumulation (c1 <= c2 guaranteed by sorted cpblock)
        r_cursor = 0
        for e1 in e_start:e_end
            c1 = cpblock[e1]
            dc1 = Int(c_dims[c1])
            co1 = boffsets_c[c1]

            for e2 in e1:e_end
                c2 = cpblock[e2]
                dc2 = Int(c_dims[c2])
                co2 = boffsets_c[c2]

                if c1 == c2
                    # DIAGONAL BLOCK: strictly upper triangular (row <= col)
                    eo = cpoffsets[e1]
                    for col in 1:dc2
                        e_off = eo + (col - 1)*Dp
                        for row in 1:col
                            prod = dot_W_E(W_scratch, r_cursor + row, E_data, e_off, s_dp)
                            S_chunk[co1 + row - 1, co1 + col - 1] -= prod
                        end
                    end
                else
                    # OFF-DIAGONAL BLOCK: c1 < c2 guaranteed
                    eo2 = cpoffsets[e2]
                    for col in 1:dc2
                        e_off = eo2 + (col - 1)*Dp
                        for row in 1:dc1
                            prod = dot_W_E(W_scratch, r_cursor + row, E_data, e_off, s_dp)
                            S_chunk[co1 + row - 1, co2 + col - 1] -= prod
                        end
                    end
                end
            end
            r_cursor += dc1
        end
    end
    return true
end

"""
    reduce_schur_sparse_group!(nzval_chunk, gS_chunk, C_inv_data, z_data, W_scratch, failed_landmark,
                               structure, C_data, E_data, S_colptr, block_colptr, block_rowval,
                               block_row_offsets, col_heights, b_p, scaling_p, lambda, p_chunk, ::StaticInt{Dp})

Pure-data reduction kernel accumulating landmark contributions into sparse CSC chunk buffers via a monotone merge-walk.
"""
function reduce_schur_sparse_group!(
    nzval_chunk::Vector{T}, gS_chunk::Vector{T},
    C_inv_data::Vector{T}, z_data::Vector{T},
    W_scratch::Matrix{T}, failed_landmark::Base.Threads.Atomic{Int},
    structure::SchurStructure, C_data::Vector{T}, E_data::Vector{T},
    S_colptr::Vector{Ti}, block_colptr::Vector{Ti}, block_rowval::Vector{Ti},
    block_row_offsets::Vector{Int}, col_heights::Vector{Int},
    b_p::V, scaling_p::V, lambda::T,
    p_chunk::UnitRange{Int}, s_dp::StaticInt{Dp}
) where {T, Ti, Dp, V<:AbstractVector{T}}
    ppoffsets = structure.ppoffsets
    boffsets_p = structure.boffsets_p
    boffsets_c = structure.boffsets_c
    c_dims = structure.c_dims
    cpptr = structure.cpptr
    cpblock = structure.cpblock
    cpoffsets = structure.cpoffsets

    @inbounds for l in p_chunk
        failed_landmark[] != 0 && return false

        # 1. Invert C_l(lambda)
        po = ppoffsets[l]
        H_blk = make_smat(C_data, po, s_dp)
        sc_off = boffsets_p[l]
        scale_blk = make_svec(scaling_p, sc_off, s_dp)
        
        res = inv_sym_block(H_blk, lambda, scale_blk)
        if !res.success
            old = failed_landmark[]
            while (old == 0 || l < old)
                old_val = Threads.atomic_cas!(failed_landmark, old, l)
                old_val == old && break
                old = failed_landmark[]
            end
            return false
        end
        C_inv = res.invH
        for i in 1:(Dp * Dp); C_inv_data[po + i - 1] = C_inv[i]; end

        # 2. Compute z_l
        g_pl = make_svec(b_p, sc_off, s_dp)
        z_l = C_inv * g_pl
        for i in 1:Dp; z_data[sc_off + i - 1] = z_l[i]; end

        # 3. Compute W_cl and update g_S
        e_start = cpptr[l]
        e_end = cpptr[l + 1] - 1
        row_cursor = 0
        for e in e_start:e_end
            c = cpblock[e]
            eo = cpoffsets[e]
            dc = Int(c_dims[c])
            co = boffsets_c[c]
            for r in 1:dc
                row_E = make_svec(E_data, eo + (r - 1)*Dp, s_dp)
                w_r = C_inv * row_E
                for i in 1:Dp; W_scratch[i, row_cursor + r] = w_r[i]; end
                gS_chunk[co + r - 1] += dot(row_E, z_l)
            end
            row_cursor += dc
        end

        # 4. Monotone Merge-Walk into Sparse S
        for e2 in e_start:e_end
            c2 = cpblock[e2]
            dc2 = Int(c_dims[c2])
            eo2 = cpoffsets[e2]
            scol_c2 = boffsets_c[c2]
            col_base = S_colptr[scol_c2]
            stride = col_heights[c2]

            b_row_idx = block_colptr[c2]
            b_row_end = block_colptr[c2 + 1] - 1

            r_cursor = 0
            for e1 in e_start:e2
                c1 = cpblock[e1]
                dc1 = Int(c_dims[c1])

                while b_row_idx <= b_row_end && block_rowval[b_row_idx] < c1
                    b_row_idx += 1
                end
                @assert b_row_idx <= b_row_end && block_rowval[b_row_idx] == c1 "Sparse pattern missing entry ($c1, $c2)"

                row_offset = block_row_offsets[b_row_idx]

                if c1 == c2
                    # Diagonal block write: strictly upper triangular (row <= col)
                    for col in 1:dc2
                        e_off = eo2 + (col - 1)*Dp
                        col_ptr = col_base + (col - 1) * stride + row_offset
                        for row in 1:col
                            prod = dot_W_E(W_scratch, r_cursor + row, E_data, e_off, s_dp)
                            nzval_chunk[col_ptr + row - 1] -= prod
                        end
                    end
                else
                    # Off-diagonal block write
                    for col in 1:dc2
                        e_off = eo2 + (col - 1)*Dp
                        col_ptr = col_base + (col - 1) * stride + row_offset
                        for row in 1:dc1
                            prod = dot_W_E(W_scratch, r_cursor + row, E_data, e_off, s_dp)
                            nzval_chunk[col_ptr + row - 1] -= prod
                        end
                    end
                end
                r_cursor += dc1
            end
        end
    end
    return true
end

"""
    reset_schur_dense!(S, B, scaling_c, lambda, ws)

Reset dense reduced system S = B + lambda * diag(scaling_c) and zero chunk accumulators.
"""
function reset_schur_dense!(
    S::Matrix{T}, B::Matrix{T}, scaling_c::AbstractVector{T},
    lambda::T, ws::SchurReductionWorkspace{T}
) where T
    copyto!(S, B)
    @inbounds for (i, sc) in enumerate(scaling_c)
        S[i, i] += lambda * sc
    end
    for k in eachindex(ws.W_scratch)
        fill!(ws.thread_g_S[k], zero(T))
        fill!(ws.thread_dense_S[k], zero(T))
    end
    ws.failed_landmark[] = 0
    return nothing
end

"""
    reset_schur_sparse!(nzval, B_nzval, b_to_s_map, diag_inds, scaling_c, lambda, ws)

Reset sparse reduced system S = B + lambda * diag(scaling_c) and zero chunk accumulators.
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
    for k in eachindex(ws.W_scratch)
        fill!(ws.thread_g_S[k], zero(T))
        fill!(ws.thread_nzvals[k], zero(T))
    end
    ws.failed_landmark[] = 0
    return nothing
end

"""
    reduce_schur_dense!(S, g_S, b_c, b_p, scaling_p, lambda, structure, C_data, E_data, ws)

Pure-data driver executing parallel dense Schur reduction across chunks and merging into S and g_S.
"""
function reduce_schur_dense!(
    S::Matrix{T}, g_S::Vector{T}, b_c::AbstractVector{T}, b_p::AbstractVector{T},
    scaling_p::AbstractVector{T}, lambda::T, structure::SchurStructure,
    C_data::Vector{T}, E_data::Vector{T}, ws::SchurReductionWorkspace{T}
) where T
    nchunks = length(ws.W_scratch)
    threaded = use_threads(nchunks, length(structure.cpblock))

    for (g_idx, p_range) in enumerate(structure.p_groups)
        dp = structure.p_sizes[g_idx]
        chunks = structure.chunks[g_idx]

        # Function barrier outside thread loop
        if dp == 3
            foreach_index(nchunks, threaded) do k
                reduce_schur_dense_group!(ws.thread_dense_S[k], ws.thread_g_S[k], ws.C_inv_data, ws.z_data, ws.W_scratch[k], ws.failed_landmark, structure, C_data, E_data, b_p, scaling_p, lambda, chunks[k], static(3))
            end
        elseif dp == 1
            foreach_index(nchunks, threaded) do k
                reduce_schur_dense_group!(ws.thread_dense_S[k], ws.thread_g_S[k], ws.C_inv_data, ws.z_data, ws.W_scratch[k], ws.failed_landmark, structure, C_data, E_data, b_p, scaling_p, lambda, chunks[k], static(1))
            end
        elseif dp == 2
            foreach_index(nchunks, threaded) do k
                reduce_schur_dense_group!(ws.thread_dense_S[k], ws.thread_g_S[k], ws.C_inv_data, ws.z_data, ws.W_scratch[k], ws.failed_landmark, structure, C_data, E_data, b_p, scaling_p, lambda, chunks[k], static(2))
            end
        else
            error("Unsupported landmark dimension $dp")
        end
        ws.failed_landmark[] != 0 && return false
    end

    merge_dense_S!(S, g_S, b_c, ws.thread_dense_S, ws.thread_g_S, structure.nc)
    return true
end

function merge_dense_S!(
    S::Matrix{T}, g_S::Vector{T}, b_c::AbstractVector{T},
    thread_dense_S::Vector{Matrix{T}}, thread_g_S::Vector{Vector{T}}, nc::Int
) where T
    copyto!(g_S, b_c)
    nchunks = length(thread_g_S)

    # Column-wise over the upper triangle, in parallel only when there is enough work
    foreach_index(nc, use_threads(nc, nc * nc * nchunks)) do j
        for k in 1:nchunks
            @inbounds g_S[j] -= thread_g_S[k][j]
        end
        for i in 1:j
            val = zero(T)
            for k in 1:nchunks
                @inbounds val += thread_dense_S[k][i, j]
            end
            @inbounds S[i, j] += val
        end
    end
    return nothing
end

"""
    reduce_schur_sparse!(S, g_S, b_c, b_p, scaling_p, lambda, structure, C_data, E_data,
                         block_colptr, block_rowval, block_row_offsets, col_heights, ws)

Pure-data driver executing parallel sparse Schur reduction across chunks and merging into S.nzval and g_S.
"""
function reduce_schur_sparse!(
    S::SparseMatrixCSC{T, Ti}, g_S::Vector{T}, b_c::AbstractVector{T}, b_p::AbstractVector{T},
    scaling_p::AbstractVector{T}, lambda::T, structure::SchurStructure,
    C_data::Vector{T}, E_data::Vector{T},
    block_colptr::Vector{Ti}, block_rowval::Vector{Ti},
    block_row_offsets::Vector{Int}, col_heights::Vector{Int},
    ws::SchurReductionWorkspace{T}
) where {T, Ti}
    nchunks = length(ws.W_scratch)
    threaded = use_threads(nchunks, length(structure.cpblock))

    for (g_idx, p_range) in enumerate(structure.p_groups)
        dp = structure.p_sizes[g_idx]
        chunks = structure.chunks[g_idx]

        if dp == 3
            foreach_index(nchunks, threaded) do k
                reduce_schur_sparse_group!(ws.thread_nzvals[k], ws.thread_g_S[k], ws.C_inv_data, ws.z_data, ws.W_scratch[k], ws.failed_landmark, structure, C_data, E_data, S.colptr, block_colptr, block_rowval, block_row_offsets, col_heights, b_p, scaling_p, lambda, chunks[k], static(3))
            end
        elseif dp == 1
            foreach_index(nchunks, threaded) do k
                reduce_schur_sparse_group!(ws.thread_nzvals[k], ws.thread_g_S[k], ws.C_inv_data, ws.z_data, ws.W_scratch[k], ws.failed_landmark, structure, C_data, E_data, S.colptr, block_colptr, block_rowval, block_row_offsets, col_heights, b_p, scaling_p, lambda, chunks[k], static(1))
            end
        elseif dp == 2
            foreach_index(nchunks, threaded) do k
                reduce_schur_sparse_group!(ws.thread_nzvals[k], ws.thread_g_S[k], ws.C_inv_data, ws.z_data, ws.W_scratch[k], ws.failed_landmark, structure, C_data, E_data, S.colptr, block_colptr, block_rowval, block_row_offsets, col_heights, b_p, scaling_p, lambda, chunks[k], static(2))
            end
        else
            error("Unsupported landmark dimension $dp")
        end
        ws.failed_landmark[] != 0 && return false
    end

    merge_sparse_S!(S.nzval, g_S, b_c, ws.thread_nzvals, ws.thread_g_S, structure.nc)
    return true
end

function merge_sparse_S!(
    nzval::Vector{T}, g_S::Vector{T}, b_c::AbstractVector{T},
    thread_nzvals::Vector{Vector{T}}, thread_g_S::Vector{Vector{T}}, nc::Int
) where T
    copyto!(g_S, b_c)
    nchunks = length(thread_g_S)

    foreach_index(nc, use_threads(nc, nc * nchunks)) do j
        for k in 1:nchunks
            @inbounds g_S[j] -= thread_g_S[k][j]
        end
    end
    foreach_index(length(nzval), use_threads(length(nzval), length(nzval) * nchunks)) do i
        val = zero(T)
        for k in 1:nchunks
            @inbounds val += thread_nzvals[k][i]
        end
        @inbounds nzval[i] += val
    end
    return nothing
end

"""
    backsubstitute!(x_p, x_c, structure, E_data, C_inv_data, z_data)

Recover step Delta x_p for all eliminable blocks independently using cached C_inv and z.
"""
function backsubstitute!(
    x_p::AbstractVector{T}, x_c::AbstractVector{T},
    structure::SchurStructure, E_data::Vector{T},
    C_inv_data::Vector{T}, z_data::Vector{T}
) where T
    cpptr = structure.cpptr
    cpblock = structure.cpblock
    cpoffsets = structure.cpoffsets
    boffsets_c = structure.boffsets_c
    boffsets_p = structure.boffsets_p
    p_dims = structure.p_dims
    c_dims = structure.c_dims

    foreach_index(structure.npblocks, use_threads(structure.npblocks, length(structure.cpblock))) do l
        dp = Int(p_dims[l])
        p_off = boffsets_p[l]
        p_data_off = structure.ppoffsets[l]

        for i in 1:dp
            x_p[p_off + i - 1] = z_data[p_off + i - 1]
        end

        e_start = cpptr[l]
        e_end = cpptr[l + 1] - 1
        
        for e in e_start:e_end
            c = cpblock[e]
            dc = Int(c_dims[c])
            co = boffsets_c[c]
            eo = cpoffsets[e]

            for r in 1:dc
                xc_val = x_c[co + r - 1]
                for i in 1:dp
                    e_val = E_data[eo + (r - 1)*dp + i - 1]
                    for j in 1:dp
                        inv_val = C_inv_data[p_data_off + (j - 1)*dp + i - 1]
                        x_p[p_off + j - 1] -= inv_val * e_val * xc_val
                    end
                end
            end
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
    build_schur_covisibility(structure::SchurStructure; extra_blocks=()) -> NamedTuple

Compute block-level and scalar-level co-visibility sparsity patterns for the reduced camera system S.
Returns `(block_covis, block_colptr, block_rowval, block_row_offsets, col_heights, S_sparse, diag_inds)`.

The pattern contains every pair of retained blocks that share an eliminated block, plus the diagonal.
Retained-retained couplings that do not go through an eliminated block (e.g. odometry factors or loop
closures between poses with no common landmark) have no such slot, so list them in `extra_blocks`, an
iterable of `(i, j)` retained block index pairs (in either order; `i == j` is allowed and ignored).
Use [`retained_block_pairs`](@ref) to obtain them from the block pattern of `B`.
"""
function build_schur_covisibility(structure::SchurStructure; extra_blocks=())
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
    S_sparse = sparse(I_list, J_list, zeros(length(I_list)), nc, nc)

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
            s_k = 0
            for idx in S_sparse.colptr[j]:(S_sparse.colptr[j + 1] - 1)
                if S_sparse.rowval[idx] == i
                    s_k = idx
                    break
                end
            end
            @assert s_k != 0 "B entry ($i, $j) not present in S sparsity pattern"
            push!(b_to_s, s_k)
        end
    end
    return b_to_s
end

"""
    build_b_to_s_map(B_sys::SparseBlockLinearSystem, S_sparse::SparseMatrixCSC, boffsets_c::Vector{Int}) -> Vector{Int}

Build mapping vector from B_sys.data to S_sparse.nzval for symmetric upper-triangle reduction.
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
                    s_k = 0
                    for idx in S_sparse.colptr[scol]:(S_sparse.colptr[scol + 1] - 1)
                        if S_sparse.rowval[idx] == srow
                            s_k = idx
                            break
                        end
                    end
                    @assert s_k != 0 "Missing diagonal entry ($srow, $scol) in S"
                    b_to_s[data_start + (col - 1)*dr + row - 1] = s_k
                end
            end
        elseif r > c
            for col in 1:dc
                srow = boffsets_c[c] + col - 1
                for row in 1:dr
                    scol = boffsets_c[r] + row - 1
                    s_k = 0
                    for idx in S_sparse.colptr[scol]:(S_sparse.colptr[scol + 1] - 1)
                        if S_sparse.rowval[idx] == srow
                            s_k = idx
                            break
                        end
                    end
                    @assert s_k != 0 "Missing entry ($srow, $scol) in S"
                    b_to_s[data_start + (col - 1)*dr + row - 1] = s_k
                end
            end
        end
    end
    return b_to_s
end

"""
    fast_bAb(H::SchurBlockMatrix, dx::AbstractVector) -> Real

Compute quadratic form dx' * H * dx in O(N) operations without materializing the full Hessian.
"""
function fast_bAb(H::SchurBlockMatrix{T}, dx::AbstractVector{T}) where T
    structure = H.structure
    nc = structure.nc
    dx_c = view(dx, 1:nc)
    dx_p = view(dx, nc+1:length(dx))
    
    # 1. dx_c' * B * dx_c
    total = zero(T)
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

    # 2. 2 * dx_c' * E * dx_p
    cpptr = structure.cpptr
    cpblock = structure.cpblock
    cpoffsets = structure.cpoffsets
    boffsets_c = structure.boffsets_c
    boffsets_p = structure.boffsets_p
    p_dims = structure.p_dims
    c_dims = structure.c_dims

    for l in 1:structure.npblocks
        dp = Int(p_dims[l])
        p_off = boffsets_p[l]
        e_start = cpptr[l]
        e_end = cpptr[l + 1] - 1
        for e in e_start:e_end
            c = cpblock[e]
            dc = Int(c_dims[c])
            co = boffsets_c[c]
            eo = cpoffsets[e]
            for col in 1:dp
                dx_p_val = dx_p[p_off + col - 1]
                for row in 1:dc
                    e_val = H.E_data[eo + (row - 1)*dp + col - 1]
                    dx_c_val = dx_c[co + row - 1]
                    total += 2 * e_val * dx_c_val * dx_p_val
                end
            end
        end
    end

    # 3. dx_p' * C * dx_p
    ppoffsets = structure.ppoffsets
    for l in 1:structure.npblocks
        dp = Int(p_dims[l])
        p_off = boffsets_p[l]
        po = ppoffsets[l]
        for col in 1:dp
            dx_p_col = dx_p[p_off + col - 1]
            for row in 1:dp
                c_val = H.C_data[po + (col - 1)*dp + row - 1]
                dx_p_row = dx_p[p_off + row - 1]
                total += c_val * dx_p_col * dx_p_row
            end
        end
    end

    return total
end
