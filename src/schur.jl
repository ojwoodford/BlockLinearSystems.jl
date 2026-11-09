using Static, StaticArrays, SparseArrays, LinearAlgebra

"""
    inv_sym_block(H::SMatrix{P, P, T}, lambda::T, scale::SVector{P, T}) where {P, T}

Invert a symmetric positive semi-definite block `H` with diagonal Levenberg-Marquardt damping
`H + lambda * diag(scale)^2` in registers.
Returns `(invH, success)` where `success` is false if the block is numerically rank-deficient.
"""
@inline function inv_sym_block(H::SMatrix{1, 1, T, 1}, lambda::T, scale::SVector{1, T}) where T
    h11 = H[1, 1] + lambda * scale[1]^2
    if h11 <= eps(T) * abs(H[1, 1]) || !isfinite(h11)
        return (invH = zeros(SMatrix{1, 1, T, 1}), success = false)
    end
    return (invH = SMatrix{1, 1, T, 1}((inv(h11),)), success = true)
end

@inline function inv_sym_block(H::SMatrix{2, 2, T, 4}, lambda::T, scale::SVector{2, T}) where T
    H11 = H[1, 1] + lambda * scale[1]^2
    H22 = H[2, 2] + lambda * scale[2]^2
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
    H11 = H[1, 1] + lambda * scale[1]^2
    H22 = H[2, 2] + lambda * scale[2]^2
    H33 = H[3, 3] + lambda * scale[3]^2
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

Reset dense reduced system S = B + lambda * diag(scaling_c)^2 and zero chunk accumulators.
"""
function reset_schur_dense!(
    S::Matrix{T}, B::Matrix{T}, scaling_c::AbstractVector{T},
    lambda::T, ws::SchurReductionWorkspace{T}
) where T
    copyto!(S, B)
    @inbounds for (i, sc) in enumerate(scaling_c)
        S[i, i] += lambda * sc^2
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

Reset sparse reduced system S = B + lambda * diag(scaling_c)^2 and zero chunk accumulators.
"""
function reset_schur_sparse!(
    nzval::Vector{T}, B_nzval::Vector{T}, b_to_s_map::Vector{Int},
    diag_inds::Vector{Int}, scaling_c::AbstractVector{T},
    lambda::T, ws::SchurReductionWorkspace{T}
) where T
    fill!(nzval, zero(T))
    @inbounds for i in eachindex(b_to_s_map)
        nzval[b_to_s_map[i]] = B_nzval[i]
    end
    @inbounds for (i, idx) in enumerate(diag_inds)
        nzval[idx] += lambda * scaling_c[i]^2
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

    for (g_idx, p_range) in enumerate(structure.p_groups)
        dp = structure.p_sizes[g_idx]
        chunks = structure.chunks[g_idx]

        # Function barrier outside thread loop
        if dp == 3
            Threads.@threads for k in 1:nchunks
                reduce_schur_dense_group!(ws.thread_dense_S[k], ws.thread_g_S[k], ws.C_inv_data, ws.z_data, ws.W_scratch[k], ws.failed_landmark, structure, C_data, E_data, b_p, scaling_p, lambda, chunks[k], static(3))
            end
        elseif dp == 1
            Threads.@threads for k in 1:nchunks
                reduce_schur_dense_group!(ws.thread_dense_S[k], ws.thread_g_S[k], ws.C_inv_data, ws.z_data, ws.W_scratch[k], ws.failed_landmark, structure, C_data, E_data, b_p, scaling_p, lambda, chunks[k], static(1))
            end
        elseif dp == 2
            Threads.@threads for k in 1:nchunks
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

    # Parallelize column-wise over upper triangle
    Threads.@threads for j in 1:nc
        for k in 1:nchunks
            g_S[j] -= thread_g_S[k][j]
        end
        for i in 1:j
            val = zero(T)
            for k in 1:nchunks
                val += thread_dense_S[k][i, j]
            end
            S[i, j] += val
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

    for (g_idx, p_range) in enumerate(structure.p_groups)
        dp = structure.p_sizes[g_idx]
        chunks = structure.chunks[g_idx]

        if dp == 3
            Threads.@threads for k in 1:nchunks
                reduce_schur_sparse_group!(ws.thread_nzvals[k], ws.thread_g_S[k], ws.C_inv_data, ws.z_data, ws.W_scratch[k], ws.failed_landmark, structure, C_data, E_data, S.colptr, block_colptr, block_rowval, block_row_offsets, col_heights, b_p, scaling_p, lambda, chunks[k], static(3))
            end
        elseif dp == 1
            Threads.@threads for k in 1:nchunks
                reduce_schur_sparse_group!(ws.thread_nzvals[k], ws.thread_g_S[k], ws.C_inv_data, ws.z_data, ws.W_scratch[k], ws.failed_landmark, structure, C_data, E_data, S.colptr, block_colptr, block_rowval, block_row_offsets, col_heights, b_p, scaling_p, lambda, chunks[k], static(1))
            end
        elseif dp == 2
            Threads.@threads for k in 1:nchunks
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

    Threads.@threads for j in 1:nc
        for k in 1:nchunks
            g_S[j] -= thread_g_S[k][j]
        end
    end
    Threads.@threads for i in eachindex(nzval)
        val = zero(T)
        for k in 1:nchunks
            val += thread_nzvals[k][i]
        end
        nzval[i] += val
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

    Threads.@threads for l in 1:structure.npblocks
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
