"""
    SchurReductionWorkspace{T}

Pre-allocated scratch memory and caches for parallel Schur reduction and back-substitution.

The reduction runs in two phases. Phase 1 (parallel over chunks of eliminated blocks) factors every
`E_e`, caching `E_inv`, `z_e` and `W_er = E_inv * C_er` (`W_data`, laid out like `C_data`). Phase 2
(parallel over `col_chunks`, disjoint ranges of retained blocks) has each chunk write straight into the
columns of `S` (and entries of `g_S`) that it owns, so no per-chunk copies of `S` are needed and nothing
has to be merged afterwards.

Memory: `E_inv` takes `d_e²` values per eliminated block, and when any eliminated block is above
`BlockLinearSystems.D_STATIC` each chunk also holds a `d_max × d_max` scratch matrix (`d_max` the largest
eliminated dimension). Eliminated blocks up to 65535 are supported, but large ones cost memory accordingly.
"""
struct SchurReductionWorkspace{T}
    W_scratch::Vector{Matrix{T}}          # per chunk: max_De x max_De scratch for blocks above D_STATIC (0 x 0 otherwise), for phase 1 and back-substitution
    W_data::Vector{T}                     # W_er = E_e^-1 C_er for every coupling, same layout as C_data
    E_inv_data::Vector{T}                 # Cached E_e(lambda)^-1: length sum_e d_e^2
    z_data::Vector{T}                     # Cached z_e = E_e(lambda)^-1 b_e: length sum_e d_e
    retained_ptr::Vector{Int}                  # Retained-block-major CSR pointer into retained_entry
    retained_entry::Vector{Int}                # Coupling entries (indices into entry_retained) of each retained block, ascending
    entry_eliminated::Vector{Int}           # Eliminated block of each coupling entry
    col_chunks::Vector{UnitRange{Int}}    # Disjoint ranges of retained blocks, one per phase-2 chunk
    failed_block::Base.Threads.Atomic{Int}
end

"Split the retained blocks `1:n` into `K` contiguous ranges of roughly equal work (couplings plus the diagonal)."
function partition_columns(retained_ptr::Vector{Int}, K::Int)
    n = length(retained_ptr) - 1
    total = retained_ptr[end] - retained_ptr[1] + n
    ranges = UnitRange{Int}[]
    start = 1
    for k in 1:K
        stop = start - 1
        if k == K
            stop = n
        else
            target = (total * k) / K
            while stop < n && (retained_ptr[stop + 2] - retained_ptr[1] + (stop + 1)) <= target
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
    max_De = max_run_blocksize(structure.e_dims; init=3)

    # One scratch matrix per phase-1 chunk, only used for blocks above D_STATIC
    nscratch = max(nchunks, maximum(length, structure.chunks; init=1))
    W_scratch = [Matrix{T}(undef, max_De > D_STATIC ? max_De : 0, max_De > D_STATIC ? max_De : 0) for _ in 1:nscratch]

    total_r_size = sum(Int(d)^2 for d in structure.e_dims; init=0)
    total_e_scalars = sum(Int(d) for d in structure.e_dims; init=0)
    E_inv_data = Vector{T}(undef, total_r_size)
    z_data = Vector{T}(undef, total_e_scalars)

    ncoup = length(structure.entry_retained)
    total_c = 0
    for e in 1:structure.neblocks, c in structure.eliminated_ptr[e]:(structure.eliminated_ptr[e + 1] - 1)
        total_c += Int(structure.r_dims[structure.entry_retained[c]]) * Int(structure.e_dims[e])
    end
    W_data = Vector{T}(undef, total_c)

    # Retained-block-major index of the couplings
    entry_eliminated = Vector{Int}(undef, ncoup)
    counts = zeros(Int, structure.nrblocks + 1)
    for e in 1:structure.neblocks, c in structure.eliminated_ptr[e]:(structure.eliminated_ptr[e + 1] - 1)
        entry_eliminated[c] = e
        counts[structure.entry_retained[c] + 1] += 1
    end
    retained_ptr = cumsum(vcat(1, counts[2:end]))
    retained_entry = Vector{Int}(undef, ncoup)
    fill_pos = copy(retained_ptr)
    for c in 1:ncoup   # ascending c means ascending eliminated block within each retained block
        r = structure.entry_retained[c]
        retained_entry[fill_pos[r]] = c
        fill_pos[r] += 1
    end

    return SchurReductionWorkspace{T}(
        W_scratch, W_data, E_inv_data, z_data,
        retained_ptr, retained_entry, entry_eliminated, partition_columns(retained_ptr, max(nchunks, 1)),
        Base.Threads.Atomic{Int}(0)
    )
end

"""
    trial_workspace(ws)

A workspace for a second linear system with the same structure: own `E_inv`/`z` caches and failure flag;
the scratch (`W_data`, `W_scratch`) is only live within one solve and is shared.
"""
trial_workspace(ws::SchurReductionWorkspace{T}) where T = SchurReductionWorkspace{T}(
    ws.W_scratch, ws.W_data, similar(ws.E_inv_data), similar(ws.z_data),
    ws.retained_ptr, ws.retained_entry, ws.entry_eliminated, ws.col_chunks, Base.Threads.Atomic{Int}(0)
)

"""
    factor_eliminated!(E_inv_data, z_data, W_data, scratch, structure, E_data, C_data, b_e, scaling_e, lambda, e, s_de)

Phase 1 for eliminated block `e`: invert `E_e + lambda * diag(scaling)`, cache `E_inv` and
`z_e = E_inv * b_e`, and store `W_er = E_inv * C_er` for every coupled retained block in `W_data`.
Returns `false` if the damped block is numerically singular or not positive definite.

`s_de` is a `StaticInt` for blocks of up to `D_STATIC` dimensions (register arithmetic), otherwise the
`Int` dimension, which uses a LAPACK Cholesky on the `scratch` matrix (O(d^3) per eliminated block plus
O(d^2) per coupling row; no allocation).
"""
@inline function factor_eliminated!(
    E_inv_data::Vector{T}, z_data::Vector{T}, W_data::Vector{T}, ::Matrix{T},
    structure::SchurStructure, E_data::Vector{T}, C_data::Vector{T},
    b_e::AbstractVector{T}, scaling_e::AbstractVector{T}, lambda::T, e::Int, s_de::StaticInt{De}
) where {T, De}
    eo = structure.eoffsets[e]
    sc_off = structure.boffsets_e[e]

    res = inv_sym_block(make_smat(E_data, eo, s_de), lambda, make_svec(scaling_e, sc_off, s_de))
    res.success || return false
    E_inv = res.invH
    @inbounds for i in 1:(De * De)
        E_inv_data[eo + i - 1] = E_inv[i]
    end

    z_e = E_inv * make_svec(b_e, sc_off, s_de)
    @inbounds for i in 1:De
        z_data[sc_off + i - 1] = z_e[i]
    end

    @inbounds for c in structure.eliminated_ptr[e]:(structure.eliminated_ptr[e + 1] - 1)
        co = structure.coffsets[c]
        dr = Int(structure.r_dims[structure.entry_retained[c]])
        for ir in 1:dr
            w_ir = E_inv * make_svec(C_data, co + (ir - 1) * De, s_de)
            for i in 1:De
                W_data[co + (ir - 1) * De + i - 1] = w_ir[i]
            end
        end
    end
    return true
end

@inline function factor_eliminated!(
    E_inv_data::Vector{T}, z_data::Vector{T}, W_data::Vector{T}, scratch::Matrix{T},
    structure::SchurStructure, E_data::Vector{T}, C_data::Vector{T},
    b_e::AbstractVector{T}, scaling_e::AbstractVector{T}, lambda::T, e::Int, de::Int
) where T
    eo = structure.eoffsets[e]
    sc_off = structure.boffsets_e[e]

    M = view(scratch, 1:de, 1:de)
    @inbounds for j in 1:de, i in 1:de
        M[i, j] = E_data[eo + (j - 1) * de + i - 1]
    end
    @inbounds for i in 1:de
        M[i, i] += lambda * scaling_e[sc_off + i - 1]
    end
    # Cholesky; every pivot must be safely positive relative to its (damped) diagonal entry. The damped
    # diagonal entry is recomputed from the inputs.
    _, info = LinearAlgebra.LAPACK.potrf!('U', M)
    info == 0 || return false
    @inbounds for i in 1:de
        hii = E_data[eo + (i - 1) * (de + 1)] + lambda * scaling_e[sc_off + i - 1]
        ratio = abs2(M[i, i]) / hii
        (isfinite(ratio) && ratio > singular_tol(T)) || return false
    end
    LinearAlgebra.LAPACK.potri!('U', M)

    @inbounds for j in 1:de, i in 1:j
        v = M[i, j]
        E_inv_data[eo + (j - 1) * de + i - 1] = v
        E_inv_data[eo + (i - 1) * de + j - 1] = v
    end

    @inbounds for i in 1:de
        acc = zero(T)
        for j in 1:de
            acc += E_inv_data[eo + (j - 1) * de + i - 1] * b_e[sc_off + j - 1]
        end
        z_data[sc_off + i - 1] = acc
    end

    @inbounds for c in structure.eliminated_ptr[e]:(structure.eliminated_ptr[e + 1] - 1)
        co = structure.coffsets[c]
        dr = Int(structure.r_dims[structure.entry_retained[c]])
        for ir in 1:dr
            row_off = co + (ir - 1) * de
            for i in 1:de
                acc = zero(T)
                for j in 1:de
                    acc += E_inv_data[eo + (j - 1) * de + i - 1] * C_data[row_off + j - 1]
                end
                W_data[row_off + i - 1] = acc
            end
        end
    end
    return true
end

"""
    factor_eliminated_group!(ws..., e_chunk, s_de) -> Bool

Phase 1 over a chunk of eliminated blocks of uniform dimension. On failure the lowest failing
eliminated block index is recorded in `failed_block`.
"""
function factor_eliminated_group!(
    E_inv_data::Vector{T}, z_data::Vector{T}, W_data::Vector{T}, scratch::Matrix{T},
    failed_block::Base.Threads.Atomic{Int},
    structure::SchurStructure, E_data::Vector{T}, C_data::Vector{T},
    b_e::AbstractVector{T}, scaling_e::AbstractVector{T}, lambda::T,
    e_chunk::UnitRange{Int}, s_de::Union{StaticInt, Int}
) where T
    @inbounds for e in e_chunk
        fb = failed_block[]                  # a lower failure elsewhere: no block above it can be the lowest
        fb != 0 && e > fb && return false
        if !factor_eliminated!(E_inv_data, z_data, W_data, scratch, structure, E_data, C_data, b_e, scaling_e, lambda, e, s_de)
            old = failed_block[]
            while (old == 0 || e < old)
                old_val = Threads.atomic_cas!(failed_block, old, e)
                old_val == old && break
                # Retry only when another thread changed failed_block between the read and the CAS: a race that no
                # deterministic test can force, so these two lines are excluded from coverage.
                old = failed_block[]  # COV_EXCL_LINE
            end  # COV_EXCL_LINE
            return false
        end
    end
    return true
end

# Contribution of coupling entry c2 (eliminated block e, retained block r2) to g_S and to the column
# block r2 of S: g_S[r2] -= C_er2' z_e and S[r1, r2] -= W_er1' C_er2 for every r1 <= r2 coupled to e.
@inline function accumulate_entry_dense!(
    S::Matrix{T}, g_S::Vector{T}, structure::SchurStructure, C_data::Vector{T}, W_data::Vector{T}, z_data::Vector{T},
    c2::Int, e::Int, s_de::Union{StaticInt, Int}
) where T
    de = dimval(s_de)
    r2 = structure.entry_retained[c2]
    dr2 = Int(structure.r_dims[r2])
    ro2 = structure.boffsets_r[r2]
    co2 = structure.coffsets[c2]
    sc_off = structure.boffsets_e[e]
    @inbounds for ir in 1:dr2
        g_S[ro2 + ir - 1] -= dot_dim(C_data, co2 + (ir - 1) * de, z_data, sc_off, s_de)
    end
    @inbounds for c1 in structure.eliminated_ptr[e]:c2
        r1 = structure.entry_retained[c1]
        dr1 = Int(structure.r_dims[r1])
        ro1 = structure.boffsets_r[r1]
        wo1 = structure.coffsets[c1]
        for col in 1:dr2
            c_off = co2 + (col - 1) * de
            nrow = r1 == r2 ? col : dr1      # diagonal block: upper triangle only
            for row in 1:nrow
                S[ro1 + row - 1, ro2 + col - 1] -= dot_dim(W_data, wo1 + (row - 1) * de, C_data, c_off, s_de)
            end
        end
    end
    return nothing
end

"""
    accumulate_dense_columns!(S, g_S, structure, C_data, W_data, z_data, retained_ptr, retained_entry, entry_eliminated, r_range)

Phase 2 for the retained blocks in `r_range`: write all contributions to their columns of the dense upper
triangle of `S` and to `g_S` directly. Chunks with disjoint `r_range` never write the same entry.
"""
function accumulate_dense_columns!(
    S::Matrix{T}, g_S::Vector{T}, structure::SchurStructure, C_data::Vector{T}, W_data::Vector{T}, z_data::Vector{T},
    retained_ptr::Vector{Int}, retained_entry::Vector{Int}, entry_eliminated::Vector{Int}, r_range::UnitRange{Int}
) where T
    @inbounds for r2 in r_range
        for k in retained_ptr[r2]:(retained_ptr[r2 + 1] - 1)
            c2 = retained_entry[k]
            e = entry_eliminated[c2]
            dispatch_dim(Int(structure.e_dims[e])) do s_de
                accumulate_entry_dense!(S, g_S, structure, C_data, W_data, z_data, c2, e, s_de)
            end
        end
    end
    return nothing
end

@inline function accumulate_entry_sparse!(
    nzval::Vector{T}, g_S::Vector{T}, structure::SchurStructure, C_data::Vector{T}, W_data::Vector{T}, z_data::Vector{T},
    S_colptr::Vector{Ti}, block_colptr::Vector{Ti}, block_rowval::Vector{Ti}, block_row_offsets::Vector{Int}, col_heights::Vector{Int},
    c2::Int, e::Int, s_de::Union{StaticInt, Int}
) where {T, Ti}
    de = dimval(s_de)
    r2 = structure.entry_retained[c2]
    dr2 = Int(structure.r_dims[r2])
    ro2 = structure.boffsets_r[r2]
    co2 = structure.coffsets[c2]
    sc_off = structure.boffsets_e[e]
    @inbounds for ir in 1:dr2
        g_S[ro2 + ir - 1] -= dot_dim(C_data, co2 + (ir - 1) * de, z_data, sc_off, s_de)
    end
    col_base = S_colptr[ro2]
    stride = col_heights[r2]
    b_row_idx = block_colptr[r2]
    b_row_end = block_colptr[r2 + 1] - 1
    @inbounds for c1 in structure.eliminated_ptr[e]:c2
        r1 = structure.entry_retained[c1]
        dr1 = Int(structure.r_dims[r1])
        wo1 = structure.coffsets[c1]
        # monotone merge-walk down the block column
        while b_row_idx <= b_row_end && block_rowval[b_row_idx] < r1
            b_row_idx += 1
        end
        @assert b_row_idx <= b_row_end && block_rowval[b_row_idx] == r1 "Sparse pattern missing entry ($r1, $r2)"
        row_offset = block_row_offsets[b_row_idx]
        for col in 1:dr2
            c_off = co2 + (col - 1) * de
            col_ptr = col_base + (col - 1) * stride + row_offset
            nrow = r1 == r2 ? col : dr1
            for row in 1:nrow
                nzval[col_ptr + row - 1] -= dot_dim(W_data, wo1 + (row - 1) * de, C_data, c_off, s_de)
            end
        end
    end
    return nothing
end

"""
    accumulate_sparse_columns!(nzval, g_S, structure, C_data, W_data, z_data, retained_ptr, retained_entry, entry_eliminated,
                               S_colptr, block_colptr, block_rowval, block_row_offsets, col_heights, r_range)

Sparse counterpart of [`accumulate_dense_columns!`](@ref), writing into `S.nzval`.
"""
function accumulate_sparse_columns!(
    nzval::Vector{T}, g_S::Vector{T}, structure::SchurStructure, C_data::Vector{T}, W_data::Vector{T}, z_data::Vector{T},
    retained_ptr::Vector{Int}, retained_entry::Vector{Int}, entry_eliminated::Vector{Int},
    S_colptr::Vector{Ti}, block_colptr::Vector{Ti}, block_rowval::Vector{Ti},
    block_row_offsets::Vector{Int}, col_heights::Vector{Int}, r_range::UnitRange{Int}
) where {T, Ti}
    @inbounds for r2 in r_range
        for k in retained_ptr[r2]:(retained_ptr[r2 + 1] - 1)
            c2 = retained_entry[k]
            e = entry_eliminated[c2]
            dispatch_dim(Int(structure.e_dims[e])) do s_de
                accumulate_entry_sparse!(nzval, g_S, structure, C_data, W_data, z_data, S_colptr, block_colptr, block_rowval,
                                         block_row_offsets, col_heights, c2, e, s_de)
            end
        end
    end
    return nothing
end

"""
    reset_schur_dense!(S, R, scaling_r, lambda, ws)

Reset dense reduced system S = R + lambda * diag(scaling_r) and the failure flag. `lambda` and `scaling_r` must be
finite and non-negative (`ArgumentError` otherwise, before `S` is written).
"""
function reset_schur_dense!(
    S::Matrix{T}, R::Matrix{T}, scaling_r::AbstractVector{T},
    lambda::T, ws::SchurReductionWorkspace{T}
) where T
    check_damping(lambda, scaling_r, :scaling_r)
    copyto!(S, R)
    @inbounds for (i, sc) in enumerate(scaling_r)
        S[i, i] += lambda * sc
    end
    ws.failed_block[] = 0
    return nothing
end

"""
    reset_schur_sparse!(nzval, R_nzval, r_to_s_map, diag_inds, scaling_r, lambda, ws)

Reset sparse reduced system S = R + lambda * diag(scaling_r) and the failure flag. `lambda` and `scaling_r` must be
finite and non-negative (`ArgumentError` otherwise, before `nzval` is written).
"""
function reset_schur_sparse!(
    nzval::Vector{T}, R_nzval::Vector{T}, r_to_s_map::Vector{Int},
    diag_inds::Vector{Int}, scaling_r::AbstractVector{T},
    lambda::T, ws::SchurReductionWorkspace{T}
) where T
    check_damping(lambda, scaling_r, :scaling_r)
    fill!(nzval, zero(T))
    @inbounds for i in eachindex(r_to_s_map)
        idx = r_to_s_map[i]
        if idx > 0
            nzval[idx] = R_nzval[i]
        end
    end
    @inbounds for (i, idx) in enumerate(diag_inds)
        nzval[idx] += lambda * scaling_r[i]
    end
    ws.failed_block[] = 0
    return nothing
end

# Phase 1 for all groups of eliminated blocks; false if any block failed
function factor_all_eliminated!(
    structure::SchurStructure, E_data::Vector{T}, C_data::Vector{T}, b_e::AbstractVector{T},
    scaling_e::AbstractVector{T}, lambda::T, ws::SchurReductionWorkspace{T}
) where T
    for g_idx in 1:nruns(structure.e_dims)
        chunks = structure.chunks[g_idx]
        threaded = use_threads(length(chunks), length(structure.entry_retained))
        dispatch_dim(runblocksize(structure.e_dims, g_idx)) do s_de
            foreach_index(length(chunks), threaded) do k
                factor_eliminated_group!(ws.E_inv_data, ws.z_data, ws.W_data, ws.W_scratch[k], ws.failed_block,
                                        structure, E_data, C_data, b_e, scaling_e, lambda, chunks[k], s_de)
            end
        end
        ws.failed_block[] != 0 && return false
    end
    return true
end

"""
    reduce_schur_dense!(S, g_S, b_r, b_e, scaling_e, lambda, structure, E_data, C_data, ws)

Two-phase Schur reduction into the dense upper triangle of `S` (already holding `R` plus damping) and `g_S`.
`lambda` and `scaling_e` must be finite and non-negative (`ArgumentError` otherwise, before any work).
"""
function reduce_schur_dense!(
    S::Matrix{T}, g_S::Vector{T}, b_r::AbstractVector{T}, b_e::AbstractVector{T},
    scaling_e::AbstractVector{T}, lambda::T, structure::SchurStructure,
    E_data::Vector{T}, C_data::Vector{T}, ws::SchurReductionWorkspace{T}
) where T
    check_damping(lambda, scaling_e, :scaling_e)
    factor_all_eliminated!(structure, E_data, C_data, b_e, scaling_e, lambda, ws) || return false
    copyto!(g_S, b_r)
    nk = length(ws.col_chunks)
    foreach_index(nk, use_threads(nk, length(structure.entry_retained))) do k
        accumulate_dense_columns!(S, g_S, structure, C_data, ws.W_data, ws.z_data, ws.retained_ptr, ws.retained_entry, ws.entry_eliminated, ws.col_chunks[k])
    end
    return true
end

"""
    reduce_schur_sparse!(S, g_S, b_r, b_e, scaling_e, lambda, structure, E_data, C_data,
                         block_colptr, block_rowval, block_row_offsets, col_heights, ws)

Two-phase Schur reduction into `S.nzval` (already holding `R` plus damping) and `g_S`.
`lambda` and `scaling_e` must be finite and non-negative (`ArgumentError` otherwise, before any work).
"""
function reduce_schur_sparse!(
    S::SparseMatrixCSC{T, Ti}, g_S::Vector{T}, b_r::AbstractVector{T}, b_e::AbstractVector{T},
    scaling_e::AbstractVector{T}, lambda::T, structure::SchurStructure,
    E_data::Vector{T}, C_data::Vector{T},
    block_colptr::Vector{Ti}, block_rowval::Vector{Ti},
    block_row_offsets::Vector{Int}, col_heights::Vector{Int},
    ws::SchurReductionWorkspace{T}
) where {T, Ti}
    check_damping(lambda, scaling_e, :scaling_e)
    factor_all_eliminated!(structure, E_data, C_data, b_e, scaling_e, lambda, ws) || return false
    copyto!(g_S, b_r)
    nk = length(ws.col_chunks)
    foreach_index(nk, use_threads(nk, length(structure.entry_retained))) do k
        accumulate_sparse_columns!(S.nzval, g_S, structure, C_data, ws.W_data, ws.z_data, ws.retained_ptr, ws.retained_entry, ws.entry_eliminated,
                                   S.colptr, block_colptr, block_rowval, block_row_offsets, col_heights, ws.col_chunks[k])
    end
    return true
end

"""
    backsubstitute!(x_e, x_r, structure, C_data, ws::SchurReductionWorkspace)

Recover step Delta x_e for all eliminated blocks independently using the `E_inv` and `z` cached in `ws`
by [`reduce_schur_dense!`](@ref) / [`reduce_schur_sparse!`](@ref):
`x_e = z_e - E_e^-1 * (sum_r C_er x_r)` for each eliminated block e. The coupling sum is accumulated first
and `E_inv` is applied once per eliminated block. Blocks above `BlockLinearSystems.D_STATIC` dimensions use
the workspace's per-chunk scratch (shared with its [`trial_workspace`](@ref)s), so nothing is allocated.
"""
function backsubstitute!(
    x_e::AbstractVector{T}, x_r::AbstractVector{T},
    structure::SchurStructure, C_data::Vector{T},
    ws::SchurReductionWorkspace{T}
) where T
    E_inv_data, z_data, scratch = ws.E_inv_data, ws.z_data, ws.W_scratch
    threaded = use_threads(structure.neblocks, length(structure.entry_retained))
    for g_idx in 1:nruns(structure.e_dims)
        de = runblocksize(structure.e_dims, g_idx)
        chunks = structure.chunks[g_idx]
        # Function barrier on the (static) block dimension
        dispatch_dim(de) do s_de
            foreach_index(length(chunks), threaded) do k
                backsubstitute_group!(x_e, x_r, structure, C_data, E_inv_data, z_data, scratch[k], chunks[k], s_de)
            end
        end
    end
    return nothing
end

function backsubstitute_group!(
    x_e::AbstractVector{T}, x_r::AbstractVector{T},
    structure::SchurStructure, C_data::Vector{T},
    E_inv_data::Vector{T}, z_data::Vector{T}, ::Matrix{T},
    e_chunk::UnitRange{Int}, s_de::StaticInt{De}
) where {T, De}
    eliminated_ptr = structure.eliminated_ptr
    entry_retained = structure.entry_retained
    coffsets = structure.coffsets
    boffsets_r = structure.boffsets_r
    boffsets_e = structure.boffsets_e
    eoffsets = structure.eoffsets
    r_dims = structure.r_dims

    @inbounds for e in e_chunk
        e_off = boffsets_e[e]

        # t = sum over the retained blocks r coupled to e of C_er * x_r
        t = zero(SVector{De, T})
        for c in eliminated_ptr[e]:(eliminated_ptr[e + 1] - 1)
            r = entry_retained[c]
            dr = Int(r_dims[r])
            ro = boffsets_r[r]
            co = coffsets[c]
            for ir in 1:dr
                t += make_svec(C_data, co + (ir - 1) * De, s_de) * x_r[ro + ir - 1]
            end
        end

        xe = make_svec(z_data, e_off, s_de) - make_smat(E_inv_data, eoffsets[e], s_de) * t
        for i in 1:De
            x_e[e_off + i - 1] = xe[i]
        end
    end
    return nothing
end

# Blocks above D_STATIC dimensions: same computation with run-time loops; `t` is the first `de` entries of the
# chunk's scratch matrix (max_De x max_De, max_De >= de)
function backsubstitute_group!(
    x_e::AbstractVector{T}, x_r::AbstractVector{T},
    structure::SchurStructure, C_data::Vector{T},
    E_inv_data::Vector{T}, z_data::Vector{T}, t::Matrix{T},
    e_chunk::UnitRange{Int}, de::Int
) where T
    @inbounds for e in e_chunk
        e_off = structure.boffsets_e[e]
        eo = structure.eoffsets[e]
        for i in 1:de
            t[i] = zero(T)
        end
        for c in structure.eliminated_ptr[e]:(structure.eliminated_ptr[e + 1] - 1)
            r = structure.entry_retained[c]
            dr = Int(structure.r_dims[r])
            ro = structure.boffsets_r[r]
            co = structure.coffsets[c]
            for ir in 1:dr
                xc = x_r[ro + ir - 1]
                for i in 1:de
                    t[i] += C_data[co + (ir - 1) * de + i - 1] * xc
                end
            end
        end
        for i in 1:de
            acc = z_data[e_off + i - 1]
            for j in 1:de
                acc -= E_inv_data[eo + (j - 1) * de + i - 1] * t[j]
            end
            x_e[e_off + i - 1] = acc
        end
    end
    return nothing
end
