"""
    SchurStructure

Metadata describing bipartite variable partitioning, block offsets, and coupling sparsity.
"""
struct SchurStructure
    nr::Int                          # Total scalar dimension of retained variables
    nrblocks::Int                    # Number of retained blocks
    neblocks::Int                    # Number of eliminated blocks
    r_dims::BlockRLE{BlockSize, Int} # Dimension of each retained block (indexing returns Int)
    e_dims::BlockRLE{BlockSize, Int} # Dimension of each eliminated block; its runs group the blocks of uniform dimension
    boffsets_r::Vector{Int}          # Scalar start index of each retained block in x_r
    boffsets_e::Vector{Int}          # Scalar start index of each eliminated block in x_e
    eoffsets::Vector{Int}           # Start offset in flat E_data for each E_e block
    eliminated_ptr::Vector{Int}               # Eliminated-block-major CSR pointer for coupling entries
    entry_retained::Vector{Int}             # Retained block index for each coupling entry (sorted ascending per eliminated block!)
    coffsets::Vector{Int}           # Start offset in flat C_data for each coupling block
    max_coupled_dim::Int                 # Maximum over eliminated blocks e of sum_{r coupled to e} d_r
    chunks::Vector{Vector{UnitRange{Int}}} # chunks[run][chunk_id], for each run of e_dims
end

function SchurStructure(
    r_dims::AbstractVector{<:Integer},
    e_dims::AbstractVector{<:Integer},
    couplings::AbstractVector{<:AbstractVector{<:Integer}};
    nchunks::Int=1
)
    nrblocks = length(r_dims)
    neblocks = length(e_dims)
    length(couplings) == neblocks || throw(DimensionMismatch("Number of coupling lists ($(length(couplings))) must match the number of eliminated blocks ($neblocks)"))
    nchunks >= 1 || throw(ArgumentError("nchunks must be at least 1, got $nchunks"))

    # Validate inputs here, rather than failing (or silently going wrong) deep inside the reduction. BlockRLE
    # checks the block dimensions.
    r_rle = as_block_rle(r_dims)
    e_rle = as_block_rle(e_dims)
    seen = zeros(Int, nrblocks)   # seen[r] == e once eliminated block e has been coupled to retained block r
    for (e, obs) in enumerate(couplings)
        for r in obs
            1 <= r <= nrblocks || throw(ArgumentError("Eliminated block $e references retained block $r, but there are only $nrblocks retained blocks"))
            seen[r] == e && throw(ArgumentError("Retained block $r is listed more than once in the couplings of eliminated block $e"))
            seen[r] = e
        end
    end

    nr = sum(r_rle)
    boffsets_r = computestartindices(r_rle)
    boffsets_e = computestartindices(e_rle)

    eoffsets = Vector{Int}(undef, neblocks)
    if neblocks > 0
        eoffsets[1] = 1
        for i in 2:neblocks
            eoffsets[i] = eoffsets[i-1] + e_rle[i-1]^2
        end
    end

    eliminated_ptr = Vector{Int}(undef, neblocks + 1)
    eliminated_ptr[1] = 1
    for e in 1:neblocks
        eliminated_ptr[e+1] = eliminated_ptr[e] + length(couplings[e])
    end
    num_couplings = eliminated_ptr[end] - 1

    entry_retained = Vector{Int}(undef, num_couplings)
    coffsets = Vector{Int}(undef, num_couplings)
    curr_c_off = 1
    idx = 1
    for e in 1:neblocks
        de = e_rle[e]
        sorted_retained = sort(couplings[e])
        for r in sorted_retained
            entry_retained[idx] = r
            coffsets[idx] = curr_c_off
            curr_c_off += r_rle[r] * de
            idx += 1
        end
    end

    max_coupled_dim = 0
    for e in 1:neblocks
        obs_dim = sum(r_rle[r] for r in couplings[e]; init=0)
        max_coupled_dim = max(max_coupled_dim, obs_dim)
    end

    chunks = Vector{Vector{UnitRange{Int}}}(undef, nruns(e_rle))
    for g_idx in 1:nruns(e_rle)
        grp = runblockrange(e_rle, g_idx)
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
        nr, nrblocks, neblocks,
        r_rle, e_rle, boffsets_r, boffsets_e,
        eoffsets, eliminated_ptr, entry_retained, coffsets,
        max_coupled_dim, chunks
    )
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
        for j in 1:bs
            scalar_perm[cursor] = bo + j - 1
            cursor += 1
        end
    end
    return scalar_perm
end

"""
    sort_schur_blocks(is_eliminated::AbstractVector{Bool}, block_sizes::AbstractVector{<:Integer}, couplings::AbstractVector{<:AbstractVector{<:Integer}}) -> Vector{Int}

Two-pass variable ordering:
- Pass 1: Retained variables (role 0) ordered by dimension descending; eliminated variables (role 1) grouped by dimension descending.
- Pass 2: Within each eliminated dimension group, sort eliminated blocks by their first coupled retained block index r_min(e) = min_{r coupled to e} r.
Returns a block permutation vector mapping old block indices to new positions.
"""
function sort_schur_blocks(
    is_eliminated::AbstractVector{Bool},
    block_sizes::AbstractVector{<:Integer},
    couplings::AbstractVector{<:AbstractVector}
)
    nblocks = length(is_eliminated)
    @assert length(block_sizes) == nblocks "is_eliminated and block_sizes must have equal length"

    # Retained blocks (sorted by dimension descending)
    r_indices = findall(!, is_eliminated)
    sort!(r_indices, by = i -> -Int(block_sizes[i]))

    # Eliminable blocks
    e_indices = findall(identity, is_eliminated)
    # Group by dimension descending, then within dimension by r_min
    sort!(e_indices, by = function(i)
        de = Int(block_sizes[i])
        coupled = couplings[i]
        r_min = isempty(coupled) ? typemax(Int) : minimum(coupled)
        return (-de, r_min)
    end)

    return [r_indices; e_indices]
end

"""
    build_schur_pattern(structure::SchurStructure, [T=Float64]; extra_blocks=()) -> NamedTuple

Compute block-level and scalar-level sparsity patterns of the reduced system S, where two retained blocks are linked if they are coupled to a common eliminated block.
Returns `(block_pattern, block_colptr, block_rowval, block_row_offsets, col_heights, S_sparse, diag_inds)`;
`S_sparse` is a zero-valued `SparseMatrixCSC{T}`.

The pattern contains every pair of retained blocks that share an eliminated block, plus the diagonal.
Retained-retained couplings that do not go through an eliminated block (direct couplings between
retained blocks that share no eliminated block) have no such slot, so list them in `extra_blocks`, an
iterable of `(i, j)` retained block index pairs (in either order; `i == j` is allowed and ignored).
Use [`retained_block_pairs`](@ref) to obtain them from the block pattern of `R`.
"""
function build_schur_pattern(structure::SchurStructure, ::Type{T}=Float64; extra_blocks=()) where T
    nrblocks = structure.nrblocks
    neblocks = structure.neblocks
    r_dims = structure.r_dims
    eliminated_ptr = structure.eliminated_ptr
    entry_retained = structure.entry_retained
    boffsets_r = structure.boffsets_r
    nr = structure.nr

    # 1. Build the block pattern of the reduced system (upper triangle). Collect (I, J) pairs and build the
    # sparse matrix once: inserting entries one at a time into a SparseMatrixCSC is O(nnz^2).
    I_cov = Int[]
    J_cov = Int[]
    for e in 1:neblocks
        c_start = eliminated_ptr[e]
        c_end = eliminated_ptr[e + 1] - 1
        for idx2 in c_start:c_end
            r2 = entry_retained[idx2]
            for idx1 in c_start:idx2
                push!(I_cov, entry_retained[idx1])
                push!(J_cov, r2)
            end
        end
    end
    for (i, j) in extra_blocks
        (1 <= i <= nrblocks && 1 <= j <= nrblocks) || throw(ArgumentError("Extra block pair ($i, $j) is outside the $nrblocks retained blocks"))
        push!(I_cov, min(i, j))
        push!(J_cov, max(i, j))
    end
    for r in 1:nrblocks
        push!(I_cov, r)
        push!(J_cov, r)
    end
    block_pattern = sparse(I_cov, J_cov, true, nrblocks, nrblocks, |)

    block_colptr = Vector{Int}(block_pattern.colptr)
    block_rowval = Vector{Int}(block_pattern.rowval)
    nnz_blocks = nnz(block_pattern)

    # 2. Compute scalar row offsets and column heights
    col_heights = zeros(Int, nrblocks)
    block_row_offsets = zeros(Int, nnz_blocks)
    for r2 in 1:nrblocks
        h = 0
        for idx in block_colptr[r2]:(block_colptr[r2 + 1] - 1)
            r1 = block_rowval[idx]
            block_row_offsets[idx] = h
            h += Int(r_dims[r1])
        end
        col_heights[r2] = h
    end

    # 3. Build scalar SparseMatrixCSC S pattern
    I_list = Int[]
    J_list = Int[]
    for r2 in 1:nrblocks
        for idx in block_colptr[r2]:(block_colptr[r2 + 1] - 1)
            r1 = block_rowval[idx]
            for r_col in 1:r_dims[r2], r_row in 1:r_dims[r1]
                push!(I_list, boffsets_r[r1] + r_row - 1)
                push!(J_list, boffsets_r[r2] + r_col - 1)
            end
        end
    end
    S_sparse = sparse(I_list, J_list, zeros(T, length(I_list)), nr, nr)

    # 4. Compute diagonal indices in S_sparse.nzval
    diag_inds = zeros(Int, nr)
    for j in 1:nr
        for idx in S_sparse.colptr[j]:(S_sparse.colptr[j + 1] - 1)
            if S_sparse.rowval[idx] == j
                diag_inds[j] = idx
                break
            end
        end
        @assert diag_inds[j] != 0 "Missing diagonal entry for retained scalar $j"
    end

    return (
        block_pattern = block_pattern,
        block_colptr = block_colptr,
        block_rowval = block_rowval,
        block_row_offsets = block_row_offsets,
        col_heights = col_heights,
        S_sparse = S_sparse,
        diag_inds = diag_inds
    )
end

"""
    retained_block_pairs(R::SparseBlockLinearSystem) -> Vector{Tuple{Int, Int}}

Off-diagonal stored blocks of `R` as `(row_block, col_block)` pairs. Pass them as `extra_blocks` to
[`build_schur_pattern`](@ref) so the reduced system has a slot for every coupling in `R`.
"""
function retained_block_pairs(R::SparseBlockLinearSystem)
    cols, rows, _ = SparseArrays.findnz(R.indicestransposed)
    return [(Int(brow), Int(bcol)) for (bcol, brow) in zip(cols, rows) if brow != bcol]
end

"""
    build_r_to_s_map(R_sparse::SparseMatrixCSC, S_sparse::SparseMatrixCSC) -> Vector{Int}

Build mapping vector from R_sparse.nzval to S_sparse.nzval.
"""
function build_r_to_s_map(R_sparse::SparseMatrixCSC, S_sparse::SparseMatrixCSC)
    r_to_s = Int[]
    for j in 1:R_sparse.n
        for k in R_sparse.colptr[j]:(R_sparse.colptr[j + 1] - 1)
            i = R_sparse.rowval[k]
            s_k = find_in_column(S_sparse, i, j)
            @assert s_k != 0 "R entry ($i, $j) not present in S sparsity pattern"
            push!(r_to_s, s_k)
        end
    end
    return r_to_s
end

# Index of `row` in column `j` of the sorted-pattern CSC matrix `S` (0 if absent), by binary search
@inline function find_in_column(S::SparseMatrixCSC, row::Integer, j::Integer)
    lo = Int(S.colptr[j])
    hi = Int(S.colptr[j + 1]) - 1
    idx = lo + searchsortedfirst(view(S.rowval, lo:hi), row) - 1
    return (idx <= hi && S.rowval[idx] == row) ? idx : 0
end

"""
    build_r_to_s_map(R_sys::SparseBlockLinearSystem, S_sparse::SparseMatrixCSC, boffsets_r::Vector{Int}) -> Vector{Int}

Build mapping vector from R_sys.data to S_sparse.nzval for symmetric upper-triangle reduction.
Entries of `R_sys.data` that are not mapped (the strict lower triangle of diagonal blocks) are 0.
"""
function build_r_to_s_map(R_sys::SparseBlockLinearSystem, S_sparse::SparseMatrixCSC, boffsets_r::Vector{Int})
    r_to_s = zeros(Int, length(R_sys.data))
    for (bcol, brow, data_start) in zip(SparseArrays.findnz(R_sys.indicestransposed)...)
        drow = Int(R_sys.rowblocksizes[brow])
        dcol = Int(R_sys.columnblocksizes[bcol])

        if brow == bcol
            for col in 1:dcol
                scol = boffsets_r[brow] + col - 1
                for row in 1:col
                    srow = boffsets_r[brow] + row - 1
                    s_k = find_in_column(S_sparse, srow, scol)
                    @assert s_k != 0 "Missing diagonal entry ($srow, $scol) in S"
                    r_to_s[data_start + (col - 1)*drow + row - 1] = s_k
                end
            end
        elseif brow > bcol
            for col in 1:dcol
                srow = boffsets_r[bcol] + col - 1
                for row in 1:drow
                    scol = boffsets_r[brow] + row - 1
                    s_k = find_in_column(S_sparse, srow, scol)
                    @assert s_k != 0 "Missing entry ($srow, $scol) in S"
                    r_to_s[data_start + (col - 1)*drow + row - 1] = s_k
                end
            end
        end
    end
    return r_to_s
end

"""
    build_r_to_s_map(R_sys::SparseBlockLinearSystem, pattern, boffsets_r::Vector{Int}) -> Vector{Int}

As above, but using the block-level pattern in `pattern` (the result of [`build_schur_pattern`](@ref)):
one block-level search per stored block, then the scalar position follows from `block_row_offsets` and
`col_heights` with integer arithmetic.
"""
function build_r_to_s_map(R_sys::SparseBlockLinearSystem, pattern::NamedTuple, boffsets_r::Vector{Int})
    S_colptr = pattern.S_sparse.colptr
    r_to_s = zeros(Int, length(R_sys.data))
    for (bcol, brow, data_start) in zip(SparseArrays.findnz(R_sys.indicestransposed)...)
        brow >= bcol || continue   # only lower-triangle blocks are stored in R
        drow = Int(R_sys.rowblocksizes[brow])
        dcol = Int(R_sys.columnblocksizes[bcol])
        # Block (bcol, brow) of the upper-triangle pattern: block column brow, block row bcol
        rng = pattern.block_colptr[brow]:(pattern.block_colptr[brow + 1] - 1)
        k = searchsortedfirst(view(pattern.block_rowval, rng), bcol)
        (k <= length(rng) && pattern.block_rowval[rng[k]] == bcol) || throw(AssertionError("Block ($bcol, $brow) of R is not present in the pattern of S"))
        row_offset = pattern.block_row_offsets[rng[k]]
        base = S_colptr[boffsets_r[brow]] + row_offset
        stride = pattern.col_heights[brow]
        if brow == bcol
            for col in 1:dcol, row in 1:col
                r_to_s[data_start + (col - 1)*drow + row - 1] = base + (col - 1) * stride + row - 1
            end
        else
            for col in 1:dcol, row in 1:drow
                r_to_s[data_start + (col - 1)*drow + row - 1] = base + (row - 1) * stride + col - 1
            end
        end
    end
    return r_to_s
end
