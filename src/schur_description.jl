# The layout of an elimination from a description in the caller's numbering: the block sizes, the column blocks each row
# block touches, and which column blocks to eliminate. This package orders the blocks, derives the couplings, the retained
# pairs and (on the Jacobian path) the tables, and builds the structure; the caller then fills the storage by block.

# The validated description: the eliminated column block of each row block (caller numbering, 0 for none), and the layout
# order (`order[p]` is the caller's number of layout block `p`) with its inverse
struct EliminationDescription
    nretained::Int
    order::Vector{Int}
    position::Vector{Int}
    rowblock_eliminated::Vector{Int}
    couplings::Vector{Vector{Int}}   # Per eliminated layout block, its coupled retained layout blocks, ascending
    retained_pairs::Vector{Tuple{Int, Int}}
end

function describe_elimination(blocksizes::AbstractVector{<:Integer}, rowblocks::AbstractVector,
                              eliminated::AbstractVector{Bool}, rowblocksizes::Union{Nothing, AbstractVector{<:Integer}})
    Base.require_one_based_indexing(blocksizes, rowblocks, eliminated)
    n = length(blocksizes)
    K = length(rowblocks)
    length(eliminated) == n || throw(DimensionMismatch("eliminated has length $(length(eliminated)), but there are $n column blocks"))
    if rowblocksizes !== nothing
        Base.require_one_based_indexing(rowblocksizes)
        length(rowblocksizes) == K || throw(DimensionMismatch("rowblocksizes has length $(length(rowblocksizes)), but there are $K row blocks"))
    end
    for i in 1:n
        check_block_dim(blocksizes[i], i)
    end
    nret = count(!, eliminated)
    (nret > 0 && nret < n) || throw(ArgumentError("An elimination needs at least one retained and one eliminated column block; got $nret retained and $(n - nret) eliminated"))

    # One pass over the row blocks: range, repeats, at most one eliminated block each, and the couplings (caller numbering)
    seen = zeros(Int, n)                 # seen[b] == k once row block k has listed column block b
    rowblock_eliminated = zeros(Int, K)
    coupled = [Int[] for _ in 1:n]       # Retained blocks sharing a row block with each eliminated block, with repeats
    hasrow = falses(n)
    for k in 1:K
        if rowblocksizes !== nothing
            m = rowblocksizes[k]
            m >= 1 || throw(ArgumentError("Row block $k has $m rows; every row block needs at least 1 row"))
        end
        e = 0
        for b in rowblocks[k]
            1 <= b <= n || throw(ArgumentError("Row block $k touches column block $b, but there are $n column blocks"))
            seen[b] == k && throw(ArgumentError("Row block $k lists column block $b more than once"))
            seen[b] = k
            hasrow[b] = true
            if eliminated[b]
                e == 0 || throw(ArgumentError("Row block $k touches two eliminated column blocks, $e and $b; a row block may touch at most one"))
                e = b
            end
        end
        rowblock_eliminated[k] = e
        if e != 0
            for b in rowblocks[k]
                eliminated[b] || push!(coupled[e], b)
            end
        end
    end
    if rowblocksizes !== nothing
        for b in 1:n
            (eliminated[b] && !hasrow[b]) && throw(ArgumentError("Eliminated column block $b is in no row block; eliminating a block needs at least one row"))
        end
    end
    for b in 1:n
        unique!(sort!(coupled[b]))
    end

    # The layout order: sort_schur_blocks' rule. Its second key is the smallest coupled retained block, here in the
    # caller's numbering, which orders the eliminated blocks as the retained blocks' rank among themselves would.
    order = sort_schur_blocks(eliminated, blocksizes, coupled)
    position = invperm(order)
    couplings = [sort!([position[r] for r in coupled[order[p]]]) for p in nret+1:n]

    # Pairs of retained blocks that share a row block, in layout numbering
    pairs = Tuple{Int, Int}[]
    rs = Int[]
    for k in 1:K
        empty!(rs)
        for b in rowblocks[k]
            eliminated[b] || push!(rs, position[b])
        end
        for a in 2:length(rs), c in 1:a-1
            push!(pairs, minmax(rs[a], rs[c]))
        end
    end
    unique!(sort!(pairs))
    return EliminationDescription(nret, order, position, rowblock_eliminated, couplings, pairs)
end

function description_structure(d::EliminationDescription, blocksizes::AbstractVector{<:Integer}, nchunks::Int, min_work::Integer)
    r_dims = [Int(blocksizes[d.order[p]]) for p in 1:d.nretained]
    e_dims = [Int(blocksizes[d.order[p]]) for p in d.nretained+1:length(d.order)]
    return SchurStructure(r_dims, e_dims, d.couplings; nchunks, min_work, order=d.order, retained_pairs=d.retained_pairs)
end

function SchurStructure(blocksizes::AbstractVector{<:Integer}, rowblocks::AbstractVector{<:AbstractVector{<:Integer}};
                        eliminated::AbstractVector{Bool}, nchunks::Int=1, min_work::Integer=DEFAULT_MIN_WORK)
    d = describe_elimination(blocksizes, rowblocks, eliminated, nothing)
    return description_structure(d, blocksizes, nchunks, min_work)
end

"""
    SchurJacobianStructure(blocksizes, rowblocks, rowblocksizes; eliminated, nchunks=1, min_work=4096)

The layout of an elimination on the Jacobian path, described as for [`SchurStructure`](@ref)`(blocksizes, rowblocks;
eliminated)`, with `rowblocksizes::AbstractVector{<:Integer}`, the number of rows of each row block (at least 1).
On this path a row block is the rows of `J` and of the residual that are filled together. The blocks are ordered
and coupled as on the Hessian path, and `SchurStructure(jst)` is the Hessian-path structure of the same
description.

Each row block that touches an eliminated block has its rows in that block's table; every other row block has
them in the table of its *free group*, the row blocks that touch the same set of retained blocks. A row block that
touches no column block is allowed: its rows hold only the residual. Address a row block's part of a
[`SchurBlockJacobian`](@ref) by its number in `rowblocks`, through [`blockentry`](@ref), [`block`](@ref) and
[`residual_block`](@ref).

`nchunks` and `min_work` set the kernels' threading as on the Hessian path, `min_work` in table entries.

Rejected as on the Hessian path, and also a `rowblocksizes` of the wrong length (`DimensionMismatch`), a row block
with fewer than 1 row, and an eliminated block in no row block (`ArgumentError`).

The fields of a `SchurJacobianStructure` are internal.

# Example
```jldoctest
julia> using BlockLinearSystems

julia> # Two observations of each point (2 rows each), and a prior on camera 2 (6 rows)
       jst = SchurJacobianStructure([3, 6, 3, 6], [[1, 2], [1, 4], [2, 3], [3, 4], [2]], [2, 2, 2, 2, 6];
                                    eliminated=[true, false, true, false]);

julia> blockorder(jst), nretained(jst), nblocks(jst)
([2, 4, 1, 3], 2, 4)
```
"""
function SchurJacobianStructure(blocksizes::AbstractVector{<:Integer}, rowblocks::AbstractVector{<:AbstractVector{<:Integer}},
                                rowblocksizes::AbstractVector{<:Integer}; eliminated::AbstractVector{Bool}, nchunks::Int=1,
                                min_work::Integer=DEFAULT_MIN_WORK)
    d = describe_elimination(blocksizes, rowblocks, eliminated, rowblocksizes)
    st = description_structure(d, blocksizes, nchunks, min_work)
    nret = d.nretained
    ne = length(d.order) - nret
    K = length(rowblocks)

    # Each row block's table and first row: an eliminated block's table, or the table of its free group (the row blocks
    # with the same set of retained blocks), groups numbered in order of first appearance; rows in row-block order
    row_table = Vector{Int}(undef, K)
    row_first = Vector{Int}(undef, K)
    e_rows = zeros(Int, ne)
    group = Dict{Vector{Int}, Int}()
    free_retained = Vector{Int}[]
    free_rows = Int[]
    row_ptr = Vector{Int}(undef, K + 1)
    row_ptr[1] = 1
    for k in 1:K
        row_ptr[k + 1] = row_ptr[k] + length(rowblocks[k])
    end
    row_blocks = Vector{Int}(undef, row_ptr[K + 1] - 1)
    for k in 1:K
        m = Int(rowblocksizes[k])
        i = row_ptr[k]
        for b in rowblocks[k]
            row_blocks[i] = d.position[b]
            i += 1
        end
        sort!(view(row_blocks, row_ptr[k]:row_ptr[k + 1] - 1))
        e = d.rowblock_eliminated[k]
        if e != 0
            t = d.position[e] - nret
            row_table[k] = t
            row_first[k] = e_rows[t] + 1
            e_rows[t] += m
        else
            rs = row_blocks[row_ptr[k]:row_ptr[k + 1] - 1]   # every block of the row block is retained
            g = get(group, rs, 0)
            if g == 0
                push!(free_retained, rs)
                push!(free_rows, 0)
                g = group[rs] = length(free_rows)
            end
            row_table[k] = ne + g
            row_first[k] = free_rows[g] + 1
            free_rows[g] += m
        end
    end
    rowblock_layout = (table=row_table, first=row_first, nrows=collect(Int, rowblocksizes), ptr=row_ptr, blocks=row_blocks)
    return SchurJacobianStructure(st, e_rows; free_rows, free_retained, nchunks, rowblocks=rowblock_layout)
end

# ---- Accessors ----

"""
    blockorder(structure::Union{SchurStructure, SchurJacobianStructure}) -> Vector{Int}

The layout order of the column blocks: `blockorder(structure)[p]` is the caller's number (the position in the
`blocksizes` of the description) of layout block `p`. A new vector on each call; for setup.
"""
blockorder(st::SchurStructure) = copy(st.order)

"""
    nretained(structure::Union{SchurStructure, SchurJacobianStructure}) -> Int

The number of retained column blocks: layout blocks `1:nretained(structure)` are retained, the rest eliminated.
"""
nretained(st::SchurStructure) = st.nrblocks
nblocks(st::SchurStructure) = st.nrblocks + st.neblocks
blocksizes(st::SchurStructure) = vcat(collect(Int, st.r_dims), collect(Int, st.e_dims))

blockorder(jst::SchurJacobianStructure) = blockorder(jst.schur)
nretained(jst::SchurJacobianStructure) = nretained(jst.schur)
nblocks(jst::SchurJacobianStructure) = nblocks(jst.schur)
blocksizes(jst::SchurJacobianStructure) = blocksizes(jst.schur)
SchurStructure(jst::SchurJacobianStructure) = jst.schur
