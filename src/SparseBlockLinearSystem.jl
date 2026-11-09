using StaticArrays, HybridArrays, LinearAlgebra
import SparseArrays

"""
    TransposedPattern(pattern::SparseMatrixCSC{Bool, Int})

Wrapper indicating that the provided block sparsity pattern is already in transposed orientation
(size is `(length(colblocksizes), length(rowblocksizes))`).
"""
struct TransposedPattern
    pattern::SparseArrays.SparseMatrixCSC{Bool, Int}
end

struct SparseBlockLinearSystem{T, RBS<:AbstractVector{UInt8}, CBS<:AbstractVector{UInt8}}
    data::Vector{T}                                             # Storage for all the matrix data
    indices::SparseArrays.SparseMatrixCSC{Int, Int}             # Start indices for each block matrix
    indicestransposed::SparseArrays.SparseMatrixCSC{Int, Int}   # Start indices for each block matrix, stored transposed
    rowblocksizes::RBS                                          # Number of rows for each block of rows
    columnblocksizes::CBS                                       # Number of columns for each block of columns
    m::Int                                                      # Number of rows of the represented matrix
    n::Int                                                      # Number of columns of the represented matrix

    function SparseBlockLinearSystem{T, RBS, CBS}(nnzvals, indicestransposed, rowblocksizes::RBS, colblocksizes::CBS=rowblocksizes) where {T, RBS<:AbstractVector{UInt8}, CBS<:AbstractVector{UInt8}}
        all(i -> 0 < i <= 255, rowblocksizes) || throw(ArgumentError("Row block dimensions must be between 1 and 255"))
        all(i -> 0 < i <= 255, colblocksizes) || throw(ArgumentError("Column block dimensions must be between 1 and 255"))
        sumrbs = Int(sum(rowblocksizes))
        sumcbs = rowblocksizes === colblocksizes ? sumrbs : Int(sum(colblocksizes))
        size(indicestransposed) == (length(colblocksizes), length(rowblocksizes)) || throw(DimensionMismatch("Size of indicestransposed must be ($(length(colblocksizes)), $(length(rowblocksizes))), got $(size(indicestransposed))"))
        it = SparseArrays.sparse(indicestransposed)
        indices = SparseArrays.sparse(it')
        return new{T, RBS, CBS}(zeros(T, nnzvals), indices, it, rowblocksizes, colblocksizes, sumrbs, sumcbs)
    end

    function SparseBlockLinearSystem{T, RBS, CBS}(data::Vector{T}, indices, indicestransposed, rowblocksizes::RBS, colblocksizes::CBS, m, n) where {T, RBS<:AbstractVector{UInt8}, CBS<:AbstractVector{UInt8}}
        return new{T, RBS, CBS}(data, indices, indicestransposed, rowblocksizes, colblocksizes, m, n)
    end
end

function SparseBlockLinearSystem{T}(nnzvals, indicestransposed, rowblocksizes, colblocksizes=rowblocksizes) where T
    rbs = as_blocksizes(rowblocksizes)
    cbs = rowblocksizes === colblocksizes ? rbs : as_blocksizes(colblocksizes)
    return SparseBlockLinearSystem{T, typeof(rbs), typeof(cbs)}(nnzvals, indicestransposed, rbs, cbs)
end

function SparseBlockLinearSystem{T}(sparsity::SparseArrays.SparseMatrixCSC{Bool, Int}, rowblocksizes, colblocksizes=rowblocksizes; transposed::Bool=false) where T
    rbs = as_blocksizes(rowblocksizes)
    cbs = rowblocksizes === colblocksizes ? rbs : as_blocksizes(colblocksizes)
    sparsitytransposed = transposed ? sparsity : SparseArrays.sparse(sparsity')
    size(sparsitytransposed) == (length(cbs), length(rbs)) || throw(DimensionMismatch("Size of sparsity pattern must match block dimensions: expected ($(length(rbs)), $(length(cbs))), got $(size(sparsity))"))
    # Construct the block indices sparse matrix
    indicestransposed = SparseArrays.SparseMatrixCSC{Int, Int}(sparsitytransposed.m, sparsitytransposed.n, copy(sparsitytransposed.colptr), copy(sparsitytransposed.rowval), Vector{Int}(undef, length(sparsitytransposed.rowval)))
    # Compute the block pointers and length of data storage
    start = 1
    ind = 1
    for row = 1:length(rbs)
        rowwidth = convert(Int, @inbounds rbs[row])
        for col in @inbounds view(sparsitytransposed.rowval, sparsitytransposed.colptr[row]:sparsitytransposed.colptr[row+1]-1)
            @inbounds indicestransposed.nzval[ind] = start
            ind += 1
            start += rowwidth * convert(Int, @inbounds cbs[col])
        end
    end
    # Construct the SparseBlockLinearSystem
    return SparseBlockLinearSystem{T, typeof(rbs), typeof(cbs)}(start - 1, indicestransposed, rbs, cbs)
end

function SparseBlockLinearSystem{T}(tp::TransposedPattern, rowblocksizes, colblocksizes=rowblocksizes) where T
    return SparseBlockLinearSystem{T}(tp.pattern, rowblocksizes, colblocksizes; transposed=true)
end

function SparseBlockLinearSystem{T}(blockcoords::AbstractVector{<:Tuple{Integer, Integer}}, rowblocksizes, colblocksizes=rowblocksizes) where T
    rbs = as_blocksizes(rowblocksizes)
    cbs = rowblocksizes === colblocksizes ? rbs : as_blocksizes(colblocksizes)
    I = [Int(c[1]) for c in blockcoords]
    J = [Int(c[2]) for c in blockcoords]
    sparsity = SparseArrays.sparse(I, J, true, length(rbs), length(cbs))
    return SparseBlockLinearSystem{T}(sparsity, rbs, cbs; transposed=false)
end

function SparseBlockLinearSystem{T}(bsm::SparseBlockLinearSystem{T}, m::Int, n::Int=m) where T
    return SparseBlockLinearSystem{T, typeof(bsm.rowblocksizes), typeof(bsm.columnblocksizes)}(bsm.data, bsm.indices, bsm.indicestransposed, bsm.rowblocksizes, bsm.columnblocksizes, m, n)
end

function SparseBlockLinearSystem{T}(data::Vector{T}, indices, indicestransposed, rowblocksizes, colblocksizes, m, n) where T
    rbs = as_blocksizes(rowblocksizes)
    cbs = rowblocksizes === colblocksizes ? rbs : as_blocksizes(colblocksizes)
    return SparseBlockLinearSystem{T, typeof(rbs), typeof(cbs)}(data, indices, indicestransposed, rbs, cbs, m, n)
end

copy_structure(bsm::SparseBlockLinearSystem{T, RBS, CBS}) where {T, RBS, CBS} =
    SparseBlockLinearSystem{T, RBS, CBS}(zeros(T, length(bsm.data)), bsm.indices, bsm.indicestransposed, bsm.rowblocksizes, bsm.columnblocksizes, bsm.m, bsm.n)

function uniformscaling!(M::SparseBlockLinearSystem, k)
    M.rowblocksizes == M.columnblocksizes || throw(ArgumentError("SparseBlockLinearSystem must be block-square for uniformscaling!"))
    for (i, blocksz) in enumerate(M.rowblocksizes)
        @inbounds ind = M.indicestransposed[i, i]
        ind > 0 || throw(ArgumentError("Diagonal block ($i, $i) is missing in SparseBlockLinearSystem"))
        bs = Int(blocksz)
        for j in ind:(bs + 1):(ind + bs^2 - 1)
            @inbounds M.data[j] += k
        end
    end
    return M
end

zero!(bsm::SparseBlockLinearSystem) = fill!(bsm.data, zero(eltype(bsm.data)))

@inline function checkblock(bsm::SparseBlockLinearSystem, i::Integer, j::Integer, rows::Integer, cols::Integer)
    (1 <= i <= length(bsm.rowblocksizes)) || throw(BoundsError(bsm, (i, j)))
    (1 <= j <= length(bsm.columnblocksizes)) || throw(BoundsError(bsm, (i, j)))
    validblock(bsm, i, j) || throw(ArgumentError("Block ($i, $j) is structurally zero"))
    expected_rows = Int(bsm.rowblocksizes[i])
    expected_cols = Int(bsm.columnblocksizes[j])
    (rows == expected_rows && cols == expected_cols) || throw(DimensionMismatch("Block ($i, $j) has size ($expected_rows, $expected_cols), but ($rows, $cols) was requested"))
    return nothing
end

@inline function block(bsm::SparseBlockLinearSystem, i::Integer, j::Integer, ::StaticInt{rows}, ::StaticInt{cols}) where {rows, cols}
    @boundscheck checkblock(bsm, i, j, rows, cols)
    return @inbounds SizedMatrix{rows, cols}(view(bsm.data, SR(0, rows*cols-1) .+ bsm.indicestransposed[j, i]))
end

@inline function block(bsm::SparseBlockLinearSystem, i::Integer, j::Integer, rows::Integer, cols::Integer)
    @boundscheck checkblock(bsm, i, j, rows, cols)
    return @inbounds reshape(view(bsm.data, SR(0, rows*cols-1) .+ bsm.indicestransposed[j, i]), (rows, cols))
end

@inline function block(bsm::SparseBlockLinearSystem, i::Integer, j::Integer, ::StaticInt{rows}, cols::Integer) where rows
    @boundscheck checkblock(bsm, i, j, rows, cols)
    return HybridArray{Tuple{rows, StaticArrays.Dynamic()}}(block(bsm, i, j, rows, cols))
end

@inline function block(bsm::SparseBlockLinearSystem, i::Integer, j::Integer, rows::Integer, ::StaticInt{cols}) where cols
    @boundscheck checkblock(bsm, i, j, rows, cols)
    return HybridArray{Tuple{StaticArrays.Dynamic(), cols}}(block(bsm, i, j, rows, cols))
end

@inline block(bsm::SparseBlockLinearSystem, i::Integer, j::Integer, ::Val{R}, ::Val{C}) where {R, C} = block(bsm, i, j, StaticInt{R}(), StaticInt{C}())
@inline block(bsm::SparseBlockLinearSystem, i::Integer, j::Integer, ::Val{R}, cols::Integer) where R = block(bsm, i, j, StaticInt{R}(), cols)
@inline block(bsm::SparseBlockLinearSystem, i::Integer, j::Integer, rows::Integer, ::Val{C}) where C = block(bsm, i, j, rows, StaticInt{C}())

@inline function block(bsm::SparseBlockLinearSystem, i::Integer, j::Integer)
    @boundscheck begin
        (1 <= i <= length(bsm.rowblocksizes)) || throw(BoundsError(bsm, (i, j)))
        (1 <= j <= length(bsm.columnblocksizes)) || throw(BoundsError(bsm, (i, j)))
        validblock(bsm, i, j) || throw(ArgumentError("Block ($i, $j) is structurally zero"))
    end
    return @inbounds block(bsm, i, j, bsm.rowblocksizes[i], bsm.columnblocksizes[j])
end

validblock(bsm::SparseBlockLinearSystem, i::Integer, j::Integer) =
    (1 <= i <= length(bsm.rowblocksizes)) && (1 <= j <= length(bsm.columnblocksizes)) && (@inbounds bsm.indicestransposed[j, i] != 0)

Base.size(bsm::SparseBlockLinearSystem) = (bsm.m, bsm.n)
Base.size(bsm::SparseBlockLinearSystem, dim) = dim == 1 ? bsm.m : (dim == 2 ? bsm.n : 1)
Base.length(bsm::SparseBlockLinearSystem) = bsm.m * bsm.n
Base.eltype(::SparseBlockLinearSystem{T}) where T = T
SparseArrays.nnz(bsm::SparseBlockLinearSystem) = length(bsm.data)

# Accessors
blocksizes(sys::Union{DenseBlockLinearSystem, SparseBlockLinearSystem}, dim::Integer) =
    dim == 1 ? sys.rowblocksizes : (dim == 2 ? sys.columnblocksizes : throw(ArgumentError("Dimension out of range: must be 1 or 2")))

blocksizes(sys::Union{DenseBlockLinearSystem, SparseBlockLinearSystem}) =
    (sys.rowblocksizes, sys.columnblocksizes)

nblocks(sys::Union{DenseBlockLinearSystem, SparseBlockLinearSystem}, dim::Integer) =
    length(blocksizes(sys, dim))

nblocks(sys::Union{DenseBlockLinearSystem, SparseBlockLinearSystem}) =
    (nblocks(sys, 1), nblocks(sys, 2))

blockdata(sys::DenseBlockLinearSystem) = sys.data
blockdata(sys::SparseBlockLinearSystem) = sys.data

blockrange(sys::DenseBlockLinearSystem, dim::Integer, idx::Integer) =
    dim == 1 ? (sys.rowblockoffsets[idx]:sys.rowblockoffsets[idx+1]-1) :
    dim == 2 ? (sys.columnblockoffsets[idx]:sys.columnblockoffsets[idx+1]-1) :
    throw(ArgumentError("Dimension out of range: must be 1 or 2"))

function blockrange(sys::SparseBlockLinearSystem, dim::Integer, idx::Integer)
    sizes = blocksizes(sys, dim)
    (1 <= idx <= length(sizes)) || throw(BoundsError(sizes, idx))
    start = 1
    for k in 1:idx-1
        start += Int(sizes[k])
    end
    return start:(start + Int(sizes[idx]) - 1)
end

# eachstoredblock iterator
struct EachStoredBlockIterator{S}
    sys::S
end

eachstoredblock(sys::Union{DenseBlockLinearSystem, SparseBlockLinearSystem}) = EachStoredBlockIterator(sys)

Base.IteratorSize(::Type{<:EachStoredBlockIterator}) = Base.SizeUnknown()

function Base.iterate(iter::EachStoredBlockIterator{<:DenseBlockLinearSystem}, state=(1, 1))
    i, j = state
    nr, nc = nblocks(iter.sys)
    if j > nc
        return nothing
    end
    blk = block(iter.sys, i, j)
    next_state = i < nr ? (i + 1, j) : (1, j + 1)
    return ((i, j, blk), next_state)
end

function Base.iterate(iter::EachStoredBlockIterator{<:SparseBlockLinearSystem})
    it = iter.sys.indicestransposed
    cols = SparseArrays.findnz(it)
    return iterate(iter, (cols, 1))
end

function Base.iterate(iter::EachStoredBlockIterator{<:SparseBlockLinearSystem}, state)
    (cols, idx) = state
    (cs, rs, vals) = cols
    if idx > length(cs)
        return nothing
    end
    c = cs[idx]
    r = rs[idx]
    blk = block(iter.sys, r, c)
    return ((r, c, blk), (cols, idx + 1))
end

function diagonalblockspace(indices, blocksizes)
    total = Int(0)
    @inbounds for col = 1:length(indices.colptr)-1
        ind = indices.colptr[col]
        if indices.colptr[col+1] > ind
            row = indices.rowval[ind]
            row >= col || throw(ArgumentError("SparseBlockLinearSystem must be lower triangular for symmetrification"))
            if row == col
                bs = Int(blocksizes[row])
                total += bs * bs
            end
        end
    end
    return total
end

function makesparseindices(bsm::SparseBlockLinearSystem, symmetrify::Bool=false)
    !symmetrify || bsm.rowblocksizes == bsm.columnblocksizes || throw(ArgumentError("SparseBlockLinearSystem must be block-square for symmetrification"))
    # Preallocate arrays
    nzvals = symmetrify ? length(bsm.data) * 2 - diagonalblockspace(bsm.indices, bsm.rowblocksizes) : length(bsm.data)
    rows = Vector{Int}(undef, nzvals)
    indices = Vector{Int}(undef, nzvals)
    startrow = Vector{Int}(undef, length(bsm.rowblocksizes)+1)
    startrow[1] = 1
    cumsum1!(startrow, bsm.rowblocksizes)
    cols = Vector{Int}(undef, symmetrify ? startrow[end] : sum(bsm.columnblocksizes)+1)
    # Sparsify each block column
    ind = 1
    col = 1
    @inbounds cols[1] = 1
    @inbounds for (col_, colblocksize) in enumerate(bsm.columnblocksizes)
        lower_rows = bsm.indices.colptr[col_]:bsm.indices.colptr[col_+1]-1
        upper_rows = symmetrify ? (bsm.indicestransposed.colptr[col_]:bsm.indicestransposed.colptr[col_+1]-1-(!isempty(lower_rows) && bsm.indices.rowval[lower_rows[1]] == col_)) : 1:0
        for innercol in 0:colblocksize-1
            # Above diagonal blocks if symmetrifying (transposed)
            for r in upper_rows
                row = bsm.indicestransposed.rowval[r]
                s = startrow[row]
                c = bsm.rowblocksizes[row]
                v = bsm.indicestransposed.nzval[r] + innercol
                for i in 0:c-1
                    rows[ind] = s + i
                    indices[ind] = v
                    ind += 1
                    v += colblocksize
                end
            end
            # Diagonal and below diagonal blocks
            for r in lower_rows
                row = bsm.indices.rowval[r]
                s = startrow[row]
                c = bsm.rowblocksizes[row]
                v = bsm.indices.nzval[r] + innercol * c
                for i in 0:c-1
                    rows[ind] = s + i
                    indices[ind] = v + i
                    ind += 1
                end
            end
            col += 1
            cols[col] = ind
        end
    end
    # Construct the sparse matrix
    return SparseArrays.SparseMatrixCSC{Int, Int}(startrow[end]-1, col-1, cols, rows, indices)
end

SparseArrays.sparse(bsm::SparseBlockLinearSystem{T}, indices=makesparseindices(bsm)) where T = SparseArrays.SparseMatrixCSC{T, Int}(indices.m, indices.n, indices.colptr, indices.rowval, bsm.data[indices.nzval])

symmetrifysparse(bsm::SparseBlockLinearSystem{T}) where T = SparseArrays.sparse(bsm, makesparseindices(bsm, true))

function symmetrifyfull(bsm::SparseBlockLinearSystem{T}) where T
    bsm.rowblocksizes == bsm.columnblocksizes || throw(ArgumentError("SparseBlockLinearSystem must be block-square for symmetrifyfull"))
    # Allocate the output
    len = sum(bsm.rowblocksizes)
    output = Matrix{T}(undef, len, len)
    # Fill in the matrix
    symmetrifyfull!(output, bsm)
    # Return the matrix
    return output
end

function symmetrifyfull!(mat::AbstractMatrix{T}, bsm::SparseBlockLinearSystem{T}, blocks=nothing) where T
    bsm.rowblocksizes == bsm.columnblocksizes || throw(ArgumentError("SparseBlockLinearSystem must be block-square for symmetrifyfull!"))
    if blocks === nothing
        lengths = bsm.rowblocksizes
        starts = cumsum(vcat(1, bsm.rowblocksizes[1:end-1]))
    else
        block_indices = blocks isa AbstractVector{Bool} ? findall(blocks) : blocks
        if isempty(block_indices)
            size(mat) == (0, 0) || throw(DimensionMismatch("Target matrix has size $(size(mat)), expected (0, 0)"))
            return nothing
        end
        all(b -> 1 <= b <= length(bsm.rowblocksizes), block_indices) || throw(BoundsError(bsm.rowblocksizes, block_indices))
        lengths = zeros(UInt, length(bsm.rowblocksizes))
        lengths[block_indices] .= bsm.rowblocksizes[block_indices]
        starts = zeros(UInt, length(bsm.rowblocksizes))
        starts[block_indices] .= cumsum(vcat(1, bsm.rowblocksizes[block_indices[1:end-1]]))
    end
    m = sum(lengths)
    size(mat) == (m, m) || throw(DimensionMismatch("Target matrix has size $(size(mat)), expected ($m, $m)"))
    # Copy blocks into the output
    fill!(mat, 0)
    for (c, r, index) in zip(SparseArrays.findnz(bsm.indicestransposed)...)
        cs = starts[c]
        if cs == 0
            continue
        end
        rs = starts[r]
        if rs == 0
            continue
        end
        r_ = lengths[r]
        c_ = lengths[c]
        V = reshape(view(bsm.data, index:index+(r_*c_)-1), (r_, c_))
        @inbounds mat[rs:rs+r_-1,cs:cs+c_-1] .= V
        if r != c
            @inbounds mat[cs:cs+c_-1,rs:rs+r_-1] .= V'
        end
    end
    return nothing
end

function sparse_with_map(sparse_sys::SparseBlockLinearSystem{T}; symmetric::Bool=false) where T
    sparse_idx = makesparseindices(sparse_sys, symmetric)
    return sparse_with_map(sparse_sys, sparse_idx)
end

function sparse_with_map(sparse_sys::SparseBlockLinearSystem{T}, sparse_idx::SparseArrays.SparseMatrixCSC{Int, Int}) where T
    scatter_map = copy(sparse_idx.nzval)
    nzval = Vector{T}(undef, length(scatter_map))
    csc_mat = SparseArrays.SparseMatrixCSC{T, Int}(sparse_idx.m, sparse_idx.n, sparse_idx.colptr, sparse_idx.rowval, nzval)
    copyto!(csc_mat, sparse_sys, scatter_map)
    return (csc_mat, scatter_map)
end

function Base.copyto!(dest::SparseArrays.SparseMatrixCSC{T}, src::SparseBlockLinearSystem{T}, scatter_map::AbstractVector{<:Integer}) where T
    length(dest.nzval) == length(scatter_map) || throw(DimensionMismatch("Destination nzval length ($(length(dest.nzval))) does not match scatter map length ($(length(scatter_map)))"))
    @inbounds for (i, idx) in enumerate(scatter_map)
        dest.nzval[i] = src.data[idx]
    end
    return dest
end

function Base.Matrix(bsm::SparseBlockLinearSystem{T}) where T
    # Allocate the output
    rowstarts = cumsum(bsm.rowblocksizes)
    pushfirst!(rowstarts, UInt(0))
    colstarts = cumsum(bsm.columnblocksizes)
    pushfirst!(colstarts, UInt(0))
    output = zeros(T, rowstarts[end], colstarts[end])
    # Copy blocks into the output
    for (c, r, index) in zip(SparseArrays.findnz(bsm.indicestransposed)...)
        @inbounds rs = rowstarts[r]
        @inbounds rf = rowstarts[r+1]
        @inbounds cs = colstarts[c]
        @inbounds cf = colstarts[c+1]
        r_ = rf - rs
        c_ = cf - cs
        @inbounds output[rs+1:rf,cs+1:cf] .= reshape(view(bsm.data, index:index+r_*c_-1), (r_, c_))
    end
    # Return the matrix
    return output
end
