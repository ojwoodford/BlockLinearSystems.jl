"""
    BlockOffsets(offsets::Vector{Int})

Wrapper indicating that the provided vector represents 1-based start offsets of each block row/column,
terminated by `total_dimension + 1`.
"""
struct BlockOffsets
    offsets::Vector{Int}
end

struct DenseBlockLinearSystem{T, M<:AbstractMatrix{T}}
    data::M
    rowblockoffsets::Vector{Int}
    columnblockoffsets::Vector{Int}
    rowblocksizes::BlockRLE{UInt8, Int}
    columnblocksizes::BlockRLE{UInt8, Int}

    function DenseBlockLinearSystem{T, M}(data::M, rowblocksizes::BlockRLE{UInt8, Int}, colblocksizes::BlockRLE{UInt8, Int}=rowblocksizes) where {T, M<:AbstractMatrix{T}}
        # Block sizes are already validated (1-255) by the BlockRLE constructor
        r_offsets = blockoffsets(rowblocksizes)
        c_offsets = rowblocksizes === colblocksizes ? r_offsets : blockoffsets(colblocksizes)
        size(data) == (r_offsets[end]-1, c_offsets[end]-1) || throw(DimensionMismatch("Matrix size $(size(data)) does not match block sizes ($(r_offsets[end]-1), $(c_offsets[end]-1))"))
        return new{T, M}(data, r_offsets, c_offsets, rowblocksizes, colblocksizes)
    end

    function DenseBlockLinearSystem{T, M}(data::M, rowblockoffsets::Vector{Int}, columnblockoffsets::Vector{Int}, rbs::BlockRLE{UInt8, Int}, cbs::BlockRLE{UInt8, Int}) where {T, M<:AbstractMatrix{T}}
        return new{T, M}(data, rowblockoffsets, columnblockoffsets, rbs, cbs)
    end
end

function DenseBlockLinearSystem{T, M}(data::M, rowblocksizes::AbstractVector{<:Integer}, colblocksizes::AbstractVector{<:Integer}=rowblocksizes) where {T, M<:AbstractMatrix{T}}
    rbs = as_block_rle(rowblocksizes)
    cbs = rowblocksizes === colblocksizes ? rbs : as_block_rle(colblocksizes)
    return DenseBlockLinearSystem{T, M}(data, rbs, cbs)
end

function DenseBlockLinearSystem{T, M}(data::M, bo::BlockOffsets, co::BlockOffsets=bo) where {T, M<:AbstractMatrix{T}}
    r_offsets = bo.offsets
    c_offsets = co.offsets
    (length(r_offsets) >= 2 && length(c_offsets) >= 2) || throw(ArgumentError("Offset vectors must have at least 2 entries"))
    r_offsets[1] == 1 || throw(ArgumentError("Row offsets must start with 1"))
    c_offsets[1] == 1 || throw(ArgumentError("Column offsets must start with 1"))
    rbs = blocksizes_from_offsets(r_offsets, "Row")
    cbs = r_offsets === c_offsets ? rbs : blocksizes_from_offsets(c_offsets, "Column")
    size(data) == (r_offsets[end]-1, c_offsets[end]-1) || throw(DimensionMismatch("Matrix size $(size(data)) does not match offsets ($(r_offsets[end]-1), $(c_offsets[end]-1))"))
    return DenseBlockLinearSystem{T, M}(data, r_offsets, c_offsets, rbs, cbs)
end

function DenseBlockLinearSystem{T, M}(data::M, rowblockoffsets::Vector{Int}, columnblockoffsets::Vector{Int}, rbs::AbstractVector{<:Integer}, cbs::AbstractVector{<:Integer}) where {T, M<:AbstractMatrix{T}}
    r_bs = as_block_rle(rbs)
    c_bs = rbs === cbs ? r_bs : as_block_rle(cbs)
    return DenseBlockLinearSystem{T, M}(data, rowblockoffsets, columnblockoffsets, r_bs, c_bs)
end

function DenseBlockLinearSystem{T}(rowblocksizes::AbstractVector{<:Integer}, colblocksizes::AbstractVector{<:Integer}=rowblocksizes) where T
    rbs = as_block_rle(rowblocksizes)
    cbs = rowblocksizes === colblocksizes ? rbs : as_block_rle(colblocksizes)
    r_offsets = blockoffsets(rbs)
    c_offsets = rowblocksizes === colblocksizes ? r_offsets : blockoffsets(cbs)
    data = zeros(T, r_offsets[end]-1, c_offsets[end]-1)
    return DenseBlockLinearSystem{T, typeof(data)}(data, r_offsets, c_offsets, rbs, cbs)
end

DenseBlockLinearSystem{T}(data::M, rowblocksizes::AbstractVector{<:Integer}, colblocksizes::AbstractVector{<:Integer}=rowblocksizes) where {T, M<:AbstractMatrix{T}} =
    DenseBlockLinearSystem{T, M}(data, rowblocksizes, colblocksizes)

DenseBlockLinearSystem(data::AbstractMatrix{T}, rowblocksizes::AbstractVector{<:Integer}, colblocksizes::AbstractVector{<:Integer}=rowblocksizes) where T =
    DenseBlockLinearSystem{T}(data, rowblocksizes, colblocksizes)

function DenseBlockLinearSystem{T}(bo::BlockOffsets, co::BlockOffsets=bo) where T
    r_offsets = bo.offsets
    c_offsets = co.offsets
    (length(r_offsets) >= 2 && length(c_offsets) >= 2) || throw(ArgumentError("Offset vectors must have at least 2 entries"))
    r_offsets[1] == 1 || throw(ArgumentError("Row offsets must start with 1"))
    c_offsets[1] == 1 || throw(ArgumentError("Column offsets must start with 1"))
    rbs = blocksizes_from_offsets(r_offsets, "Row")
    cbs = r_offsets === c_offsets ? rbs : blocksizes_from_offsets(c_offsets, "Column")
    data = zeros(T, r_offsets[end]-1, c_offsets[end]-1)
    return DenseBlockLinearSystem{T, typeof(data)}(data, r_offsets, c_offsets, rbs, cbs)
end

DenseBlockLinearSystem{T}(data::M, bo::BlockOffsets, co::BlockOffsets=bo) where {T, M<:AbstractMatrix{T}} =
    DenseBlockLinearSystem{T, M}(data, bo, co)

DenseBlockLinearSystem(data::AbstractMatrix{T}, bo::BlockOffsets, co::BlockOffsets=bo) where T =
    DenseBlockLinearSystem{T}(data, bo, co)

DenseBlockLinearSystem{T}(data::M, rowblockoffsets::Vector{Int}, columnblockoffsets::Vector{Int}, rbs::AbstractVector{<:Integer}, cbs::AbstractVector{<:Integer}) where {T, M<:AbstractMatrix{T}} =
    DenseBlockLinearSystem{T, M}(data, rowblockoffsets, columnblockoffsets, rbs, cbs)

DenseBlockLinearSystem{T}(; offsets::Vector{Int}, coloffsets::Vector{Int}=offsets) where T =
    DenseBlockLinearSystem{T}(BlockOffsets(offsets), BlockOffsets(coloffsets))

DenseBlockLinearSystem{T}(data::AbstractMatrix{T}; offsets::Vector{Int}, coloffsets::Vector{Int}=offsets) where T =
    DenseBlockLinearSystem{T}(data, BlockOffsets(offsets), BlockOffsets(coloffsets))

DenseBlockLinearSystem(data::AbstractMatrix{T}; offsets::Vector{Int}, coloffsets::Vector{Int}=offsets) where T =
    DenseBlockLinearSystem{T}(data, BlockOffsets(offsets), BlockOffsets(coloffsets))

function copy_structure(bdm::DenseBlockLinearSystem{T}) where T
    newdata = fill!(similar(bdm.data), zero(T))
    return DenseBlockLinearSystem{T, typeof(newdata)}(newdata, bdm.rowblockoffsets, bdm.columnblockoffsets, bdm.rowblocksizes, bdm.columnblocksizes)
end

uniformscaling!(bdm::DenseBlockLinearSystem, k) = uniformscaling!(bdm.data, k)

zero!(bdm::DenseBlockLinearSystem) = fill!(bdm.data, zero(eltype(bdm.data)))

@inline function checkblock(bdm::DenseBlockLinearSystem, i::Integer, j::Integer, rows::Integer, cols::Integer)
    (1 <= i <= length(bdm.rowblocksizes)) || throw(BoundsError(bdm, (i, j)))
    (1 <= j <= length(bdm.columnblocksizes)) || throw(BoundsError(bdm, (i, j)))
    expected_rows = Int(bdm.rowblocksizes[i])
    expected_cols = Int(bdm.columnblocksizes[j])
    (rows == expected_rows && cols == expected_cols) || throw(DimensionMismatch("Block ($i, $j) has size ($expected_rows, $expected_cols), but ($rows, $cols) was requested"))
    return nothing
end

@inline function block(bdm::DenseBlockLinearSystem, i::Integer, j::Integer, ::StaticInt{rows}, ::StaticInt{cols}) where {rows, cols}
    @boundscheck checkblock(bdm, i, j, rows, cols)
    return @inbounds SizedMatrix{rows, cols}(view(bdm.data, SR(0, rows-1) .+ bdm.rowblockoffsets[i], SR(0, cols-1) .+ bdm.columnblockoffsets[j]))
end

@inline function block(bdm::DenseBlockLinearSystem, i::Integer, j::Integer, rows::Integer, cols::Integer)
    @boundscheck checkblock(bdm, i, j, rows, cols)
    return @inbounds view(bdm.data, blockrange(bdm.rowblockoffsets[i], rows), blockrange(bdm.columnblockoffsets[j], cols))
end

@inline function block(bdm::DenseBlockLinearSystem, i::Integer, j::Integer, ::StaticInt{rows}, cols::Integer) where rows
    @boundscheck checkblock(bdm, i, j, rows, cols)
    return HybridArray{Tuple{rows, StaticArrays.Dynamic()}}(block(bdm, i, j, rows, cols))
end

@inline function block(bdm::DenseBlockLinearSystem, i::Integer, j::Integer, rows::Integer, ::StaticInt{cols}) where cols
    @boundscheck checkblock(bdm, i, j, rows, cols)
    return HybridArray{Tuple{StaticArrays.Dynamic(), cols}}(block(bdm, i, j, rows, cols))
end

@inline block(bdm::DenseBlockLinearSystem, i::Integer, j::Integer, ::Val{R}, ::Val{C}) where {R, C} = block(bdm, i, j, StaticInt{R}(), StaticInt{C}())
@inline block(bdm::DenseBlockLinearSystem, i::Integer, j::Integer, ::Val{R}, cols::Integer) where R = block(bdm, i, j, StaticInt{R}(), cols)
@inline block(bdm::DenseBlockLinearSystem, i::Integer, j::Integer, rows::Integer, ::Val{C}) where C = block(bdm, i, j, rows, StaticInt{C}())

@inline function block(bdm::DenseBlockLinearSystem, i::Integer, j::Integer)
    @boundscheck begin
        (1 <= i <= length(bdm.rowblocksizes)) || throw(BoundsError(bdm, (i, j)))
        (1 <= j <= length(bdm.columnblocksizes)) || throw(BoundsError(bdm, (i, j)))
    end
    return @inbounds block(bdm, i, j, bdm.rowblocksizes[i], bdm.columnblocksizes[j])
end

validblock(bdm::DenseBlockLinearSystem, i::Integer, j::Integer) =
    (1 <= i <= length(bdm.rowblocksizes)) && (1 <= j <= length(bdm.columnblocksizes))

Base.size(bdm::DenseBlockLinearSystem) = size(bdm.data)
Base.size(bdm::DenseBlockLinearSystem, dim) = size(bdm.data, dim)
Base.axes(bdm::DenseBlockLinearSystem) = axes(bdm.data)
Base.axes(bdm::DenseBlockLinearSystem, dim) = axes(bdm.data, dim)
Base.getindex(bdm::DenseBlockLinearSystem, i::Integer, j::Integer) = getindex(bdm.data, i, j)
Base.setindex!(bdm::DenseBlockLinearSystem, v, i::Integer, j::Integer) = setindex!(bdm.data, v, i, j)
Base.length(bdm::DenseBlockLinearSystem) = length(bdm.data)
Base.eltype(::DenseBlockLinearSystem{T}) where T = T
Base.Matrix(bdm::DenseBlockLinearSystem) = Matrix(bdm.data)

# Copy the strict lower triangle of `A` into the strict upper triangle, in cache-sized tiles so that both
# the reads and the writes stay within a small, cache-resident working set (a plain double loop reads or
# writes with a stride of a whole column).
const SYMMETRIFY_TILE = 32

function copylowertoupper!(A::AbstractMatrix)
    n = LinearAlgebra.checksquare(A)
    T = SYMMETRIFY_TILE
    @inbounds for jj in 1:T:n                      # tile of columns
        jend = min(jj + T - 1, n)
        for ii in jj:T:n                           # tiles of rows on or below the diagonal tile
            iend = min(ii + T - 1, n)
            for j in jj:jend
                for i in max(ii, j + 1):iend
                    A[j, i] = A[i, j]
                end
            end
        end
    end
    return A
end

function symmetrifyfull!(bdm::DenseBlockLinearSystem)
    bdm.rowblocksizes == bdm.columnblocksizes || throw(ArgumentError("Row and column block sizes must match for symmetrification"))
    copylowertoupper!(bdm.data)
    return bdm.data
end

function symmetrifyfull(bdm::DenseBlockLinearSystem)
    bdm.rowblocksizes == bdm.columnblocksizes || throw(ArgumentError("Row and column block sizes must match for symmetrification"))
    return copylowertoupper!(copy(bdm.data))
end
