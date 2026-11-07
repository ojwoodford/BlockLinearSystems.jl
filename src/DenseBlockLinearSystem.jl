"""
    BlockOffsets(offsets::Vector{Int})

Wrapper indicating that the provided vector represents 1-based start offsets of each block row/column,
terminated by `total_dimension + 1`.
"""
struct BlockOffsets
    offsets::Vector{Int}
end

struct DenseBlockLinearSystem{T}
    data::Matrix{T}
    rowblockoffsets::Vector{Int}
    columnblockoffsets::Vector{Int}
    rowblocksizes::Vector{UInt8}
    columnblocksizes::Vector{UInt8}

    function DenseBlockLinearSystem{T}(rowblocksizes::AbstractVector{<:Integer}, colblocksizes::AbstractVector{<:Integer}=rowblocksizes) where T
        all(i -> 0 < i <= 255, rowblocksizes) || throw(ArgumentError("Row block dimensions must be between 1 and 255"))
        all(i -> 0 < i <= 255, colblocksizes) || throw(ArgumentError("Column block dimensions must be between 1 and 255"))
        rbs = convert(Vector{UInt8}, rowblocksizes)
        cbs = rowblocksizes === colblocksizes ? rbs : convert(Vector{UInt8}, colblocksizes)
        r_offsets = computestartindices(rbs)
        push!(r_offsets, r_offsets[end] + Int(rbs[end]))
        c_offsets = rowblocksizes === colblocksizes ? r_offsets : begin
            o = computestartindices(cbs)
            push!(o, o[end] + Int(cbs[end]))
            o
        end
        return new{T}(zeros(T, r_offsets[end]-1, c_offsets[end]-1), r_offsets, c_offsets, rbs, cbs)
    end

    function DenseBlockLinearSystem{T}(data::Matrix{T}, rowblocksizes::AbstractVector{<:Integer}, colblocksizes::AbstractVector{<:Integer}=rowblocksizes) where T
        all(i -> 0 < i <= 255, rowblocksizes) || throw(ArgumentError("Row block dimensions must be between 1 and 255"))
        all(i -> 0 < i <= 255, colblocksizes) || throw(ArgumentError("Column block dimensions must be between 1 and 255"))
        rbs = convert(Vector{UInt8}, rowblocksizes)
        cbs = rowblocksizes === colblocksizes ? rbs : convert(Vector{UInt8}, colblocksizes)
        r_offsets = computestartindices(rbs)
        push!(r_offsets, r_offsets[end] + Int(rbs[end]))
        c_offsets = rowblocksizes === colblocksizes ? r_offsets : begin
            o = computestartindices(cbs)
            push!(o, o[end] + Int(cbs[end]))
            o
        end
        size(data) == (r_offsets[end]-1, c_offsets[end]-1) || throw(DimensionMismatch("Matrix size $(size(data)) does not match block sizes ($(r_offsets[end]-1), $(c_offsets[end]-1))"))
        return new{T}(data, r_offsets, c_offsets, rbs, cbs)
    end

    function DenseBlockLinearSystem{T}(bo::BlockOffsets, co::BlockOffsets=bo) where T
        r_offsets = bo.offsets
        c_offsets = co.offsets
        (length(r_offsets) >= 2 && length(c_offsets) >= 2) || throw(ArgumentError("Offset vectors must have at least 2 entries"))
        r_offsets[1] == 1 || throw(ArgumentError("Row offsets must start with 1"))
        c_offsets[1] == 1 || throw(ArgumentError("Column offsets must start with 1"))
        rbs = convert(Vector{UInt8}, diff(r_offsets))
        cbs = r_offsets === c_offsets ? rbs : convert(Vector{UInt8}, diff(c_offsets))
        return new{T}(zeros(T, r_offsets[end]-1, c_offsets[end]-1), r_offsets, c_offsets, rbs, cbs)
    end

    function DenseBlockLinearSystem{T}(data::Matrix{T}, bo::BlockOffsets, co::BlockOffsets=bo) where T
        r_offsets = bo.offsets
        c_offsets = co.offsets
        (length(r_offsets) >= 2 && length(c_offsets) >= 2) || throw(ArgumentError("Offset vectors must have at least 2 entries"))
        r_offsets[1] == 1 || throw(ArgumentError("Row offsets must start with 1"))
        c_offsets[1] == 1 || throw(ArgumentError("Column offsets must start with 1"))
        rbs = convert(Vector{UInt8}, diff(r_offsets))
        cbs = r_offsets === c_offsets ? rbs : convert(Vector{UInt8}, diff(c_offsets))
        size(data) == (r_offsets[end]-1, c_offsets[end]-1) || throw(DimensionMismatch("Matrix size $(size(data)) does not match offsets ($(r_offsets[end]-1), $(c_offsets[end]-1))"))
        return new{T}(data, r_offsets, c_offsets, rbs, cbs)
    end

    function DenseBlockLinearSystem{T}(data::Matrix{T}, rowblockoffsets::Vector{Int}, columnblockoffsets::Vector{Int}, rbs::Vector{UInt8}, cbs::Vector{UInt8}) where T
        return new{T}(data, rowblockoffsets, columnblockoffsets, rbs, cbs)
    end
end

DenseBlockLinearSystem{T}(; offsets::Vector{Int}, coloffsets::Vector{Int}=offsets) where T =
    DenseBlockLinearSystem{T}(BlockOffsets(offsets), BlockOffsets(coloffsets))

DenseBlockLinearSystem{T}(data::Matrix{T}; offsets::Vector{Int}, coloffsets::Vector{Int}=offsets) where T =
    DenseBlockLinearSystem{T}(data, BlockOffsets(offsets), BlockOffsets(coloffsets))


copy_structure(bdm::DenseBlockLinearSystem{T}) where T = DenseBlockLinearSystem{T}(zeros(T, size(bdm.data)), bdm.rowblockoffsets, bdm.columnblockoffsets, bdm.rowblocksizes, bdm.columnblocksizes)

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
    return @inbounds view(bdm.data, SR(0, rows-1) .+ bdm.rowblockoffsets[i], SR(0, cols-1) .+ bdm.columnblockoffsets[j])
end

@inline function block(bdm::DenseBlockLinearSystem, i::Integer, j::Integer, ::StaticInt{rows}, cols::Integer) where rows
    @boundscheck checkblock(bdm, i, j, rows, cols)
    return HybridArray{Tuple{rows, StaticArrays.Dynamic()}}(block(bdm, i, j, rows, cols))
end

@inline function block(bdm::DenseBlockLinearSystem, i::Integer, j::Integer, rows::Integer, ::StaticInt{cols}) where cols
    @boundscheck checkblock(bdm, i, j, rows, cols)
    return HybridArray{Tuple{StaticArrays.Dynamic(), cols}}(block(bdm, i, j, rows, cols))
end

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
Base.Matrix(bdm::DenseBlockLinearSystem) = copy(bdm.data)

function symmetrifyfull!(bdm::DenseBlockLinearSystem)
    bdm.rowblocksizes == bdm.columnblocksizes || throw(ArgumentError("Row and column block sizes must match for symmetrification"))
    for r in 2:size(bdm.data, 1)
        for c in 1:r-1
            @inbounds bdm.data[c, r] = bdm.data[r, c]
        end
    end
    return bdm.data
end

function symmetrifyfull(bdm::DenseBlockLinearSystem)
    bdm.rowblocksizes == bdm.columnblocksizes || throw(ArgumentError("Row and column block sizes must match for symmetrification"))
    output = copy(bdm.data)
    for r in 2:size(output, 1)
        for c in 1:r-1
            @inbounds output[c, r] = output[r, c]
        end
    end
    return output
end
