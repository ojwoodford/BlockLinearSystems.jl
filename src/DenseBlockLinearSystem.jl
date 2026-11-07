struct DenseBlockLinearSystem{T}
    data::Matrix{T}
    rowblockoffsets::Vector{Int}
    columnblockoffsets::Vector{Int}

    function DenseBlockLinearSystem{T}(rowblockoffsets::Vector{Int}, columnblockoffsets::Vector{Int}=rowblockoffsets) where T
        (length(rowblockoffsets) >= 1 && length(columnblockoffsets) >= 1) || throw(ArgumentError("Block offset vectors must not be empty"))
        rowblockoffsets[1] == 1 || throw(ArgumentError("rowblockoffsets must start with 1"))
        columnblockoffsets[1] == 1 || throw(ArgumentError("columnblockoffsets must start with 1"))
        return new{T}(zeros(T, rowblockoffsets[end]-1, columnblockoffsets[end]-1), rowblockoffsets, columnblockoffsets)
    end

    function DenseBlockLinearSystem{T}(data::Matrix{T}, rowblockoffsets::Vector{Int}, columnblockoffsets::Vector{Int}=rowblockoffsets) where T
        (length(rowblockoffsets) >= 1 && length(columnblockoffsets) >= 1) || throw(ArgumentError("Block offset vectors must not be empty"))
        size(data) == (rowblockoffsets[end]-1, columnblockoffsets[end]-1) || throw(DimensionMismatch("Matrix size $(size(data)) does not match block offsets ($(rowblockoffsets[end]-1), $(columnblockoffsets[end]-1))"))
        return new{T}(data, rowblockoffsets, columnblockoffsets)
    end
end

copy_structure(bdm::DenseBlockLinearSystem{T}) where T = DenseBlockLinearSystem{T}(zeros(T, size(bdm.data)), bdm.rowblockoffsets, bdm.columnblockoffsets)

uniformscaling!(bdm::DenseBlockLinearSystem, k) = uniformscaling!(bdm.data, k)

zero!(bdm::DenseBlockLinearSystem) = fill!(bdm.data, zero(eltype(bdm.data)))

@inline function checkblock(bdm::DenseBlockLinearSystem, i::Integer, j::Integer, rows::Integer, cols::Integer)
    (1 <= i <= length(bdm.rowblockoffsets) - 1) || throw(BoundsError(bdm, (i, j)))
    (1 <= j <= length(bdm.columnblockoffsets) - 1) || throw(BoundsError(bdm, (i, j)))
    expected_rows = bdm.rowblockoffsets[i+1] - bdm.rowblockoffsets[i]
    expected_cols = bdm.columnblockoffsets[j+1] - bdm.columnblockoffsets[j]
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
        (1 <= i <= length(bdm.rowblockoffsets) - 1) || throw(BoundsError(bdm, (i, j)))
        (1 <= j <= length(bdm.columnblockoffsets) - 1) || throw(BoundsError(bdm, (i, j)))
    end
    return @inbounds block(bdm, i, j, bdm.rowblockoffsets[i+1]-bdm.rowblockoffsets[i], bdm.columnblockoffsets[j+1]-bdm.columnblockoffsets[j])
end

validblock(bdm::DenseBlockLinearSystem, i::Integer, j::Integer) =
    (1 <= i <= length(bdm.rowblockoffsets) - 1) && (1 <= j <= length(bdm.columnblockoffsets) - 1)

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
    bdm.rowblockoffsets == bdm.columnblockoffsets || throw(ArgumentError("Row and column block offsets must match for symmetrification"))
    for r in 2:size(bdm.data, 1)
        for c in 1:r-1
            @inbounds bdm.data[c, r] = bdm.data[r, c]
        end
    end
    return bdm.data
end

function symmetrifyfull(bdm::DenseBlockLinearSystem)
    bdm.rowblockoffsets == bdm.columnblockoffsets || throw(ArgumentError("Row and column block offsets must match for symmetrification"))
    output = copy(bdm.data)
    for r in 2:size(output, 1)
        for c in 1:r-1
            @inbounds output[c, r] = output[r, c]
        end
    end
    return output
end
