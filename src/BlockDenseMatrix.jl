struct BlockDenseMatrix{T}
    data::Matrix{T}
    rowblockoffsets::Vector{Int}
    columnblockoffsets::Vector{Int}

    function BlockDenseMatrix{T}(rowblockoffsets, columnblockoffsets=rowblockoffsets) where T
        return new{T}(zeros(T, rowblockoffsets[end]-1, columnblockoffsets[end]-1), rowblockoffsets, columnblockoffsets)
    end

    function BlockDenseMatrix{T}(data::Matrix{T}, rowblockoffsets, columnblockoffsets=rowblockoffsets) where T
        return new{T}(data, rowblockoffsets, columnblockoffsets)
    end
end

copy_structure(bdm::BlockDenseMatrix{T}) where T = BlockDenseMatrix{T}(zeros(T, size(bdm.data)), bdm.rowblockoffsets, bdm.columnblockoffsets)

uniformscaling!(bdm::BlockDenseMatrix, k) = uniformscaling!(bdm.data, k)

zero!(bdm::BlockDenseMatrix) = fill!(bdm.data, zero(eltype(bdm.data)))
block(bdm::BlockDenseMatrix, i, j, ::StaticInt{rows}, ::StaticInt{cols}) where {rows, cols} = @inbounds SizedMatrix{rows, cols}(view(bdm.data, SR(0, rows-1).+bdm.rowblockoffsets[i], SR(0, cols-1).+bdm.columnblockoffsets[j]))
block(bdm::BlockDenseMatrix, i, j, rows::Integer, cols::Integer) = @inbounds view(bdm.data, SR(0, rows-1).+bdm.rowblockoffsets[i], SR(0, cols-1).+bdm.columnblockoffsets[j])
block(bdm::BlockDenseMatrix, i, j, ::StaticInt{rows}, cols::Integer) where rows = HybridArray{Tuple{rows, StaticArrays.Dynamic()}}(block(bdm, i, j, rows, cols))
block(bdm::BlockDenseMatrix, i, j, rows::Integer, ::StaticInt{cols}) where cols = HybridArray{Tuple{StaticArrays.Dynamic(), cols}}(block(bdm, i, j, rows, cols))
block(bdm::BlockDenseMatrix, i, j) = @inbounds block(bdm, i, j, bdm.rowblockoffsets[i+1]-bdm.rowblockoffsets[i], bdm.columnblockoffsets[j+1]-bdm.columnblockoffsets[j])
validblock(::BlockDenseMatrix, ::Integer, ::Integer) = true
Base.size(bdm::BlockDenseMatrix) = size(bdm.data)
Base.size(bdm::BlockDenseMatrix, dim) = size(bdm.data, dim)
Base.axes(bdm::BlockDenseMatrix) = axes(bdm.data)
Base.axes(bdm::BlockDenseMatrix, dim) = axes(bdm.data, dim)
Base.getindex(bdm::BlockDenseMatrix, i::Integer, j::Integer) = getindex(bdm.data, i, j)
Base.setindex!(bdm::BlockDenseMatrix, v, i::Integer, j::Integer) = setindex!(bdm.data, v, i, j)
Base.length(bdm::BlockDenseMatrix) = length(bdm.data)
Base.eltype(::BlockDenseMatrix{T}) where T = T
Base.Matrix(bdm::BlockDenseMatrix) = copy(bdm.data)

function symmetrifyfull(bdm::BlockDenseMatrix)
    @assert bdm.rowblockoffsets == bdm.columnblockoffsets
    # Symmetrify in-place: mirror lower triangle to upper triangle
    for r in 2:size(bdm.data, 1)
        for c in 1:r-1
            @inbounds bdm.data[c, r] = bdm.data[r, c]
        end
    end
    return bdm.data
end
