using Static, StaticArrays, SparseArrays, LinearAlgebra

SR(first, last) = StaticArrays.SUnitRange(dynamic(first), dynamic(last))

zero!(aa::AbstractArray) = fill!(aa, zero(eltype(aa)))

"""
    inplacecumsum!(A::AbstractVector)

Single-argument in-place cumulative sum overwriting `A` with `cumsum(A)` and returning `A`.
"""
function inplacecumsum!(A::AbstractVector)
    total = zero(eltype(A))
    @inbounds for ind in eachindex(A)
        @fastmath total += A[ind]
        A[ind] = total
    end
    return A
end

precompile(inplacecumsum!, (Vector{Int},))

function uniformscaling!(M::AbstractMatrix, k)
    LinearAlgebra.checksquare(M)
    for i in axes(M, 1)
        @inbounds M[i, i] += k
    end
    return M
end

function cumsum1!(A, B)
    total = A[1]
    @inbounds for ind in eachindex(B)
        total += B[ind]
        A[ind + 1] = total
    end
    return A
end

function computestartindices(blocksizes)
    startind = Vector{Int}(undef, length(blocksizes))
    if !isempty(blocksizes)
        startind[1] = 1
        @inbounds startind[2:end] .= view(blocksizes, 1:length(blocksizes) - 1)
        inplacecumsum!(startind)
    end
    return startind
end

function block_sparse_nnz(sparsity, blocksizes)
    # Compute the number of non-zeros in a block sparse matrix
    nnz = 0
    for col in 1:size(sparsity, 2)
        @inbounds bc = blocksizes[col]
        for r in nzrange(sparsity, col)
            nnz += bc * @inbounds blocksizes[rowvals(sparsity)[r]]
        end
    end
    return nnz
end

function find_diag_inds(A::SparseMatrixCSC{T, Ti}) where {T, Ti}
    n = A.n
    diag_inds = Vector{Ti}(undef, n)
    @inbounds for col in 1:n
        for idx in nzrange(A, col)
            if A.rowval[idx] == col
                diag_inds[col] = idx
                break
            end
        end
    end
    return diag_inds
end
