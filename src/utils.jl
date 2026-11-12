using Static, StaticArrays, SparseArrays, LinearAlgebra

SR(first, last) = StaticArrays.SUnitRange(dynamic(first), dynamic(last))

# Range of `len` indices starting at `first`. A static length gives a statically sized range (its type
# carries the length); a run-time length must give a plain `UnitRange`, since an `SUnitRange` with a
# run-time length would create a new type on every call (type-unstable, allocating).
blockrange(first, len::StaticInt) = SR(0, len - static(1)) .+ first
blockrange(first, len::Integer) = dynamic(first):(dynamic(first) + Int(len) - 1)

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

# Offsets vector (length nblocks + 1, starting at 1) of a block layout; valid for an empty layout too
function blockoffsets(sizes)
    offsets = computestartindices(sizes)
    push!(offsets, isempty(offsets) ? 1 : offsets[end] + Int(sizes[end]))
    return offsets
end

# Block sizes from start offsets, with a descriptive error for unsorted / empty / oversized blocks
function blocksizes_from_offsets(offsets::AbstractVector{<:Integer}, what::AbstractString="Block")
    @inbounds for i in 1:length(offsets)-1
        d = offsets[i+1] - offsets[i]
        1 <= d <= 255 || throw(ArgumentError("$what offsets must be strictly increasing with block sizes between 1 and 255, but block $i spans offsets $(offsets[i]) to $(offsets[i+1]) (size $d)"))
    end
    return BlockRLE(diff(offsets))
end

function block_sparse_nnz(sparsity, blocksizes)
    # Compute the number of non-zeros in a block sparse matrix
    nnz = 0
    for col in 1:size(sparsity, 2)
        @inbounds bc = Int(blocksizes[col])
        for r in nzrange(sparsity, col)
            nnz += bc * Int(@inbounds blocksizes[rowvals(sparsity)[r]])
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
