"""
    BlockSparseMatrix{T}(sparsitytransposed::SparseMatrixCSC{Bool, Int}, rowblocksizes, colblocksizes=rowblocksizes)
    BlockSparseMatrix{T}(nnzvals, indicestransposed, rowblocksizes, colblocksizes=rowblocksizes)

Cache-friendly block-sparse matrix storing each non-zero block contiguously in column-major order within a flat `data::Vector{T}` buffer.

The block sparsity pattern is provided transposed (`sparsitytransposed` has size `(length(colblocksizes), length(rowblocksizes))`) so that blocks along a block-row are contiguous in the CSC structure of `indicestransposed`.

# Fields
- `data::Vector{T}`: Contiguous storage for all non-zero block entries.
- `indices::SparseMatrixCSC{Int, Int}`: Cached start indices in `data` for each block `(row, col)`.
- `indicestransposed::SparseMatrixCSC{Int, Int}`: Transposed start indices in `data` (`indicestransposed[col, row]`).
- `rowblocksizes::Vector{UInt8}`: Row dimensions of each block row.
- `columnblocksizes::Vector{UInt8}`: Column dimensions of each block column.
- `m::Int`: Total number of scalar rows.
- `n::Int`: Total number of scalar columns.

# Example
```julia
using BlockMatrices, SparseArrays

sparsity_t = sparse([true false; true true])
bsm = BlockSparseMatrix{Float64}(sparsity_t, [2, 3], [2, 3])
block(bsm, 1, 1) .= [1.0 2.0; 2.0 4.0]
```
"""
BlockSparseMatrix

"""
    BlockDenseMatrix{T}(rowblockoffsets, columnblockoffsets=rowblockoffsets)
    BlockDenseMatrix{T}(data::Matrix{T}, rowblockoffsets, columnblockoffsets=rowblockoffsets)

Block-partitioned dense matrix backed by a standard `Matrix{T}` alongside 1-based sentinel-terminated block offset vectors `rowblockoffsets` and `columnblockoffsets`.

# Fields
- `data::Matrix{T}`: Underlying dense matrix of size `(rowblockoffsets[end]-1, columnblockoffsets[end]-1)`.
- `rowblockoffsets::Vector{Int}`: 1-based start row of each block row, terminated by `m + 1`.
- `columnblockoffsets::Vector{Int}`: 1-based start column of each block column, terminated by `n + 1`.

# Example
```julia
using BlockMatrices

bdm = BlockDenseMatrix{Float64}([1, 3, 6]) # Two blocks of sizes 2 and 3
block(bdm, 2, 1) .= reshape(1.0:6.0, 3, 2)
```
"""
BlockDenseMatrix

"""
    block(M, i, j)
    block(M, i, j, rows, cols)

Returns a mutable view into block `(i, j)` of `M` (`BlockSparseMatrix` or `BlockDenseMatrix`).
When `rows` and/or `cols` are passed as `StaticInt`, returns a stack-wrapped `SizedMatrix` or `HybridArray` view for SIMD-accelerated block operations.

# Example
```julia
using BlockMatrices, Static

bdm = BlockDenseMatrix{Float64}([1, 3, 5])
B = block(bdm, 1, 1, static(2), static(2))
B .= 1.0
```
"""
block

"""
    validblock(M, i, j) -> Bool

Returns `true` if block `(i, j)` is structurally present in `M` (always `true` for [`BlockDenseMatrix`](@ref), and `M.indicestransposed[j, i] != 0` for [`BlockSparseMatrix`](@ref)).

# Example
```julia
using BlockMatrices, SparseArrays

bsm = BlockSparseMatrix{Float64}(sparse([true false; true true]), [2, 2], [2, 2])
validblock(bsm, 1, 2) # true
validblock(bsm, 2, 1) # false
```
"""
validblock

"""
    zero!(M)

Fills all stored entries of `M` (`BlockSparseMatrix`, `BlockDenseMatrix`, or `AbstractArray`) with zero in-place.

# Example
```julia
using BlockMatrices

bdm = BlockDenseMatrix{Float64}([1, 3])
bdm[1, 1] = 5.0
zero!(bdm)
```
"""
zero!

"""
    copy_structure(M)

Creates a new [`BlockSparseMatrix`](@ref) or [`BlockDenseMatrix`](@ref) sharing the block offset and sparsity index structures of `M`, with a freshly allocated zeroed `data` buffer.

# Example
```julia
using BlockMatrices

bdm = BlockDenseMatrix{Float64}([1, 3, 5])
bdm_copy = copy_structure(bdm)
```
"""
copy_structure

"""
    uniformscaling!(M, k)

Adds scalar `k` to the main diagonal of square matrix `M` (`BlockSparseMatrix`, `BlockDenseMatrix`, or `AbstractMatrix`) in-place: `M[i, i] += k`.

# Example
```julia
using BlockMatrices

bdm = BlockDenseMatrix{Float64}([1, 3, 5])
uniformscaling!(bdm, 2.0)
```
"""
uniformscaling!

"""
    symmetrifyfull(M) -> Matrix{T}

Constructs or updates a full symmetric `Matrix{T}` from a lower-triangular block matrix `M`:
- For [`BlockDenseMatrix`](@ref): mirrors the lower triangle to the upper triangle of `M.data` in-place and returns `M.data`.
- For [`BlockSparseMatrix`](@ref): allocates a dense `Matrix{T}` and populates both triangles via [`symmetrifyfull!`](@ref).

# Example
```julia
using BlockMatrices

bdm = BlockDenseMatrix{Float64}([1, 3, 5])
bdm[3, 1] = 4.0
mat = symmetrifyfull(bdm) # mat[1, 3] == 4.0
```
"""
symmetrifyfull

"""
    symmetrifyfull!(mat::Matrix{T}, bsm::BlockSparseMatrix{T}, blocks=nothing)

Populates dense matrix `mat` in-place with the symmetric expansion of lower-triangular `bsm`.
If `blocks` is a vector of block indices, extracts and symmetrifies only the principal submatrix corresponding to `blocks`.

# Example
```julia
using BlockMatrices, SparseArrays

bsm = BlockSparseMatrix{Float64}(sparse([true true; false true]), [2, 2], [2, 2])
block(bsm, 2, 1) .= 3.0
mat = zeros(4, 4)
symmetrifyfull!(mat, bsm)
```
"""
symmetrifyfull!

"""
    symmetrifysparse(bsm::BlockSparseMatrix{T}) -> SparseMatrixCSC{T, Int}

Converts a block-lower-triangular [`BlockSparseMatrix`](@ref) into a full symmetric `SparseMatrixCSC{T, Int}` by mirroring off-diagonal blocks to the upper triangle.

# Example
```julia
using BlockMatrices, SparseArrays

bsm = BlockSparseMatrix{Float64}(sparse([true true; false true]), [2, 2], [2, 2])
block(bsm, 2, 1) .= 5.0
S = symmetrifysparse(bsm)
```
"""
symmetrifysparse

"""
    makesparseindices(bsm::BlockSparseMatrix, symmetrify::Bool=false) -> SparseMatrixCSC{Int, Int}

Precomputes a `SparseMatrixCSC{Int, Int}` whose `nzval` holds the 1-based indices into `bsm.data` for every scalar non-zero entry of `sparse(bsm)` (or `symmetrifysparse(bsm)` when `symmetrify=true`).

This enables zero-allocation copying from `bsm.data` into an existing `SparseMatrixCSC` via `H.nzval[i] = bsm.data[sparseindices.nzval[i]]`.

# Example
```julia
using BlockMatrices, SparseArrays

bsm = BlockSparseMatrix{Float64}(sparse([true true; false true]), [2, 2], [2, 2])
idx_csc = makesparseindices(bsm, true)
S = sparse(bsm, idx_csc)
```
"""
makesparseindices

"""
    cacheindices(bsm::BlockSparseMatrix)

Populates `bsm.indices` as the transpose of `bsm.indicestransposed` if not already cached.

# Example
```julia
using BlockMatrices, SparseArrays

bsm = BlockSparseMatrix{Float64}(sparse([true true; false true]), [2, 2], [2, 2])
cacheindices(bsm)
```
"""
cacheindices

"""
    clearindicescache(bsm::BlockSparseMatrix)

Clears the cached `bsm.indices` non-zero values to zero length.

# Example
```julia
using BlockMatrices, SparseArrays

bsm = BlockSparseMatrix{Float64}(sparse([true true; false true]), [2, 2], [2, 2])
cacheindices(bsm)
clearindicescache(bsm)
```
"""
clearindicescache

"""
    updatelastrowsym!(bsm::BlockSparseMatrix{T}, rowindices, blocksz) -> BlockSparseMatrix{T}

Updates the block size of the final row/column block of a symmetric `BlockSparseMatrix` to `blocksz`.
"""
updatelastrowsym!

"""
    sparse_dense_decision(ndims, sparsity, blocksizes) -> Bool
    sparse_dense_decision(ndims, nnz) -> Bool

Heuristic returning `true` when a symmetric linear system of dimension `ndims` with lower-triangular non-zero count `nnz` is faster to factorize as a sparse matrix than as a dense matrix:
`64 * nnz < 25 * ndims * (ndims - 40)`.

# Example
```julia
using BlockMatrices

sparse_dense_decision(100, 1500) # true
sparse_dense_decision(30, 100)   # false
```
"""
sparse_dense_decision

"""
    block_sparse_nnz(sparsity, blocksizes) -> Int

Computes the total number of scalar non-zeros in a block-sparse matrix given its block `sparsity::SparseMatrixCSC` pattern and `blocksizes` vector.

# Example
```julia
using BlockMatrices, SparseArrays

sparsity = sparse([true true; false true])
block_sparse_nnz(sparsity, [2, 3]) # 2*2 + 2*3 + 3*3 = 19
```
"""
block_sparse_nnz

"""
    computestartindices(blocksizes) -> Vector{Int}

Computes the 1-based starting scalar index for each block given a vector of `blocksizes`.

# Example
```julia
using BlockMatrices

computestartindices([2, 3, 4]) # [1, 3, 6]
```
"""
computestartindices

"""
    find_diag_inds(A::SparseMatrixCSC) -> Vector{Int}

Finds the index `k` in `A.nzval` corresponding to the diagonal entry `A[col, col]` for each column `col in 1:size(A, 2)`.

# Example
```julia
using BlockMatrices, SparseArrays

S = sparse([1, 2, 2], [1, 1, 2], [10.0, 20.0, 30.0], 2, 2)
find_diag_inds(S) # [1, 3]
```
"""
find_diag_inds
