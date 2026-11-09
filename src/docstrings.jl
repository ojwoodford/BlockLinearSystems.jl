"""
    SparseBlockLinearSystem{T}(sparsitytransposed::SparseMatrixCSC{Bool, Int}, rowblocksizes, colblocksizes=rowblocksizes)
    SparseBlockLinearSystem{T}(nnzvals, indicestransposed, rowblocksizes, colblocksizes=rowblocksizes)

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
using BlockLinearSystems, SparseArrays

sparsity_t = sparse([true false; true true])
bsm = SparseBlockLinearSystem{Float64}(sparsity_t, [2, 3], [2, 3])
block(bsm, 1, 1) .= [1.0 2.0; 2.0 4.0]
```
"""
SparseBlockLinearSystem

"""
    DenseBlockLinearSystem{T}(rowblockoffsets, columnblockoffsets=rowblockoffsets)
    DenseBlockLinearSystem{T}(data::AbstractMatrix{T}, rowblockoffsets, columnblockoffsets=rowblockoffsets)
    DenseBlockLinearSystem(data::AbstractMatrix{T}, rowblockoffsets, columnblockoffsets=rowblockoffsets)

Block-partitioned dense matrix backed by a matrix buffer `data` (such as `Matrix{T}`, `SubArray`, or `FixedSizeMatrix{T}`) alongside 1-based sentinel-terminated block offset vectors `rowblockoffsets` and `columnblockoffsets`.

# Fields
- `data::AbstractMatrix{T}`: Underlying matrix buffer of size `(rowblockoffsets[end]-1, columnblockoffsets[end]-1)`.
- `rowblockoffsets::Vector{Int}`: 1-based start row of each block row, terminated by `m + 1`.
- `columnblockoffsets::Vector{Int}`: 1-based start column of each block column, terminated by `n + 1`.

# Example
```julia
using BlockLinearSystems

bdm = DenseBlockLinearSystem{Float64}([1, 3, 6]) # Two blocks of sizes 2 and 3
block(bdm, 2, 1) .= reshape(1.0:6.0, 3, 2)
```
"""
DenseBlockLinearSystem

"""
    block(M, i, j)
    block(M, i, j, rows, cols)

Returns a mutable view into block `(i, j)` of `M` (`SparseBlockLinearSystem` or `DenseBlockLinearSystem`).
When `rows` and/or `cols` are passed as `StaticInt`, returns a stack-wrapped `SizedMatrix` or `HybridArray` view for SIMD-accelerated block operations.

# Example
```julia
using BlockLinearSystems, Static

bdm = DenseBlockLinearSystem{Float64}([1, 3, 5])
B = block(bdm, 1, 1, static(2), static(2))
B .= 1.0
```
"""
block

"""
    validblock(M, i, j) -> Bool

Returns `true` if block `(i, j)` is structurally present in `M` (always `true` for [`DenseBlockLinearSystem`](@ref), and `M.indicestransposed[j, i] != 0` for [`SparseBlockLinearSystem`](@ref)).

# Example
```julia
using BlockLinearSystems, SparseArrays

bsm = SparseBlockLinearSystem{Float64}(sparse([true false; true true]), [2, 2], [2, 2])
validblock(bsm, 1, 2) # true
validblock(bsm, 2, 1) # false
```
"""
validblock

"""
    zero!(M)

Fills all stored entries of `M` (`SparseBlockLinearSystem`, `DenseBlockLinearSystem`, or `AbstractArray`) with zero in-place.

# Example
```julia
using BlockLinearSystems

bdm = DenseBlockLinearSystem{Float64}([1, 3])
bdm[1, 1] = 5.0
zero!(bdm)
```
"""
zero!

"""
    copy_structure(M)

Creates a new [`SparseBlockLinearSystem`](@ref) or [`DenseBlockLinearSystem`](@ref) sharing the block offset and sparsity index structures of `M`, with a freshly allocated zeroed `data` buffer.

# Example
```julia
using BlockLinearSystems

bdm = DenseBlockLinearSystem{Float64}([1, 3, 5])
bdm_copy = copy_structure(bdm)
```
"""
copy_structure

"""
    uniformscaling!(M, k)

Adds scalar `k` to the main diagonal of square matrix `M` (`SparseBlockLinearSystem`, `DenseBlockLinearSystem`, or `AbstractMatrix`) in-place: `M[i, i] += k`.

# Example
```julia
using BlockLinearSystems

bdm = DenseBlockLinearSystem{Float64}([1, 3, 5])
uniformscaling!(bdm, 2.0)
```
"""
uniformscaling!

"""
    symmetrifyfull(M) -> Matrix{T}

Constructs a full symmetric `Matrix{T}` from a lower-triangular block matrix `M` (`SparseBlockLinearSystem` or `DenseBlockLinearSystem`).
Allocates and returns a new symmetric `Matrix{T}`. See [`symmetrifyfull!`](@ref) for the in-place version.

# Example
```julia
using BlockLinearSystems

bdm = DenseBlockLinearSystem{Float64}([1, 3, 5])
bdm[3, 1] = 4.0
mat = symmetrifyfull(bdm) # mat[1, 3] == 4.0, mat !== bdm.data
```
"""
symmetrifyfull

"""
    symmetrifyfull!(bdm::DenseBlockLinearSystem) -> AbstractMatrix{T}
    symmetrifyfull!(mat::AbstractMatrix{T}, bsm::SparseBlockLinearSystem{T}, blocks=nothing)

Symmetrifies a lower-triangular block matrix in-place:
- For [`DenseBlockLinearSystem`](@ref): mirrors the lower triangle to the upper triangle of `bdm.data` in-place and returns `bdm.data`.
- For [`SparseBlockLinearSystem`](@ref): populates preallocated dense matrix `mat` in-place. If `blocks` is a vector of block indices, extracts and symmetrifies only the principal submatrix corresponding to `blocks`.

# Example
```julia
using BlockLinearSystems, SparseArrays

bsm = SparseBlockLinearSystem{Float64}(sparse([true true; false true]), [2, 2], [2, 2])
block(bsm, 2, 1) .= 3.0
mat = zeros(4, 4)
symmetrifyfull!(mat, bsm)
```
"""
symmetrifyfull!

"""
    symmetrifysparse(bsm::SparseBlockLinearSystem{T}) -> SparseMatrixCSC{T, Int}

Converts a block-lower-triangular [`SparseBlockLinearSystem`](@ref) into a full symmetric `SparseMatrixCSC{T, Int}` by mirroring off-diagonal blocks to the upper triangle.

# Example
```julia
using BlockLinearSystems, SparseArrays

bsm = SparseBlockLinearSystem{Float64}(sparse([true true; false true]), [2, 2], [2, 2])
block(bsm, 2, 1) .= 5.0
S = symmetrifysparse(bsm)
```
"""
symmetrifysparse

"""
    makesparseindices(bsm::SparseBlockLinearSystem, symmetrify::Bool=false) -> SparseMatrixCSC{Int, Int}

Precomputes a `SparseMatrixCSC{Int, Int}` whose `nzval` holds the 1-based indices into `bsm.data` for every scalar non-zero entry of `sparse(bsm)` (or `symmetrifysparse(bsm)` when `symmetrify=true`).

This enables zero-allocation copying from `bsm.data` into an existing `SparseMatrixCSC` via `H.nzval[i] = bsm.data[sparseindices.nzval[i]]`.

# Example
```julia
using BlockLinearSystems, SparseArrays

bsm = SparseBlockLinearSystem{Float64}(sparse([true true; false true]), [2, 2], [2, 2])
idx_csc = makesparseindices(bsm, true)
S = sparse(bsm, idx_csc)
```
"""
makesparseindices

"""
    blocksizes(sys, dim) -> Vector{UInt8}
    blocksizes(sys) -> Tuple{Vector{UInt8}, Vector{UInt8}}

Returns the block dimensions along dimension `dim` (1 for rows, 2 for columns), or a tuple of `(rowblocksizes, columnblocksizes)`.

# Example
```julia
using BlockLinearSystems

sys = DenseBlockLinearSystem{Float64}([2, 3])
blocksizes(sys, 1) # [2, 3]
```
"""
blocksizes

"""
    nblocks(sys, dim) -> Int
    nblocks(sys) -> Tuple{Int, Int}

Returns the number of blocks along dimension `dim`, or `(n_rowblocks, n_colblocks)`.

# Example
```julia
using BlockLinearSystems

sys = DenseBlockLinearSystem{Float64}([2, 3])
nblocks(sys, 1) # 2
nblocks(sys)    # (2, 2)
```
"""
nblocks

"""
    blockdata(sys) -> Array

Returns the underlying data storage buffer (a contiguous flat `Vector{T}` for [`SparseBlockLinearSystem`](@ref) or `AbstractMatrix{T}` for [`DenseBlockLinearSystem`](@ref)).
"""
blockdata

"""
    eachstoredblock(sys)

Returns an iterator yielding `(i, j, block_view)` for every structurally non-zero block in `sys`.

# Example
```julia
using BlockLinearSystems

sys = DenseBlockLinearSystem{Float64}([2, 2])
for (i, j, B) in eachstoredblock(sys)
    B .= 1.0
end
```
"""
eachstoredblock

"""
    block_sparse_nnz(sparsity, blocksizes) -> Int

Computes the total number of scalar non-zeros in a block-sparse matrix given its block `sparsity::SparseMatrixCSC` pattern and `blocksizes` vector.

# Example
```julia
using BlockLinearSystems, SparseArrays

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
using BlockLinearSystems

computestartindices([2, 3, 4]) # [1, 3, 6]
```
"""
computestartindices

"""
    find_diag_inds(A::SparseMatrixCSC) -> Vector{Int}

Finds the index `k` in `A.nzval` corresponding to the diagonal entry `A[col, col]` for each column `col in 1:size(A, 2)`.

# Example
```julia
using BlockLinearSystems, SparseArrays

S = sparse([1, 2, 2], [1, 1, 2], [10.0, 20.0, 30.0], 2, 2)
find_diag_inds(S) # [1, 3]
```
"""
find_diag_inds

"""
    sparse_with_map(sparse_sys::SparseBlockLinearSystem{T}; symmetric::Bool=false) -> (SparseMatrixCSC{T, Int}, Vector{Int})
    sparse_with_map(sparse_sys::SparseBlockLinearSystem{T}, sparse_idx::SparseMatrixCSC{Int, Int}) -> (SparseMatrixCSC{T, Int}, Vector{Int})

Precomputes a standard `SparseMatrixCSC{T, Int}` alongside a `scatter_map::Vector{Int}`.
The `scatter_map` maps entries from `sparse_sys.data` to the CSC nonzeros vector.

Subsequent updates can be performed in-place with **zero allocations** via:
```julia
copyto!(csc_mat, sparse_sys, scatter_map)
```

# Example
```julia
using BlockLinearSystems, SparseArrays

sys = SparseBlockLinearSystem{Float64}(sparse([true true; true true]), [2, 2])
block(sys, 1, 1) .= 1.0
csc_mat, scatter_map = sparse_with_map(sys; symmetric=false)

# In-place update in solver loop (0 allocations):
block(sys, 1, 1) .= 2.0
copyto!(csc_mat, sys, scatter_map)
```
"""
sparse_with_map

"""
    BlockRLE(runlengths, blocksizes)
    BlockRLE(runs::Vector{Pair})
    BlockRLE(vec::AbstractVector{<:Integer})

Run-Length Encoded (RLE) block layout metadata capturing contiguous runs of uniform block sizes.
Subtypes `AbstractVector{UInt8}` for transparent interop with all linear systems while providing
\$O(1)\$ matrix dimension lookup, \$O(1)\$ run indexing, and \$O(\\log R)\$ block-to-coordinate mapping.

# Fields
- `runlengths::Vector{Int}`: Number of blocks in each run.
- `blocksizes::Vector{UInt8}`: Block dimension in each run.
- `cumblocks::Vector{Int}`: 1-based start block index for each run, terminated by `total_blocks + 1`.
- `cummatrixdims::Vector{Int}`: 1-based start scalar matrix coordinate for each run, terminated by `total_dimension + 1`.

# Example
```julia
# 100 camera poses (dim 6) followed by 50,000 landmarks (dim 3):
rle = BlockRLE([100, 50_000], [6, 3])
length(rle)            # 50100
matrixdimension(rle)   # 150600
nruns(rle)             # 2
```
"""
BlockRLE

"""
    nruns(rle::BlockRLE) -> Int

Returns the number of uniform block runs in `rle`.
"""
nruns

"""
    runlength(rle::BlockRLE, k::Integer) -> Int

Returns the number of blocks in run `k`.
"""
runlength

"""
    runblocksize(rle::BlockRLE, k::Integer) -> UInt8

Returns the block dimension for blocks in run `k`.
"""
runblocksize

"""
    matrixdimension(rle::BlockRLE) -> Int

Returns the total scalar matrix row/column dimension represented by `rle` (equivalent to `sum(rle)`).
"""
matrixdimension

"""
    runblockrange(rle::BlockRLE, k::Integer) -> UnitRange{Int}

Returns the 1-based range of block indices belonging to run `k`.
"""
runblockrange

"""
    runmatrixrange(rle::BlockRLE, k::Integer) -> UnitRange{Int}

Returns the 1-based range of scalar rows/columns in the represented matrix belonging to run `k`.
"""
runmatrixrange

"""
    eachrun(rle::BlockRLE)

Iterator yielding named tuples `(run = k, blockrange = r, blocksize = s, matrixrange = m)`
for each uniform run in `rle`.
"""
eachrun

"""
    sort_blocks_hierarchical(roles::AbstractVector, dimensions::AbstractVector{<:Integer}; rev_dim::Bool=true)
        -> (permutation::Vector{Int}, rle::BlockRLE)

Sorts blocks using a two-tier hierarchy `(Role, Dimension)` to cleanly separate Schur-eliminated
variables from retained variables while grouping identical block dimensions into contiguous RLE runs.

Returns the permutation vector `permutation` and the resulting `BlockRLE` metadata.
"""
sort_blocks_hierarchical
