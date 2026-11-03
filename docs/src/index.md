# BlockMatrices.jl

**High-Performance Block-Dense and Block-Sparse Matrix Accumulators in Julia**

`BlockMatrices.jl` provides cache-friendly, zero-allocation block-structured matrix representations—[`BlockSparseMatrix`](@ref) and [`BlockDenseMatrix`](@ref)—designed for fast assembly of Gauss-Newton and second-order Hessians, factor-graph linear systems, and Schur complement solvers.

## Key Features

- **Contiguous Block Storage ([`BlockSparseMatrix`](@ref))**: Each non-zero matrix block $(i, j)$ is stored contiguously in column-major order within a single flat `data::Vector{T}` buffer. Accumulating a residual's local Hessian block into `block(bsm, i, j)` writes to contiguous memory without binary searches across compressed sparse columns.
- **Zero-Allocation Static Views**: Indexing [`block(M, i, j, rows, cols)`](@ref block) with `StaticInt` dimensions returns stack-wrapped `SizedMatrix` or `HybridArray` views over the underlying storage for SIMD-accelerated block operations.
- **Fast Symmetric CSC Conversion**: [`makesparseindices`](@ref) precomputes the 1-based gather indices mapping `bsm.data` to a full symmetric `SparseMatrixCSC`, enabling $O(\text{nnz})$ in-place scatter into sparse Cholesky / $LDL^T$ solver matrices.
- **Dense & Submatrix Symmetrification**: [`symmetrifyfull`](@ref) and [`symmetrifyfull!`](@ref) mirror lower-triangular block accumulators into full symmetric dense matrices or principal block submatrices (e.g. for Schur complement reduced camera systems).
- **Automatic Sparse vs. Dense Thresholding**: [`sparse_dense_decision`](@ref) and [`block_sparse_nnz`](@ref) determine whether a block-structured system is faster to solve using sparse or dense linear algebra.

## Quick Start

```julia
using BlockMatrices, SparseArrays, Static, StaticArrays

# 1. Construct a lower-triangular BlockSparseMatrix (blocks of size 2 and 3)
# Note: sparsity pattern is passed transposed (upper-triangular)
sparsity_t = sparse([true true; false true])
bsm = BlockSparseMatrix{Float64}(sparsity_t, [2, 3], [2, 3])

# 2. Accumulate into individual blocks using StaticInt dimensions
block(bsm, 1, 1, static(2), static(2)) .+= @SMatrix [4.0 1.0; 1.0 5.0]
block(bsm, 2, 1, static(3), static(2)) .+= @SMatrix [0.5 0.0; 0.0 0.5; -0.5 0.5]
block(bsm, 2, 2, static(3), static(3)) .+= @SMatrix [3.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 3.0]

# 3. Convert to a full symmetric SparseMatrixCSC or dense Matrix
H_sparse = symmetrifysparse(bsm)
H_dense  = symmetrifyfull(bsm)
```
