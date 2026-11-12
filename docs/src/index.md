# BlockLinearSystems.jl

**High-Performance Block-Dense and Block-Sparse Linear System Accumulators in Julia**

`BlockLinearSystems.jl` provides cache-friendly, zero-allocation block-structured matrix representations—[`SparseBlockLinearSystem`](@ref) and [`DenseBlockLinearSystem`](@ref)—designed for fast assembly of Gauss-Newton and second-order Hessians, factor-graph linear systems, and Schur complement solvers.

## Key Features

- **Contiguous Block Storage ([`SparseBlockLinearSystem`](@ref))**: Each non-zero matrix block $(i, j)$ is stored contiguously in column-major order within a single flat `data::Vector{T}` buffer. Accumulating a residual's local Hessian block into `block(bsm, i, j)` writes to contiguous memory without binary searches across compressed sparse columns.
- **Zero-Allocation Static Views**: Indexing [`block(M, i, j, rows, cols)`](@ref block) with `StaticInt` or `Val` dimensions returns stack-wrapped `SizedMatrix` or `HybridArray` views over the underlying storage for SIMD-accelerated block operations.
- **Zero-Allocation CSC Scatter**: [`sparse_with_map`](@ref) precomputes a `SparseMatrixCSC` and an integer scatter map once during solver setup. In subsequent optimization iterations, `copyto!(csc_mat, bsm, scatter_map)` updates the sparse solver matrix in-place with **zero allocations**.
- **Dense & Submatrix Symmetrification**: [`symmetrifyfull`](@ref) and [`symmetrifyfull!`](@ref) mirror lower-triangular block accumulators into full symmetric dense matrices or principal block submatrices (e.g. for the reduced systems of Schur complement solvers).
- **Schur Complement Reduction**: [`SchurStructure`](@ref), [`SchurBlockMatrix`](@ref) and the parallel [`reduce_schur_dense!`](@ref) / [`reduce_schur_sparse!`](@ref) kernels eliminate blocks of any dimension (1–255) from bipartite systems, with dense or sparse reduced systems. See the [tutorial](tutorials/schur_reduction.md).
- **Block Sparsity Statistics**: [`block_sparse_nnz`](@ref) computes the scalar non-zero count implied by a block sparsity pattern, e.g. to help a solver choose between sparse and dense linear algebra.

## Quick Start

```julia
using BlockLinearSystems, SparseArrays, Static, StaticArrays

# 1. Construct a lower-triangular SparseBlockLinearSystem (blocks of size 2 and 3)
sparsity = sparse([
    true  false;
    true  true
])
bsm = SparseBlockLinearSystem{Float64}(sparsity, [2, 3])

# 2. Setup CSC matrix and scatter map for the solver
H_sparse, scatter_map = sparse_with_map(bsm; symmetric=true)

# 3. Accumulate into individual blocks using StaticInt or Val dimensions
zero!(bsm)
block(bsm, 1, 1, Val(2), Val(2)) .+= @SMatrix [4.0 1.0; 1.0 5.0]
block(bsm, 2, 1, Val(3), Val(2)) .+= @SMatrix [0.5 0.0; 0.0 0.5; -0.5 0.5]
block(bsm, 2, 2, Val(3), Val(3)) .+= @SMatrix [3.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 3.0]

# 4. In-place zero-allocation scatter into the solver's CSC matrix
copyto!(H_sparse, bsm, scatter_map)
```
