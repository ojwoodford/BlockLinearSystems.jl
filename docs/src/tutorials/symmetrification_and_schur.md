# Symmetrification & CSC Scatter

During Gauss-Newton or second-order optimization, only the block-lower triangle ($i \ge j$) of the symmetric Hessian $H$ needs to be accumulated. Once assembly is complete, the linear solver requires either a full symmetric `Matrix{T}`, a full symmetric `SparseMatrixCSC{T, Int}`, or a reduced Schur complement system.

## Zero-Allocation Scatter into `SparseMatrixCSC`

Calling [`symmetrifysparse(bsm)`](@ref symmetrifysparse) constructs a new symmetric `SparseMatrixCSC` in a single call.

In iterative optimization loops where the sparsity pattern is fixed across iterations, precompute the solver matrix and scatter map once using [`sparse_with_map`](@ref), and scatter `bsm.data` into `H_csc` with **zero allocations** in each step:

```julia
using BlockLinearSystems, SparseArrays

sparsity = sparse([
    true  false;
    true  true
])
bsm = SparseBlockLinearSystem{Float64}(sparsity, [2, 2])

# Precompute once during solver setup:
H_csc, scatter_map = sparse_with_map(bsm; symmetric=true)

# Inside each optimization iteration (0 allocations):
zero!(bsm)
block(bsm, 1, 1, Val(2), Val(2)) .= [4.0 1.0; 1.0 5.0]
block(bsm, 2, 1, Val(2), Val(2)) .= [0.5 0.2; -0.1 0.3]
block(bsm, 2, 2, Val(2), Val(2)) .= [6.0 0.0; 0.0 7.0]

copyto!(H_csc, bsm, scatter_map)
```

## Extracting Principal Submatrices

In bipartite problems, variables partition into retained blocks $R$ and eliminated blocks $E$. [`symmetrifyfull!`](@ref) accepts an optional `blocks` index vector (or boolean mask) to extract and symmetrify only a selected subset of blocks into a preallocated dense matrix:

```julia
# Extract only block 1 into a 2x2 dense matrix:
H_sub = zeros(Float64, 2, 2)
symmetrifyfull!(H_sub, bsm, [1])

# Or using a boolean mask:
symmetrifyfull!(H_sub, bsm, [true, false])
```

## Choosing Between Sparse and Dense Representations

For small or densely connected graphs, dense Cholesky factorization outperforms sparse factorization due to lower indexing overhead and BLAS/SIMD efficiency. [`block_sparse_nnz`](@ref) computes the number of scalar non-zeros implied by a block sparsity pattern, which a solver can use to choose between a [`SparseBlockLinearSystem`](@ref) and a [`DenseBlockLinearSystem`](@ref):

```julia
blocksizes = [2, 2]
nnz_lower = block_sparse_nnz(sparsity, blocksizes) # 2*2 + 2*2 + 2*2 = 12
```

For eliminating blocks rather than extracting a submatrix, see the [Schur reduction tutorial](schur_reduction.md).
