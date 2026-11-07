# Symmetrification, CSC Scatter & Schur Systems

During Gauss-Newton or second-order optimization, only the block-lower triangle ($i \ge j$) of the symmetric Hessian $H$ needs to be accumulated. Once assembly is complete, the linear solver requires either a full symmetric `Matrix{T}`, a full symmetric `SparseMatrixCSC{T, Int}`, or a reduced Schur complement system.

## Zero-Allocation Scatter into `SparseMatrixCSC`

Calling [`symmetrifysparse(bsm)`](@ref symmetrifysparse) constructs a new symmetric `SparseMatrixCSC` in a single call. In iterative optimization loops where the sparsity pattern is fixed across iterations, you can precompute the index map once using [`makesparseindices`](@ref) and scatter `bsm.data` into an existing `SparseMatrixCSC` with zero allocations:

```julia
using BlockMatrices, SparseArrays

sparsity_t = sparse([true true; false true])
bsm = BlockSparseMatrix{Float64}(sparsity_t, [2, 2], [2, 2])

# Precompute once during solver setup:
idx_map = makesparseindices(bsm, true) # true = symmetrify upper & lower triangles
H_csc = SparseMatrixCSC{Float64, Int}(
    idx_map.m, idx_map.n, idx_map.colptr, idx_map.rowval,
    zeros(Float64, length(idx_map.nzval))
)
scatter_indices = idx_map.nzval
diag_inds = find_diag_inds(H_csc)

# Inside each optimization iteration (0 allocations):
zero!(bsm)
block(bsm, 1, 1) .= [4.0 1.0; 1.0 5.0]
block(bsm, 2, 1) .= [0.5 0.2; -0.1 0.3]
block(bsm, 2, 2) .= [6.0 0.0; 0.0 7.0]

@inbounds for (k, si) in enumerate(scatter_indices)
    H_csc.nzval[k] = bsm.data[si]
end
```

## Extracting Principal Submatrices for Schur Complement Solvers

In bipartite problems such as Bundle Adjustment, variables partition into camera blocks $C$ and landmark blocks $L$. [`symmetrifyfull!`](@ref) accepts an optional `blocks` index vector to extract and symmetrify only a selected subset of blocks into a preallocated dense matrix:

```julia
# Extract only block 1 into a 2x2 dense matrix:
H_sub = zeros(Float64, 2, 2)
symmetrifyfull!(H_sub, bsm, [1])
```

## Choosing Between Sparse and Dense Representations

For small or densely connected graphs, dense Cholesky factorization outperforms sparse factorization due to lower indexing overhead and BLAS/SIMD efficiency. [`block_sparse_nnz`](@ref) computes the number of scalar non-zeros implied by a block sparsity pattern, which a solver can use to choose between a [`BlockSparseMatrix`](@ref) and a [`BlockDenseMatrix`](@ref):

```julia
blocksizes = [2, 2]
nnz_lower = block_sparse_nnz(sparsity_t, blocksizes) # 2*2 + 2*2 + 2*2 = 12
```
