# Block Matrix Assembly & Indexing

In factor-graph and nonlinear least squares problems, variables are naturally grouped into blocks $\mathbf{x}_1, \dots, \mathbf{x}_K$ of dimensions $d_1, \dots, d_K$, and each cost factor contributes derivatives to a small subset of block pairs $(i, j)$.

`BlockLinearSystems.jl` provides a unified interface across [`DenseBlockLinearSystem`](@ref) and [`SparseBlockLinearSystem`](@ref) so assembly code can remain agnostic to whether the underlying storage is dense or sparse.

## Constructing a `DenseBlockLinearSystem`

A [`DenseBlockLinearSystem`](@ref) wraps a standard `Matrix{T}` along with 1-based block start offsets terminated by `total_dim + 1`:

```julia
using BlockLinearSystems, Static, StaticArrays

blocksizes = [2, 3, 2]
offsets = computestartindices(blocksizes)
push!(offsets, offsets[end] + blocksizes[end]) # [1, 3, 6, 8]

bdm = DenseBlockLinearSystem{Float64}(offsets)
```

## Constructing a `SparseBlockLinearSystem`

A [`SparseBlockLinearSystem`](@ref) stores only the structurally non-zero blocks in a contiguous `Vector{T}`. Because blocks are laid out row-by-row for cache locality, the constructor accepts the **transposed** block sparsity pattern (`sparsitytransposed::SparseMatrixCSC{Bool, Int}`):

```julia
using BlockLinearSystems, SparseArrays

# For a lower-triangular 3x3 block matrix where blocks (1,1), (2,1), (2,2), (3,2), (3,3) are non-zero,
# the transposed sparsity pattern is upper-triangular:
sparsity_t = sparse([
    true  true  false;
    false true  true;
    false false true
])

bsm = SparseBlockLinearSystem{Float64}(sparsity_t, [2, 3, 2], [2, 3, 2])
```

## Indexing Blocks with `block`

Use [`block`](@ref) to obtain a mutable view into block $(i, j)$:

- **Dynamic dimensions:** `block(M, i, j)` or `block(M, i, j, rows, cols)` returns a `SubArray` or `ReshapedArray` view.
- **Static dimensions:** `block(M, i, j, static(R), static(C))` returns a `StaticArrays.SizedMatrix{R, C}` view, allowing unrolled SIMD addition `.+=` with `SMatrix{R, C}`.
- **Hybrid dimensions:** `block(M, i, j, static(R), cols)` or `block(M, i, j, rows, static(C))` returns a `HybridArrays.HybridArray` view when one dimension is known at compile time and the other is dynamic.

```julia
H_11 = @SMatrix [2.0 0.5; 0.5 3.0]
block(bsm, 1, 1, static(2), static(2)) .+= H_11
```

## Reusing Structure Across Threads

When parallelizing Hessian assembly across threads, [`copy_structure`](@ref) creates a lightweight clone that shares the immutable index arrays (`indices`, `indicestransposed`, `rowblocksizes`, `columnblocksizes`) while allocating a separate `data` vector per thread:

```julia
bsm_thread2 = copy_structure(bsm)
zero!(bsm_thread2)
```
After thread-local accumulation, the flat buffers can be reduced directly via `bsm.data .+= bsm_thread2.data`.
