# Block Matrix Assembly & Indexing

In factor-graph and nonlinear least squares problems, variables are naturally grouped into blocks $\mathbf{x}_1, \dots, \mathbf{x}_K$ of dimensions $d_1, \dots, d_K$, and each cost factor contributes derivatives to a small subset of block pairs $(i, j)$.

`BlockLinearSystems.jl` provides a unified interface across [`DenseBlockLinearSystem`](@ref) and [`SparseBlockLinearSystem`](@ref) so assembly code can remain agnostic to whether the underlying storage is dense or sparse.

## Constructing a `DenseBlockLinearSystem`

A [`DenseBlockLinearSystem`](@ref) automatically allocates a full dense `Matrix{T}` partitioned into given block dimensions:

```julia
using BlockLinearSystems, Static, StaticArrays

# Construct a block-square dense system with blocks of sizes 2, 3, 2
bdm = DenseBlockLinearSystem{Float64}([2, 3, 2])

# Or specify distinct row and column block sizes:
bdm_rect = DenseBlockLinearSystem{Float64}([2, 3], [3, 2])
```

If you already have precomputed 1-based start offsets, you can pass the explicit `offsets` keyword or wrapper:
```julia
bdm_offsets = DenseBlockLinearSystem{Float64}(; offsets=BlockOffsets([1, 3, 6, 8]))
```

## Constructing a `SparseBlockLinearSystem`

A [`SparseBlockLinearSystem`](@ref) stores only the structurally non-zero blocks in a contiguous `Vector{T}`.

You can construct it using standard natural orientation sparsity patterns or coordinate lists:

```julia
using BlockLinearSystems, SparseArrays

# Option A: Natural orientation lower-triangular pattern
sparsity = sparse([
    true  false false;
    true  true  false;
    false true  true
])
bsm = SparseBlockLinearSystem{Float64}(sparsity, [2, 3, 2])

# Option B: Block coordinate list [(row, col), ...]
coords = [(1, 1), (2, 1), (2, 2), (3, 2), (3, 3)]
bsm_coords = SparseBlockLinearSystem{Float64}(coords, [2, 3, 2])

# Option C: Pre-transposed pattern (fast path for performance-critical pipelines)
sparsity_t = TransposedPattern(sparse(sparsity'))
bsm_fast = SparseBlockLinearSystem{Float64}(sparsity_t, [2, 3, 2])
```

## Indexing Blocks with `block`

Use [`block`](@ref) to obtain a mutable view into block $(i, j)$:

- **Dynamic dimensions:** `block(M, i, j)` or `block(M, i, j, rows, cols)` returns a `SubArray` or `ReshapedArray` view.
- **Static dimensions:** `block(M, i, j, Val(R), Val(C))` or `block(M, i, j, static(R), static(C))` returns a `StaticArrays.SizedMatrix{R, C}` view, allowing unrolled SIMD addition `.+=` with `SMatrix{R, C}`.
- **Hybrid dimensions:** `block(M, i, j, Val(R), cols)` or `block(M, i, j, rows, Val(C))` returns a `HybridArrays.HybridArray` view when one dimension is known at compile time and the other is dynamic.

```julia
H_11 = @SMatrix [2.0 0.5; 0.5 3.0]
block(bsm, 1, 1, Val(2), Val(2)) .+= H_11
```

## Reusing Structure Across Threads

When parallelizing Hessian assembly across threads, [`copy_structure`](@ref) creates a lightweight clone that shares the immutable index arrays (`indices`, `indicestransposed`, `rowblocksizes`, `columnblocksizes`) while allocating a separate `data` vector per thread:

```julia
bsm_thread2 = copy_structure(bsm)
zero!(bsm_thread2)
```
After thread-local accumulation, the flat buffers can be reduced directly via `bsm.data .+= bsm_thread2.data`.
