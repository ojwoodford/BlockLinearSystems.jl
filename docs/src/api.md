```@meta
CurrentModule = BlockLinearSystems
```

# API Reference

## Linear System Types

```@docs
SparseBlockLinearSystem
DenseBlockLinearSystem
TransposedPattern
```

## Block Indexing & Accessors

```@docs
block
blockrange
blocksizes
nblocks
zero!
copy_structure
uniformscaling!
```

## Conversion & Symmetrification

```@docs
symmetrifyfull
symmetrifyfull!
sparse_with_map
```

## Sparsity Statistics

```@docs
block_sparse_nnz
```

## Index Utilities

```@docs
computestartindices
find_diag_inds
```

## Run-Length Encoded Block Layouts

```@docs
BlockRLE
```

## Elimination

Bipartite problems, where a few retained variable blocks are coupled to many small eliminated variable
blocks: the caller describes the problem in its own block numbering, this package lays it out and stores it,
the caller fills it by block, and the linear system built from it by DampedLinearSolve's `makehessiansystem` or
`makejacobiansystem` eliminates the blocks, solves the reduced system and back-substitutes. See the
[Schur reduction](tutorials/schur_reduction.md) and [QR elimination](tutorials/qr_elimination.md) tutorials, and
[Solving with DampedLinearSolve.jl](tutorials/dampedlinearsolve.md).

### The layout

```@docs
SchurStructure
SchurJacobianStructure
blockorder
nretained
build_schur_pattern
```

### The Hessian path

```@docs
SchurBlockHessian
addblock!
LinearAlgebra.dot(::AbstractVector{<:Real}, ::SchurBlockHessian{<:Real}, ::AbstractVector{<:Real})
```

### The Jacobian path

`SchurBlockJacobian{T, RType}` carries the storage of the reduced system as its second type parameter
(`Matrix{T}` or `SparseMatrixCSC{T, Int}`), set by whoever builds it, as the type of `R` does for
`SchurBlockHessian{T, RType}`.

```@docs
SchurBlockJacobian
blockentry
residual_block
```

## Threading

The elimination's kernels thread through Polyester, a weak dependency: with
`using Polyester` (or any package that loads it, such as NLLSModel) they thread over their chunks when
more than one thread is available, the structure has more than one chunk (its `nchunks`) and enough work (its
`min_work`), and allocate nothing, serial or threaded. The exception is Julia 1.10: there a threaded call allocates a
few hundred bytes per chunk, because rebuilding the kernels' nested closures inside the threaded loop
allocates on that version (later versions do not; 1.13 is tested). Without Polyester the kernels run
serially. Their closures reach the threads through `threadargs` / `fromthreadargs`, public so that other
packages' `@batch` loops can use them too (they may move to Polyester later).

```@docs
threadargs
fromthreadargs
```
