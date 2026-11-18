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

## Schur Complement Reduction

Data structures and kernels for bipartite problems, where a few retained variable blocks are coupled
to many small eliminated variable blocks. See the [Schur reduction tutorial](tutorials/schur_reduction.md).

```@docs
SchurStructure
blockorder
nretained
SchurBlockHessian
SchurReductionWorkspace
trial_workspace
reset_schur_dense!
reset_schur_sparse!
reduce_schur_dense!
reduce_schur_sparse!
backsubstitute!
LinearAlgebra.dot(::AbstractVector{<:Real}, ::SchurBlockHessian{<:Real}, ::AbstractVector{<:Real})
addblock!
```

### Reduced-System Sparsity

```@docs
build_schur_pattern
retained_block_pairs
build_r_to_s_map
```

## QR Elimination (Jacobian Path)

The Jacobian-path counterpart of the Schur reduction: elimination of the same blocks from a tall
least-squares problem `min ‖J x - r‖²` by QR. See the [QR elimination tutorial](tutorials/qr_elimination.md).
`SchurBlockJacobian{T, RType}` carries the storage of the reduced system as its second type parameter
(`Matrix{T}` or `SparseMatrixCSC{T, Int}`), set by whoever builds it, as the type of `R` does for
`SchurBlockHessian{T, RType}`.

```@docs
SchurJacobianStructure
SchurBlockJacobian
schur_jacobian_bytes
residual_block
blockentry
QRReductionWorkspace
reduce_qr!
reduced_nrows
free_block_pairs
reduced_mul!
reduced_tmul!
reduced_residual!
reduced_jacobian!
build_reduced_jacobian_pattern
reduced_normal_dense!
reduced_normal_sparse!
reduced_diagonal_blocks!
factor_diagonal_blocks!
solve_diagonal_blocks!
jtr!
jtj_diag!
jtj_quadform
```

## Threading

The Schur and QR-elimination kernels thread through Polyester, a weak dependency: with
`using Polyester` (or any package that loads it, such as NLLSModel) they thread over their chunks when
more than one thread is available and there is enough work (`BlockLinearSystems.SCHUR_THREAD_MIN_WORK`),
and allocate nothing, serial or threaded. The exception is Julia 1.10: there a threaded call allocates a
few hundred bytes per chunk, because rebuilding the kernels' nested closures inside the threaded loop
allocates on that version (later versions do not; 1.13 is tested). Without Polyester the kernels run
serially. Their closures reach the threads through `threadargs` / `fromthreadargs`, public so that other
packages' `@batch` loops can use them too (they may move to Polyester later).

```@docs
threadargs
fromthreadargs
```
