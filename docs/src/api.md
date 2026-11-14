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
SchurBlockMatrix
SchurReductionWorkspace
trial_workspace
reset_schur_dense!
reset_schur_sparse!
reduce_schur_dense!
reduce_schur_sparse!
backsubstitute!
LinearAlgebra.dot(::AbstractVector{<:Real}, ::SchurBlockMatrix{<:Real}, ::AbstractVector{<:Real})
retained_block
```

### Reduced-System Sparsity

```@docs
build_schur_pattern
retained_block_pairs
build_r_to_s_map
```

### Variable Ordering

```@docs
sort_schur_blocks
```

## QR Elimination (Jacobian Path)

The Jacobian-path counterpart of the Schur reduction: elimination of the same blocks from a tall
least-squares problem `min ‖J x - r‖²` by QR. See the [QR elimination tutorial](tutorials/qr_elimination.md).

```@docs
SchurJacobianStructure
SchurBlockJacobian
schur_jacobian_bytes
coupling_block
eliminated_block
residual_block
coupling_entry
QRReductionWorkspace
reduce_qr!
reduced_nrows
free_block_pairs
reduced_mul!
reduced_tmul!
reduced_residual!
reduced_jacobian!
reduced_normal_dense!
reduced_normal_sparse!
reduced_diagonal_blocks!
jtr!
jtj_diag!
jtj_quadform
```
