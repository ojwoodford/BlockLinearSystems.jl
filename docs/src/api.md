# API Reference

## Linear System Types

```@docs
SparseBlockLinearSystem
DenseBlockLinearSystem
TransposedPattern
BlockOffsets
```

## Block Indexing & Accessors

```@docs
block
blockrange
validblock
blocksizes
nblocks
blockdata
eachstoredblock
zero!
copy_structure
uniformscaling!
```

## Conversion & Symmetrification

```@docs
symmetrifyfull
symmetrifyfull!
symmetrifysparse
makesparseindices
sparse_with_map
```

## Sparsity Statistics

```@docs
block_sparse_nnz
```

## Run-Length Encoded Block Arrangements

```@docs
BlockRLE
nruns
runlength
runblocksize
matrixdimension
runblockrange
runmatrixrange
eachrun
sort_blocks_hierarchical
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
fast_bAb
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
expand_block_perm
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
```

## Internal Utilities

```@docs
BlockLinearSystems.computestartindices
BlockLinearSystems.find_diag_inds
BlockLinearSystems.inplacecumsum!
```

### Schur Reduction Kernels

```@docs
BlockLinearSystems.factor_eliminated!
BlockLinearSystems.factor_eliminated_group!
BlockLinearSystems.accumulate_dense_columns!
BlockLinearSystems.accumulate_sparse_columns!
BlockLinearSystems.partition_columns
```
