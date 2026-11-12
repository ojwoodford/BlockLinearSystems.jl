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
```

### Reduced-System Sparsity

```@docs
build_schur_pattern
retained_block_pairs
build_b_to_s_map
```

### Variable Ordering

```@docs
sort_schur_blocks
expand_block_perm
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
