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
validblock
blocksizes
nblocks
blockdata
blockrange
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

## Internal Utilities

```@docs
BlockLinearSystems.computestartindices
BlockLinearSystems.find_diag_inds
BlockLinearSystems.inplacecumsum!
```
