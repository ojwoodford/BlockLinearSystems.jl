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
