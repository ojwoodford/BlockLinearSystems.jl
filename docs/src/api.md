# API Reference

## Linear System Types

```@docs
SparseBlockLinearSystem
DenseBlockLinearSystem
```

## Block Indexing & Mutation

```@docs
block
validblock
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

## Sparsity & Offset Utilities

```@docs
block_sparse_nnz
computestartindices
find_diag_inds
inplacecumsum!
```
