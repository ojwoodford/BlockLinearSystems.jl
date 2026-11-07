# API Reference

## Matrix Types

```@docs
BlockSparseMatrix
BlockDenseMatrix
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
cacheindices
clearindicescache
updatelastrowsym!
```

## Sparsity & Offset Utilities

```@docs
block_sparse_nnz
computestartindices
find_diag_inds
```
