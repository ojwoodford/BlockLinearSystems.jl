# BlockMatrices.jl

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://ojwoodford.github.io/BlockMatrices.jl/dev/)
[![Build Status](https://github.com/ojwoodford/BlockMatrices.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ojwoodford/BlockMatrices.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/ojwoodford/BlockMatrices.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ojwoodford/BlockMatrices.jl)

High-performance block-structured dense (`BlockDenseMatrix`) and sparse (`BlockSparseMatrix`) matrices in Julia, designed for zero-allocation assembly of Jacobians, Gauss-Newton / second-order Hessians, and Schur complement systems.

## Features

- **`BlockSparseMatrix{T}`**: Contiguous column-major storage of each non-zero block inside a single flat `Vector{T}`, eliminating pointer chasing during block accumulation.
- **`BlockDenseMatrix{T}`**: Dense `Matrix{T}` partitioned by variable block offsets, returning stack-wrapped `SizedMatrix` or `HybridArray` views when indexed with `StaticInt` dimensions.
- **Fast Conversion & Symmetrification**: Precompute gather indices once with `makesparseindices(bsm, true)` to scatter lower-triangular block-sparse accumulators into symmetric `SparseMatrixCSC` matrices with zero allocations.
- **Submatrix & Schur Support**: Extract and symmetrify principal block submatrices in-place via `symmetrifyfull!(mat, bsm, blocks)`.

## Quick Start

```julia
using BlockMatrices, SparseArrays, Static, StaticArrays

# 1. BlockDenseMatrix with blocks of sizes [2, 3] (offsets [1, 3, 6])
bdm = BlockDenseMatrix{Float64}([1, 3, 6])
block(bdm, 1, 1, static(2), static(2)) .= @SMatrix [2.0 0.5; 0.5 3.0]
block(bdm, 2, 1, static(3), static(2)) .= @SMatrix [1.0 0.0; 0.0 1.0; -1.0 1.0]
block(bdm, 2, 2, static(3), static(3)) .= @SMatrix [4.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 4.0]
H_dense = symmetrifyfull(bdm)

# 2. BlockSparseMatrix (lower-triangular block pattern, stored transposed)
sparsity_t = sparse([true true; false true])
bsm = BlockSparseMatrix{Float64}(sparsity_t, [2, 3], [2, 3])
block(bsm, 1, 1) .= [2.0 0.5; 0.5 3.0]
block(bsm, 2, 1) .= [1.0 0.0; 0.0 1.0; -1.0 1.0]
block(bsm, 2, 2) .= [4.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 4.0]
H_sparse = symmetrifysparse(bsm)
```
