# Changelog

All notable changes to `BlockLinearSystems.jl` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-08

### Changed (Breaking Changes)
- **Package Renamed**: Renamed from `BlockMatrices.jl` to `BlockLinearSystems.jl` to clearly communicate its intended purpose as a block linear system accumulator and Schur assembler rather than a general matrix library.
- **Core Structs Renamed**:
  - `BlockSparseMatrix{T}` -> `SparseBlockLinearSystem{T}`
  - `BlockDenseMatrix{T}` -> `DenseBlockLinearSystem{T}`
- **AbstractMatrix Subtyping Intentionally Omitted**: The structs do not inherit `AbstractMatrix` or implement generic linear algebra (`mul!`, `*`, scalar iteration). This avoids user expectations of general matrix operations and keeps block layout strictly specialized for fast zero-allocation block assembly and Schur elimination.
- **`sparse_dense_decision` Moved**: Relocated optimization-specific heuristic `sparse_dense_decision` out of this package into `NLLSModel.jl`.
- **Type Piracy Eliminated**: Removed `SparseArrays.nnz(aa::AbstractArray) = length(aa)`.
- **Eager Structural Transpose**: Replaced mutable `bsm.indices` cache (`cacheindices`/`clearindicescache`) with an eagerly constructed, immutable transpose upon construction, making `copy_structure` completely thread-safe.
- **Exceptions & Bounds Checking**: Replaced argument `@assert` statements with standard `ArgumentError`, `DimensionMismatch`, and `@boundscheck BoundsError` checks.
- **Symmetrify Mutation Semantics**:
  - `symmetrifyfull(M)` allocates and returns a new symmetric `Matrix{T}`.
  - `symmetrifyfull!(M)` mutates in-place (for `DenseBlockLinearSystem`) or writes into a preallocated target (for `SparseBlockLinearSystem`).

### Added
- **Constructor Unification**:
  - `DenseBlockLinearSystem{T}(rowblocksizes, colblocksizes=rowblocksizes)` constructs a dense system directly from block sizes without manual offset computation.
  - Added `BlockOffsets` wrapper and keyword argument `offsets=...` for explicit start offsets.
  - `SparseBlockLinearSystem{T}` accepts natural orientation sparsity patterns (`SparseMatrixCSC{Bool, Int}`) with `transposed=false` by default.
  - Added `TransposedPattern` wrapper for fast construction when sparsity pattern is already in block-row-major transposed orientation.
  - Added block coordinate constructor: `SparseBlockLinearSystem{T}(coords::Vector{Tuple{Int, Int}}, blocksizes)`.
- **Zero-Allocation CSC Scatter API**:
  - Added `sparse_with_map(sparse_sys; symmetric=true)` returning `(csc_mat, scatter_map)`.
  - Added non-allocating in-place updater `copyto!(csc_mat, sparse_sys, scatter_map)`.
- **Static Dimensions & View Helpers**:
  - Supported `Val(R), Val(C)` in addition to `StaticInt(R), StaticInt(C)` in `block(sys, i, j, Val(R), Val(C))` for zero-allocation `SizedMatrix` and `HybridArray` views.
- **Accessors**:
  - Added exported accessors: `blocksizes(sys, [dim])`, `nblocks(sys, [dim])`, `blockdata(sys)`, and `eachstoredblock(sys)` iterator. (Earlier versions of this entry listed a `blockrange(sys, dim, idx)` accessor that never existed; run ranges are available as `runblockrange(rle, k)`.)
- **Principal Submatrix Extraction**:
  - `symmetrifyfull!(mat, bsm, blocks)` supports both integer vectors and boolean masks (`BitVector`, `AbstractVector{Bool}`) for Schur complement systems.
- **Benchmarks & Documentation**:
  - Added `benchmark/benchmarks.jl` measuring block accumulation, zero-allocation CSC scatter, and Schur submatrix extraction.
  - Comprehensive documentation with Documenter, tutorials, and doctests.
  - Aqua and JET integration tests for code quality and type stability.

## [0.1.1] - 2026-11-04
- Initial release as `BlockMatrices.jl`.
