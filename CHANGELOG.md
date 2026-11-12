# Changelog

All notable changes to `BlockLinearSystems.jl` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Schur extension: trial systems no longer share the Hessian (B1).** `copy_structure` of the dense and sparse Schur linear systems allocates a new `SchurBlockMatrix` (new `copy_structure(::SchurBlockMatrix)`) and shares `x`, so linearizing at a trial point no longer overwrites the Hessian that rejected-step retries still use.
- **Schur damping is `lambda * scaling`, not `lambda * scaling^2` (B2).** This matches SecondOrderOptimization's `:levenbergmarquardt_dtd`. The unused `update_marquardt_scaling!` was removed. **Breaking** for callers of `reset_schur_*!`/`inv_sym_block` that passed square-rooted scalings.
- Sparse Schur Cholesky solve works on Julia 1.10 and 1.11, which lack `ldiv!(x, ::CHOLMOD.Factor, b)` (B3), and a non-positive-definite reduced system now falls back instead of throwing `PosDefException` (B5).
- The sparse reduced system has slots for retained-retained couplings that share no eliminated variable (B4): `build_schur_covisibility(structure; extra_blocks)` and `retained_block_pairs(B)`.
- `UInt8` overflow for blocks of 16x16 and larger (B7): `BlockRLE` size accessors (`getindex`, `runblocksize`, `eachrun`, `collect`) return `Int`, and `BlockRLE <: AbstractVector{Int}`. `block_sparse_nnz` previously returned a silently wrong count.
- `block` with run-time sizes is type stable and allocation free (B9, B14); new exported `blockrange(first, len)`.
- `quadratic_model_reduction` overrides for Schur systems (wrong sign, stale `lambda`) were removed in favour of SecondOrderOptimization's generic method (B10).
- `SchurStructure` validates dimensions, observation indices and duplicate observations (B13); `makelinearsystem` rejects contradictory `:dense_schur`/`:sparse_schur` requests, and `solve!` no longer clashes with the `OptimizationLinearSolver` method (B12).
- `BlockOffsets` that are unsorted, repeated or wider than 255 raise an `ArgumentError`; empty block lists build empty systems (M8). `BlockRLE{T}` no longer validates sizes of zero-length runs (M11). Removed a non-existent `blockrange(sys, dim, idx)` from the 0.2.0 notes (M10).

### Changed
- **Eliminated blocks may have any dimension from 1 to 255 (M5)**: static register arithmetic up to `D_STATIC = 8`, a LAPACK-based dynamic path above that.
- Performance: Cholesky is tried first for undamped solves and works in place for the dense system (B5); `fast_bAb` for sparse `B` is allocation free (B6); the covisibility pattern is built in one `sparse` call (B8) and can be passed to the sparse system constructor via `covis=`; `build_b_to_s_map` no longer scans columns (M6); back-substitution applies `C^-1` once per eliminated block (M2); the Schur reduction is two-phase (M3): landmarks are factored in parallel storing `W = C^-1 E'`, then chunks own disjoint ranges of retained column blocks and write straight into one `S`, so there are no per-chunk copies of `S` and no merge pass (**breaking**: `merge_dense_S!`/`merge_sparse_S!` and the workspace `thread_*` buffers are gone, and the per-group reduction kernels were replaced by `factor_landmarks_group!` and `accumulate_{dense,sparse}_columns!`); Schur loops run serially unless more than one thread is available and the work is large enough (B11), now using Polyester like the assembly code in NLLSModel.jl so the two share one threading mechanism; tiled dense `symmetrifyfull!` (M7); redundant block-size validation removed (M1).
- Element type and tolerances are no longer hard-coded to `Float64`/`1e-12` (M4); `SparseSchurLinearSystem` has a fourth type parameter for the CHOLMOD factor type.

### Moved
- **The SecondOrderOptimization extension now lives in SecondOrderOptimization.jl** (`ext/BlockLinearSystemsExt.jl`), which has `BlockLinearSystems` as a weak dependency. This package no longer has a weak dependency, extension or unregistered test dependency. The Schur API the extension builds on is public: `SchurBlockMatrix`, `SchurReductionWorkspace`, `trial_workspace`, `reset_schur_*!`, `reduce_schur_*!`, `backsubstitute!`, `build_schur_covisibility`, `build_b_to_s_map`, `retained_block_pairs`, `fast_bAb`, `copy_structure`.

### Added
- Schur tests (M12, T1) using only the public kernels: every Schur solve is compared with a dense Cholesky solve of the full damped system, including D'D scaling, retained-retained couplings, Float32 and eliminated blocks of every size class. The optimizer-facing linear systems are tested in SecondOrderOptimization.

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
