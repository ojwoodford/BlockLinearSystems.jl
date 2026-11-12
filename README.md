# BlockLinearSystems.jl

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://ojwoodford.github.io/BlockLinearSystems.jl/dev/)
[![Build Status](https://github.com/ojwoodford/BlockLinearSystems.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ojwoodford/BlockLinearSystems.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/ojwoodford/BlockLinearSystems.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ojwoodford/BlockLinearSystems.jl)

High-performance block-structured dense (`DenseBlockLinearSystem`) and sparse (`SparseBlockLinearSystem`) linear system accumulators in Julia, designed for zero-allocation assembly of Jacobians, Gauss-Newton / Levenberg-Marquardt Hessians, and Schur complement systems in nonlinear optimization and factor graphs.

---

## Why BlockLinearSystems.jl?

In nonlinear least squares (NLS), bundle adjustment (BA), and SLAM, optimization problems are naturally organized around variable blocks of different sizes (e.g. blocks of 6 and 3 variables). Accumulating cost factor Hessian contributions into a standard `SparseMatrixCSC` during optimization requires costly searches across compressed column pointers, causing significant allocation and cache penalties.

`BlockLinearSystems.jl` provides dedicated block linear system accumulators specifically tuned for this workflow:
- **Contiguous Block Storage**: Each non-zero block $(i, j)$ is stored contiguously in column-major order in a flat buffer, making block accumulation a single contiguous memory write.
- **Zero-Allocation Views**: Accessing blocks with static dimensions (`static(N)` or `Val(N)`) returns stack-allocated `SizedMatrix` or `HybridArray` views, enabling unrolled SIMD addition with StaticArrays (`.+=`).
- **Zero-Allocation Solver Scatter**: In iterative solvers, `sparse_with_map` precomputes the CSC pattern and an integer scatter map once during setup. In each iteration, `copyto!(csc_mat, bsm, scatter_map)` updates the sparse solver matrix with **zero allocations**.
- **Schur Complement Reduction**: `SchurStructure`, `SchurBlockMatrix` and the parallel `reduce_schur_dense!` / `reduce_schur_sparse!` kernels eliminate blocks (any dimension, 1–255) from bipartite systems, with dense or sparse reduced systems. `symmetrifyfull!(dest, bsm, blocks)` extracts principal block submatrices in-place.
- **Thread-Local Clones**: `copy_structure(sys)` creates a lightweight thread-local accumulator sharing index structures, reduced after assembly via a fast BLAS vector addition (`sys.data .+= thread_sys.data`).

### Design Scope: Accumulators, Not General Matrices

`DenseBlockLinearSystem` and `SparseBlockLinearSystem` intentionally do **not** subtype `AbstractMatrix` or implement generic linear algebra (`mul!`, `*`, scalar iteration). This package is strictly an ultra-fast block accumulator and Schur assembler. Once accumulated, systems are converted or scattered into standard `Matrix` or `SparseMatrixCSC` formats for factorization with SuiteSparse, LDLFactorizations, or BLAS.

---

## Comparison

| Capability | Standard `SparseMatrixCSC` | Standard `Matrix` | `SparseBlockLinearSystem` | `DenseBlockLinearSystem` |
| :--- | :--- | :--- | :--- | :--- |
| **Block Access Overhead** | High (search per element) | SubArray / slice allocation | **Zero allocations** (`SizedMatrix`/`HybridArray`) | **Zero allocations** (`SizedMatrix`/`HybridArray`) |
| **Block Write Locality** | Strided across columns | Strided across rows/columns | **Contiguous slice in `Vector{T}`** | Contiguous column slices |
| **Multi-Threaded Assembly** | Complex non-zero merge | High memory footprint | **Flat `copy_structure` + vector reduction** | Flat buffer reduction |
| **Iterative Solver Scatter** | Full re-assembly or nzval math | N/A | **Zero allocations** (`copyto!`) | Zero allocations (direct buffer) |
| **Schur Subsystem Extraction** | Expensive sparse matrix indexing | Dense matrix slicing | **$O(\text{nnz}_C)$ in-place submatrix extraction** | Submatrix copy |

---

## Quick Start

### 1. Zero-Allocation Optimization Loop (Sparse)

```julia
using BlockLinearSystems, SparseArrays, Static, StaticArrays

# 1. Define block sizes and block sparsity pattern (lower-triangular)
#    Blocks: 2 blocks of dim 6, 1 block of dim 3
blocksizes = [6, 6, 3]
block_coords = [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3)]
bsm = SparseBlockLinearSystem{Float64}(block_coords, blocksizes)

# 2. Setup solver matrix and scatter map once (setup phase)
H_csc, scatter_map = sparse_with_map(bsm; symmetric=true)

# 3. Inside iterative solver loop (0 allocations):
zero!(bsm)

# Accumulate pose-pose block using static dimensions
H_pose = @SMatrix rand(6, 6)
block(bsm, 1, 1, Val(6), Val(6)) .+= H_pose

# Accumulate an off-diagonal block using static dimensions
H_meas = @SMatrix rand(3, 6)
block(bsm, 3, 1, Val(3), Val(6)) .+= H_meas

# Scatter into CSC matrix with exactly 0 allocations:
copyto!(H_csc, bsm, scatter_map)
# Now pass H_csc to your sparse Cholesky / LDLᵀ solver!
```

### 2. Dense Block System

```julia
using BlockLinearSystems, Static

# Construct a dense system partitioned into blocks of sizes [2, 3]
bdm = DenseBlockLinearSystem{Float64}([2, 3])

# Zero-allocation block indexing
block(bdm, 1, 1, Val(2), Val(2)) .= [2.0 0.5; 0.5 3.0]
block(bdm, 2, 1, Val(3), Val(2)) .= [1.0 0.0; 0.0 1.0; -1.0 1.0]
block(bdm, 2, 2, Val(3), Val(3)) .= [4.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 4.0]

# Symmetrify in-place: mirrors lower triangle to upper triangle
symmetrifyfull!(bdm)
```

### 3. Schur Complement Submatrix Extraction

```julia
# Extract only the submatrix of blocks 1 and 2 in-place:
H_sub = Matrix{Float64}(undef, 12, 12)
symmetrifyfull!(H_sub, bsm, [1, 2])
```

### 4. Schur Complement Reduction

```julia
st = SchurStructure([6, 6, 6], [3, 3, 3, 1], [[1, 2], [1, 2, 3], [2, 3], [3]])  # retained block dims, eliminated block dims, retained blocks coupled to each eliminated block
ws = SchurReductionWorkspace{Float64}(st, 1)
# ... fill B, C_data, E_data in a SchurBlockMatrix, then:
reset_schur_dense!(S, B, scaling_c, lambda, ws)
reduce_schur_dense!(S, g, b_c, b_p, scaling_p, lambda, st, C_data, E_data, ws)   # false if a block is singular
# solve S x_c = g, then recover the eliminated variables:
backsubstitute!(x_p, x_c, st, E_data, ws.C_inv_data, ws.z_data)
```

See the [Schur reduction tutorial](https://ojwoodford.github.io/BlockLinearSystems.jl/dev/tutorials/schur_reduction/) for a complete, runnable example.

### Use with SecondOrderOptimization.jl

Loading `BlockLinearSystems` together with [SecondOrderOptimization.jl](https://github.com/ojwoodford/SecondOrderOptimization.jl)
activates an extension (defined in that package) with `DenseSchurLinearSystem` and `SparseSchurLinearSystem`,
selected with `linsolver = :schur`, `:dense_schur` or `:sparse_schur`.
