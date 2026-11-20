# BlockLinearSystems.jl

[![docs-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ojwoodford.github.io/BlockLinearSystems.jl/stable/)
[![docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ojwoodford.github.io/BlockLinearSystems.jl/dev/)
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
- **Schur Complement Reduction**: `SchurStructure`, `SchurBlockHessian` and the parallel `reduce_schur_dense!` / `reduce_schur_sparse!` kernels eliminate blocks (any dimension, 1–65535) from bipartite systems, with dense or sparse reduced systems. `symmetrifyfull!(dest, bsm, blocks)` extracts principal block submatrices in-place.
- **QR Elimination (Jacobian path)**: `SchurJacobianStructure`, `SchurBlockJacobian` and `reduce_qr!` eliminate the same blocks from a least-squares problem `min ‖J x − r‖²` by QR, without forming `JᵀJ`, so the elimination works at the condition number of `J` rather than its square.
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
sizes = [6, 6, 3]
block_coords = [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3)]
bsm = SparseBlockLinearSystem{Float64}(block_coords, sizes)

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
# Extract only the submatrix of blocks 1 and 2 of `bsm` (from example 1) in-place:
H_sub = Matrix{Float64}(undef, 12, 12)
symmetrifyfull!(H_sub, bsm, [1, 2])
```

### 4. Elimination (Schur Complement Reduction)

Describe the problem in your own block numbering (the size of each block of unknowns, which blocks each group of
terms couples, which blocks to eliminate); `SchurStructure` lays it out, `SchurBlockHessian` allocates it, and
`addblock!` fills it by naming pairs of blocks. The linear system that reduces it, solves the small reduced system
and back-substitutes is [DampedLinearSolve.jl](https://github.com/ojwoodford/DampedLinearSolve.jl)'s contract,
which this package implements (DampedLinearSolve is a dependency):

```julia
using BlockLinearSystems, DampedLinearSolve, LinearAlgebra

# Blocks 1-3 retained (6 each), 4-7 eliminated (3, 3, 3, 1); each row block couples an eliminated block to retained ones
st = SchurStructure([6, 6, 6, 3, 3, 3, 1], [[4, 1, 2], [5, 1, 2, 3], [6, 2, 3], [7, 3]];
                    eliminated = [false, false, false, true, true, true, true])
sizes = blocksizes(st)                          # in layout order: retained blocks first
off = computestartindices(sizes); n = sum(sizes)
rg(p) = off[p]:off[p] + sizes[p] - 1           # the scalar range of layout block p
lay = invperm(blockorder(st))                   # the layout number of each of the caller's blocks
coupled = ((4, (1, 2)), (5, (1, 2, 3)), (6, (2, 3)), (7, (3,)))
# A diagonally dominant symmetric matrix in layout order, as the contributions to add
B = Matrix(Diagonal([p <= nretained(st) ? 10.0 : 4.0 for p in 1:nblocks(st) for _ in 1:sizes[p]]))
for (e, rs) in coupled, r in rs
    B[rg(lay[r]), rg(lay[e])] .= 0.1; B[rg(lay[e]), rg(lay[r])] .= 0.1
end
H = SchurBlockHessian{Float64, Matrix{Float64}}(st)
# Fill it by block: every block with itself, and each coupled pair once (in either order)
foreach(p -> addblock!(H, p, p, B, rg(p), rg(p)), 1:nblocks(st))
for (e, rs) in coupled, r in rs
    addblock!(H, lay[r], lay[e], B, rg(lay[r]), rg(lay[e]))
end

ls = makehessiansystem(H, :auto)
getb(ls) .= ones(n)                             # the right-hand side, in layout order
dampen!(ls, 0.1, ones(n))                       # damping lambda * diag(scaling)
solve!(ls, :auto)                               # Success; NotPositiveDefinite if a block is singular
x = getx(ls)                                    # [x_r; x_e], in layout order
```

A `SparseBlockLinearSystem` retained part (`SchurBlockHessian{Float64, SparseBlockLinearSystem{Float64}}(st)`) gives
a sparse reduced system, on the pattern of `build_schur_pattern(st)`.

See the [Schur reduction tutorial](https://ojwoodford.github.io/BlockLinearSystems.jl/dev/tutorials/schur_reduction/) for a complete, runnable example.

### 5. QR Elimination (Jacobian Path)

This API is new in this release and may change in the next minor version.

The same elimination, with residuals and a Jacobian instead of a Hessian: one dense table `[J_r | J_e | r]` per
eliminated block, with the rows that involve it. The elimination applies the Householder QR of each table's
eliminated columns to the whole table; the remaining rows of every table form the reduced problem
`min ‖J̃ x_r − r̃‖²`, with `J̃ᵀJ̃` equal to the Schur complement of `JᵀJ`. The description adds the number of rows of
each row block, and the Jacobian is filled by row block:

```julia
using BlockLinearSystems, DampedLinearSolve, LinearAlgebra

rowblocks, rows = [[4, 1, 2], [5, 1, 2, 3], [6, 2, 3], [7, 3]], [8, 10, 6, 3]
jst = SchurJacobianStructure([6, 6, 6, 3, 3, 3, 1], rowblocks, rows;
                             eliminated = [false, false, false, true, true, true, true])
J = SchurBlockJacobian{Float64, Matrix{Float64}}(jst)   # reduced-system storage: Matrix{T} or SparseMatrixCSC{T, Int}
sizes, lay = blocksizes(jst), invperm(blockorder(jst))
for (k, rb) in enumerate(rowblocks)
    for b in rb                                 # row block k's rows in block b's columns
        block(J, k, blockentry(jst, k, lay[b]), rows[k], sizes[lay[b]]) .= randn.()
    end
    residual_block(J, k) .= randn.()            # ... and in the residual
end
n = sum(sizes)

ls = makejacobiansystem(J, n, :qr)              # QR of the reduced Jacobian; or :cholesky, :ldl, :lsmr (with Krylov)
DampedLinearSolve.gradient!(getb(ls), getA(ls)) # b = Jᵀr
markstale!(getA(ls))                            # the tables are new
dampen!(ls, 0.1, ones(n))
solve!(ls, :qr)
x_r, x_e = getx(ls)[1:sum(sizes[1:nretained(jst)])], getx(ls)[sum(sizes[1:nretained(jst)])+1:end]
```

**Memory caveat:** each table is dense in every retained block coupled to its eliminated block, so its size
is (rows + d_e) × (coupled retained dimension + d_e + 1), stored twice (raw and factored). For eliminated
blocks seen by many residuals this is far more than the Schur route needs. See the
[QR elimination tutorial](https://ojwoodford.github.io/BlockLinearSystems.jl/dev/tutorials/qr_elimination/).

### Use with DampedLinearSolve.jl and SecondOrderOptimization.jl

This package implements the linear-system contract of [DampedLinearSolve.jl](https://github.com/ojwoodford/DampedLinearSolve.jl),
a dependency, for its Schur-complement and QR-eliminated systems, so that
[SecondOrderOptimization.jl](https://github.com/ojwoodford/SecondOrderOptimization.jl) and any other optimizer written
against the contract run on them; nothing else has to be loaded.
Passing a `SchurBlockHessian` as the Hessian prototype selects elimination; there is no option to set.
Whether the reduced system is dense or sparse follows the storage of `R` (a `Matrix` or a `SparseBlockLinearSystem`),
and `linsolver` names the solver applied to the reduced system (`:auto`, `:cholesky` or `:ldl`).
