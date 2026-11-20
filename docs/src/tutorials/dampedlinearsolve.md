# Solving with DampedLinearSolve.jl

[DampedLinearSolve.jl](https://github.com/ojwoodford/DampedLinearSolve.jl) defines the linear-system
contract a second-order optimizer uses: build a system from a prototype, fill it, damp it, solve it,
and query it (see its documentation, "The linear-system contract"). It is a dependency of this package,
which implements that contract for its eliminated systems, so that any optimizer written against the
contract (SecondOrderOptimization.jl, for one) runs on them unchanged; nothing else has to be loaded.

## Which system a prototype builds

| Prototype | System | `linsolver` for the reduced system |
|---|---|---|
| [`SchurBlockHessian`](@ref) with a dense `R` | dense Schur complement | `:auto`, `:cholesky` (Cholesky, falling back to Bunch–Kaufman), `:ldl` (Bunch–Kaufman) |
| [`SchurBlockHessian`](@ref) with a [`SparseBlockLinearSystem`](@ref) `R` | sparse Schur complement | `:auto`, `:cholesky` (CHOLMOD, falling back to LDLᵀ), `:ldl` (LDLᵀ) |
| [`SchurBlockJacobian`](@ref)`{Float64, Matrix{Float64}}` | QR elimination, dense reduced system | `:auto`, `:qr` (QR of the reduced Jacobian); `:cholesky`, `:ldl` (its normal equations); `:lsmr` (with Krylov.jl) |
| [`SchurBlockJacobian`](@ref)`{Float64, SparseMatrixCSC{Float64, Int}}` | QR elimination, sparse reduced system | `:auto`, `:qr` (SPQR of the reduced Jacobian); `:cholesky`, `:ldl` (its normal equations); `:lsmr` (with Krylov.jl) |

`makehessiansystem(H, linsolver)` builds the Schur systems and `makejacobiansystem(J, n, linsolver)` the
QR-eliminated ones; `checkhessian`, `checkjacobian` and `check_jacobian_linsolver` validate a
prototype first, and `structurekey` identifies its structure for a cache.

## How a solve goes

Each solve applies the recorded damping `λ·Diagonal(s)` during the reduction (to `R` and to each
eliminated block, so that the eliminated blocks are inverted damped), reduces to the retained
variables, solves the reduced system, and back-substitutes for the eliminated ones. The reduced
system is itself a DampedLinearSolve system wherever a standard one holds it exactly:

- a dense reduced matrix `S`, or dense reduced normal equations, is an undamped dense system;
- the dense reduced least-squares problem `[J̃; √(λ s_r)] x_r ≈ [r̃; 0]` is an undamped dense Jacobian
  system, so its single Householder QR is of the whole augmented matrix;
- the sparse reduced Jacobian is a sparse Jacobian system, damped by `λ s_r`, solved by SPQR with its
  symbolic analysis done once;
- LSMR on the reduced Jacobian (with Krylov.jl loaded) is a matrix-free reduced system of this
  package's Krylov extension, `KrylovExt`: `J̃` is applied through the factored tables and never formed.

A sparse reduced matrix `S` (the sparse Schur system, and the sparse reduced normal equations) is
solved by this package itself, with a persistent CHOLMOD copy and LDLFactorizations' upper-triangle
LDLᵀ: the reduction writes one triangle of `S`, and a DampedLinearSolve sparse system stores the
full symmetric matrix.

## An example

```julia
using BlockLinearSystems, DampedLinearSolve

st = SchurStructure(blocksizes, rowblocks; eliminated)   # the layout, from a description
H = SchurBlockHessian{Float64, Matrix{Float64}}(st)      # dense retained part
ls = makehessiansystem(H, :auto)
# fill getA(ls) (the SchurBlockHessian, with addblock!) and getb(ls), in layout order, then
dampen!(ls, 1e-3)
solve!(ls, :auto)        # Success; getx(ls) holds the solution, in layout order
```

## Conformance

Every one of these systems passes DampedLinearSolve's `check_linear_system` in this package's test
suite, for each solver it accepts; this package provides the conformance hooks
(`conformance_matrix`, `conformance_fill!`, `conformance_fillb!`, `conformance_traits`) for a
`SchurBlockHessian` and the tables of a `SchurBlockJacobian`.
