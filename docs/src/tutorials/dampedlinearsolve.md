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
| [`SchurBlockJacobian`](@ref)`{T, Matrix{T}}` | QR elimination, dense reduced system | `:auto`, `:qr` (QR of the reduced Jacobian); `:cholesky`, `:ldl` (its normal equations); `:lsmr` (with Krylov.jl) |
| [`SchurBlockJacobian`](@ref)`{T, SparseMatrixCSC{T, Int}}` | QR elimination, sparse reduced system | `:auto`, `:qr` (SPQR of the reduced Jacobian); `:cholesky`, `:ldl` (its normal equations); `:lsmr` (with Krylov.jl) |

**Element type.** A `SchurBlockHessian{T}` gives a Schur system in `T`, `Float64` or `Float32`: the
Hessian, its reduction, the reduced system and its factorization are held in `T`, while `b`, `x`, the damping
and every query are `Float64` (DampedLinearSolve's `systemeltype` gives `T`). Of the sparse reduced solvers,
`:ldl` offers `Float32` on every Julia version and `:auto` and `:cholesky` (CHOLMOD) only where SparseArrays
factorizes in single precision (1.13, not 1.10); a solver that does not offer it is an `ArgumentError` when the
system is built. A `SchurBlockJacobian{T}` gives a QR-eliminated system in `T` likewise: the tables, their
elimination and the reduced solve in `T`; the residual (a `Float64` vector of its own, beside the tables), the
gradient `Jᵀr`, `x` and every query in `Float64`. With the dense reduced storage every solver offers `Float32`;
with the sparse one SPQR does not (`:auto`, `:qr`), `:cholesky` only where CHOLMOD factorizes in single precision,
and `:ldl` and `:lsmr` always.

**Conditioning.** Every system but LSMR's conditions the damped problem before the elimination, as
DampedLinearSolve's systems do (its contract's "Conditioning"): with `S = Diagonal(s)` over all the variables,
retained and eliminated, the reduction, the reduced solve and the back-substitution run on the scaled system
(`S H S` and `S b`; on the Jacobian side the columns of `J` scaled, with the damping in the scaled variables) and
`x = S y` is returned. `s_j` is the power of two nearest to `1/(1e-12 + √|d_j + λ₀ D_jj|)`, `d_j` the diagonal as
filled (on the Jacobian side the squared column norms) and `λ₀ D` the first solve's damping after a fill, reused
for the fill's later dampings and recomputed when one has moved far from it (the scaled damped diagonal outside
`[2^-16, 2^16]`). The scale is applied where the kernels read the stored values, so `H`, `J`, `b` and `r` are left as
filled and nothing more is held or allocated. Powers of two make it exact: what it changes is the range of the entries
(in `Float32`) and the decisions made by comparing magnitudes, such as the sparse reduced solvers' indefiniteness
test. The `1e-12` and the `2^16` are starting values.

**A limit of `Float32` elimination.** The `Float32` QR-eliminated solve is not refined, and where the residual at the
solution is large against the gradient its error is the `Float32` elimination's own, not removed by carrying `r` in
`Float64`: a relative error of 0.15 to 1.9 in the step on the package's test problems with `‖Jᵀr‖ = 1e-6·‖J‖·‖r‖`,
on every reduced storage and solver. An optimizer then takes rejected steps near the optimum. On bundle adjustments
of 10 × 300 to 60 × 5000 cameras and points (Levenberg–Marquardt, dense reduced storage, one thread), `Float32` took
one to three more iterations and 1.6 to 2.0 solves per iteration (`Float64`: 1.0 to 1.1), stopped on lack of progress
6e-5 to 1e-4 from the `Float64` solution, and spent 1.4 to 2.3 times as long in the linear solver, although each
solve was 22 % to 36 % faster. Iterative refinement in `Float64` removed the error but cost more than it saved, so it
is not done. Use `Float64` where the time matters, and `Float32` where the memory of the tables does.

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
