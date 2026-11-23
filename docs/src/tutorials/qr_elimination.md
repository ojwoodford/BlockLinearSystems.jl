# QR Elimination (Jacobian Path)

The [Schur complement](schur_reduction.md) eliminates independent variable blocks from the normal
equations `H x = b`. When the problem is available as residuals and a Jacobian, the same blocks can be
eliminated from the least-squares problem itself, by QR, without forming `H = JᵀJ`. This is the method of
Demmel, Sommer, Cremers and Usenko, "Square Root Bundle Adjustment for Large-Scale Reconstruction"
(CVPR 2021). Nothing here is specific to bundle adjustment.

!!! note "New API"
    The QR-elimination API is new in this release and may change in the next minor version.

## Conventions

The same as for the Schur systems, so that the two routes give the same numbers:

- The problem is `min ‖J x − r‖²`, described as on the Hessian path, with one more number per row block: its
  number of rows. [`SchurJacobianStructure`](@ref)`(blocksizes, rowblocks, rowblocksizes; eliminated)` lays it
  out in the same order as [`SchurStructure`](@ref) does, retained blocks first, so `x` is in that layout order.
  Its normal equations are `JᵀJ x = Jᵀr`, i.e. `[R Cᵀ; C E] x = b` with `H = JᵀJ` and `b = Jᵀr`. Callers that
  want a descent step from `f(x0 + dx) ≈ f(x0) + J dx` store `r = f(x0)` and negate `x`, as with the Schur
  systems.
- Each row block touches at most one eliminated block (the Schur condition).
- Levenberg–Marquardt damping solves `(JᵀJ + λ·Diagonal(s)) x = Jᵀr`, with `s` in the units of `diag(JᵀJ)` (for
  Marquardt scaling, `s = diag(JᵀJ)`), not squared.

## The table layout

The rows that involve eliminated block `e` (the row blocks that touch it) are stored as one dense, column-major
table, with the columns of every retained block coupled to `e`, then `e`'s own columns, then the residual. `d_e`
damping rows sit underneath, zero in the raw storage. The raw storage keeps the residual column apart, in a
`Float64` vector of its own (`J.r`, a column per table), so that it and the gradient stay in double precision
whatever the tables' element type. The elimination keeps it apart too: its working copy of the tables has the
raw layout, and the residual goes through the elimination in a `Float64` vector of the same layout, the
reflectors applied to it in `Float64`, so that the back-substitution's `q` and the reduced residual `r̃` are
`Float64` too:

```
                 retained blocks coupled to e      e     r
               ┌───────────┬──────────────────┬───────┬───┐
  m_e data     │   J_r1    │   J_r2   …       │  J_e  │ r │
  rows         │           │                  │       │   │
               ├───────────┼──────────────────┼───────┼───┤
  d_e damping  │     0     │        0         │ √λs_e │ 0 │   written by the elimination
               └───────────┴──────────────────┴───────┴───┘
```

The elimination copies each table, writes the damping rows and applies the Householder QR of its eliminated
columns to the whole table:

```
          Qᵀ ·  [ J_r | J_e | r ]   =   [ B   | U | q   ]    d_e rows:   U x_e + B x_r = q
                [  0  | √λD |  0 ]       [ J̃_e | 0 | r̃_e ]    m_e rows:   J̃_e x_r = r̃_e
```

The top rows give `x_e = U⁻¹ (q − B x_r)` once `x_r` is known. The bottom rows do not involve `x_e`. Together
with the rows that involve no eliminated block (*free* rows: the row blocks that touch only retained blocks, one
table per set of retained blocks), they form the reduced problem `min ‖J̃ x_r − r̃‖²`, whose normal equations are
exactly the Schur complement: `J̃ᵀJ̃ = R − Cᵀ E⁻¹ C` and `J̃ᵀr̃ = b_r − Cᵀ E⁻¹ b_e`, with `E` including its damping.
The retained part of the damping is added by the reduced solve.

## A complete example

The structure of the Schur tutorial, as a Jacobian: three retained blocks, six eliminated blocks, each row block
a few rows that touch one eliminated block and the retained blocks it is coupled to, and one more row block of
12 rows that touches only two retained blocks (a free group):

```@example qr
using BlockLinearSystems, DampedLinearSolve, LinearAlgebra, SparseArrays, Random

rng = Random.Xoshiro(1)
sizes0 = [3, 6, 3, 6, 3, 6, 3, 2, 1]            # column blocks, in the caller's order
elim = [true, false, true, false, true, false, true, true, true]
rowblocks = [[1, 2, 4], [3, 2, 4, 6], [5, 4, 6], [7, 2, 6], [8, 2, 4], [9, 6], [2, 6]]
rows = [8, 10, 6, 6, 4, 3, 12]
jst = SchurJacobianStructure(sizes0, rowblocks, rows; eliminated = elim)
J = SchurBlockJacobian{Float64, Matrix{Float64}}(jst)   # dense reduced system (see below)
nothing # hide
```

Fill it by row block: [`blockentry`](@ref) gives, once, a handle for each (row block, column block) to be
written; [`block`](@ref)`(J, k, entry, nrows, ncols)` is the view of row block `k`'s rows in that block's columns,
and [`residual_block`](@ref)`(J, k)` its rows of the residual. A dense copy is kept for checking, its rows in
row-block order:

```@example qr
sizes = blocksizes(jst)
off = computestartindices(sizes); n = sum(sizes)
lay = invperm(blockorder(jst))                  # the layout number of each of the caller's blocks
Jd, rd = zeros(sum(rows), n), zeros(sum(rows))
row0 = 0
for (k, rb) in enumerate(rowblocks)
    m = rows[k]
    for c in rb
        p = lay[c]
        B = randn(rng, m, sizes[p])
        block(J, k, blockentry(jst, k, p), m, sizes[p]) .= B
        Jd[row0 .+ (1:m), off[p] .+ (0:sizes[p]-1)] .= B
    end
    rk = randn(rng, m)
    residual_block(J, k) .= rk
    rd[row0 .+ (1:m)] .= rk
    global row0 += m
end
size(Jd)
```

Solve through DampedLinearSolve's contract: build the system from the prototype (it reads `J` in place), compute
its gradient `Jᵀr` and mark the tables as new, damp and solve. With `λ = 0` this is plain least squares:

```@example qr
ls = makejacobiansystem(J, n, :qr)              # a dense QR of the reduced Jacobian
DampedLinearSolve.gradient!(getb(ls), getA(ls))
markstale!(getA(ls))
dampen!(ls, 0.0, ones(n))
solve!(ls, :qr)
norm(getx(ls) - Jd \ rd) / norm(Jd \ rd)        # against a dense least-squares solve
```

## The same problem by the Schur complement

Form `H = JᵀJ` and `b = Jᵀr`, fill a [`SchurBlockHessian`](@ref) of the same layout (the Jacobian structure's
own Hessian-path structure, `SchurStructure(jst)`), and compare the two solves, damped this time with Marquardt
scaling:

```@example qr
H, b = Jd' * Jd, Jd' * rd
st = SchurStructure(jst)
Hs = SchurBlockHessian{Float64, Matrix{Float64}}(st)
rg(p) = off[p]:off[p] + sizes[p] - 1
for p in 1:nblocks(st), q in 1:p
    if p != q
        (p > nretained(st) && q > nretained(st)) && continue   # two eliminated blocks are never coupled
        iszero(H[rg(p), rg(q)]) && continue                    # no row block touches both
    end
    addblock!(Hs, p, q, H, rg(p), rg(q))
end

lambda, s = 0.1, diag(H)
lh = makehessiansystem(Hs, :cholesky)
getb(lh) .= b
dampen!(lh, lambda, s); solve!(lh, :cholesky)
dampen!(ls, lambda, s); solve!(ls, :qr)
norm(getx(ls) - getx(lh)) / norm(getx(lh))
```

## Solving the reduced problem

The second type parameter of [`SchurBlockJacobian`](@ref) names the storage of the reduced system: `Matrix{T}`
for a dense `J̃` / `S = J̃ᵀJ̃`, or `SparseMatrixCSC{T, Int}` for a sparse one. It is the counterpart of the type of
`R` in a `SchurBlockHessian`, and is set by whoever builds the prototype (a modelling package, from its own
dense-or-sparse option), so that a solver handed only `J` can dispatch on it. There is no default. The tables are
the same for both. `linsolver` then chooses how the reduced problem is solved:

- `:auto`, `:qr`: a QR of `J̃` with the retained damping appended as rows: dense, or SPQR for the sparse storage.
  It keeps the conditioning of `J`.
- `:cholesky`, `:ldl`: the normal equations `S = J̃ᵀJ̃`, the same system as the Schur route, which squares the
  condition number again.
- `:lsmr` (with Krylov.jl loaded, which loads the extension `KrylovExt`): matrix-free LSMR, `J̃` applied through the
  factored tables and never formed, block-Jacobi preconditioned.

```@example qr
Js = SchurBlockJacobian{Float64, SparseMatrixCSC{Float64, Int}}(copy(J.data), copy(J.r), jst)   # the same tables, sparse J̃
lss = makejacobiansystem(Js, n, :qr)
DampedLinearSolve.gradient!(getb(lss), getA(lss)); markstale!(getA(lss))
dampen!(lss, lambda, s); solve!(lss, :qr)
norm(getx(lss) - getx(ls)) / norm(getx(ls))
```

## Which to use

| | Schur (Hessian path) | QR elimination (Jacobian path) |
|---|---|---|
| Works at the condition number of | `JᵀJ`: `cond(J)²` | `J`, if the reduced problem is solved by QR or LSMR |
| Reduced system | can lose positive definiteness to rounding | `J̃ᵀJ̃`, positive semi-definite by construction |
| Memory per eliminated block | `d_e² + d_e·c_e` | `(m_e + d_e)(c_e + d_e + 1)`, twice (raw and factored) |
| A change of `λ` | re-factors from the stored blocks | re-factors from the raw tables |

Choose QR elimination when conditioning matters: in single precision, or when `JᵀJ` is close to singular. On an
ill-conditioned test problem with `cond(J)` about `8·10⁸`, the Schur route's `S` is no longer positive definite and
its solution has a relative error of about `5·10⁻²`, while QR elimination followed by a dense QR of `J̃` gets about
`10⁻⁸`. Choose Schur when memory matters: a table is dense in every retained block coupled to its eliminated block,
so its size grows with the number of rows times the coupled dimension (`m_e` rows, `d_e` eliminated and `c_e`
coupled retained columns, in the table above).

## Failure, threading, trial systems

- `solve!` reports `NotPositiveDefinite` if an eliminated block is under-determined (fewer rows than its
  dimension), rank deficient or zero without damping, or its table holds a `NaN` or `Inf`. Damping fixes the
  first three.
- Filling: the rows of different row blocks are disjoint, so threads that fill different row blocks never write
  the same memory. The elimination and back-substitution are threaded over chunks of eliminated blocks, the
  products and accumulations over tables or by retained-column ownership, as for the Schur kernels: threading
  needs Polyester loaded (without it everything runs serially), and nothing allocates, serial or threaded, except
  a threaded call on Julia 1.10 (see [Threading](@ref)).
- [`copy_structure`](@ref) gives a zeroed copy of the tables; DampedLinearSolve's `copy_structure` of the linear
  system gives a trial system with its own factored tables.
