# QR Elimination (Jacobian Path)

The [Schur complement](schur_reduction.md) eliminates independent variable blocks from the normal
equations `H x = b`. When the problem is available as residuals and a Jacobian, the same blocks can be
eliminated from the least-squares problem itself, by QR, without forming `H = JᵀJ`. This is the method of
Demmel, Sommer, Cremers and Usenko, "Square Root Bundle Adjustment for Large-Scale Reconstruction"
(CVPR 2021). Nothing here is specific to bundle adjustment.

## Conventions

The same as for the Schur systems, so that the two routes give the same numbers:

- The problem is `min ‖J x − r‖²` over `x = [x_r; x_e]`: **retained** variables first, then **eliminated**
  ones, with the partition, block order and coupling lists of a [`SchurStructure`](@ref). Its normal
  equations are `JᵀJ x = Jᵀr`, i.e. `[R Cᵀ; C E] x = b` with `H = JᵀJ` and `b = Jᵀr`. Callers that want a
  descent step from `f(x0 + dx) ≈ f(x0) + J dx` store `r = f(x0)` and negate `x`, as with the Schur systems.
- Each residual row involves at most one eliminated block (the Schur condition).
- Levenberg–Marquardt damping solves `(JᵀJ + λ·Diagonal(s)) x = Jᵀr`, with `s` in the units of `diag(JᵀJ)`
  (for Marquardt scaling, `s = diag(JᵀJ)`, which [`jtj_diag!`](@ref) computes), not squared.

## The table layout

The rows that involve eliminated block `e` are stored as one dense, column-major table, with the columns
of every retained block coupled to `e`, then `e`'s own columns, then the residual. `d_e` damping rows sit
underneath, zero in the raw storage:

```
                 retained blocks coupled to e      e     r
               ┌───────────┬──────────────────┬───────┬───┐
  m_e data     │   J_r1    │   J_r2   …       │  J_e  │ r │
  rows         │           │                  │       │   │
               ├───────────┼──────────────────┼───────┼───┤
  d_e damping  │     0     │        0         │ √λs_e │ 0 │   written by reduce_qr!
               └───────────┴──────────────────┴───────┴───┘
```

[`reduce_qr!`](@ref) copies each table, writes the damping rows and applies the Householder QR of its
eliminated columns to the whole table:

```
          Qᵀ ·  [ J_r | J_e | r ]   =   [ B   | U | q   ]    d_e rows:   U x_e + B x_r = q
                [  0  | √λD |  0 ]       [ J̃_e | 0 | r̃_e ]    m_e rows:   J̃_e x_r = r̃_e
```

The top rows give `x_e = U⁻¹ (q − B x_r)` once `x_r` is known ([`backsubstitute!`](@ref)). The bottom rows
do not involve `x_e`. Together with the rows that involve no eliminated block ("free" rows, given as free
groups), they form the reduced problem `min ‖J̃ x_r − r̃‖²`, whose normal equations are exactly the Schur
complement: `J̃ᵀJ̃ = R − Cᵀ E⁻¹ C` and `J̃ᵀr̃ = b_r − Cᵀ E⁻¹ b_e`, with `E` including its damping.
The retained part of the damping is left to the caller: rows `√(λ s_r)` appended to `J̃`, or `λ s_r` added
to the diagonal of `J̃ᵀJ̃`.

Each table contributes `m_e` rows to `J̃` whatever `λ` is ([`reduced_nrows`](@ref)): with `λ = 0` the
damping rows stay zero and `d_e` of those rows are zero too.

## A complete example

The same kind of small problem as in the Schur tutorial, built as a Jacobian. Three retained blocks, six
eliminated blocks, and one free group of rows that touches only retained blocks 1 and 3:

```@example qr
using BlockLinearSystems, LinearAlgebra, Random

rng = Random.Xoshiro(1)
r_dims = [6, 6, 6]
e_dims = [3, 3, 3, 3, 2, 1]
couplings = [[1, 2], [1, 2, 3], [2, 3], [1, 3], [1, 2], [3]]
e_rows = [8, 10, 6, 6, 4, 3]                 # residual rows that involve each eliminated block
jst = SchurJacobianStructure(r_dims, e_dims, couplings, e_rows; free_rows=[12], free_retained=[[1, 3]])
st = jst.schur                               # the SchurStructure it wraps
J = SchurBlockJacobian{Float64}(jst)

# Fill every table with random values through the assembly views; keep a dense copy for checking
nr, ne = st.nr, sum(e_dims)
Jd, rd = zeros(0, nr + ne), zeros(0)
for t in 1:jst.ntables
    m = t <= st.neblocks ? e_rows[t] : jst.free_rows[t - st.neblocks]
    rows = zeros(m, nr + ne)
    for u in jst.table_ptr[t]:jst.table_ptr[t+1]-1          # the retained blocks of table t
        r = jst.entry_retained[u]
        B = randn(rng, m, r_dims[r])
        coupling_block(J, u) .= B
        rows[:, st.boffsets_r[r] .+ (0:r_dims[r]-1)] .= B
    end
    if t <= st.neblocks
        E = randn(rng, m, e_dims[t])
        eliminated_block(J, t) .= E
        rows[:, nr .+ st.boffsets_e[t] .+ (0:e_dims[t]-1)] .= E
    end
    rv = randn(rng, m)
    residual_block(J, t) .= rv
    global Jd = [Jd; rows]
    global rd = [rd; rv]
end
size(Jd)
```

Eliminate, solve the reduced problem by a dense QR, and back-substitute. With `λ = 0` this is plain least
squares:

```@example qr
ws = QRReductionWorkspace(J)
reduce_qr!(ws, J, ones(ne), 0.0) || error("an eliminated block is rank deficient")

Jt, rt = zeros(reduced_nrows(jst), nr), zeros(reduced_nrows(jst))
reduced_jacobian!(Jt, rt, ws)                # J̃ and r̃
x_r = qr(Jt) \ rt
x_e = zeros(ne)
backsubstitute!(x_e, x_r, ws)

norm([x_r; x_e] - Jd \ rd) / norm(Jd \ rd)    # against a dense least-squares solve
```

## The same problem by the Schur complement

Form `H = JᵀJ` and `b = Jᵀr`, fill a [`SchurBlockMatrix`](@ref) as in the
[Schur tutorial](schur_reduction.md), and compare the two reduced systems, damped this time:

```@example qr
H, b = Jd' * Jd, Jd' * rd
R = H[1:nr, 1:nr]
E_data = zeros(sum(abs2, e_dims))
C_data = zeros(sum(r_dims[st.entry_retained[c]] * e_dims[e] for e in 1:st.neblocks for c in st.eliminated_ptr[e]:st.eliminated_ptr[e+1]-1))
for e in 1:st.neblocks
    de, eo, e0 = e_dims[e], st.eoffsets[e], nr + st.boffsets_e[e]
    E_data[eo:eo+de^2-1] .= vec(H[e0:e0+de-1, e0:e0+de-1])
    for c in st.eliminated_ptr[e]:st.eliminated_ptr[e+1]-1
        r = st.entry_retained[c]; dr = r_dims[r]; r0 = st.boffsets_r[r]
        C_data[st.coffsets[c]:st.coffsets[c]+de*dr-1] .= vec(H[e0:e0+de-1, r0:r0+dr-1])
    end
end

lambda = 0.1
s = jtj_diag!(zeros(nr + ne), J)              # Marquardt scaling: diag(JᵀJ)
s_r, s_e = s[1:nr], s[nr+1:end]

# Schur: S = R + λ diag(s_r) − Cᵀ (E + λ diag(s_e))⁻¹ C
sws = SchurReductionWorkspace{Float64}(st, 1)
S_schur, g_schur = zeros(nr, nr), zeros(nr)
reset_schur_dense!(S_schur, R, s_r, lambda, sws)
reduce_schur_dense!(S_schur, g_schur, b[1:nr], b[nr+1:end], s_e, lambda, st, E_data, C_data, sws)

# QR: J̃ᵀJ̃, plus the retained damping
reduce_qr!(ws, J, s_e, lambda)
S_qr, g_qr = zeros(nr, nr), zeros(nr)
reduced_normal_dense!(S_qr, g_qr, ws)
S_qr[diagind(S_qr)] .+= lambda .* s_r

(norm(triu(S_qr) - triu(S_schur)) / norm(triu(S_schur)), norm(g_qr - g_schur) / norm(g_schur))
```

[`reduced_normal_sparse!`](@ref) does the same into the sparse storage of
[`build_schur_pattern`](@ref); pass the free groups' pairs as extra blocks:
`build_schur_pattern(st; extra_blocks=free_block_pairs(jst))`.

## Solving the reduced problem

- **Dense QR**: [`reduced_jacobian!`](@ref) writes `J̃` and `r̃` into a `Matrix` and `Vector`.
- **Iterative (LSMR, CG on the normal equations)**: [`reduced_mul!`](@ref) and [`reduced_tmul!`](@ref) give
  `J̃ v` and `J̃ᵀ w` without forming anything, [`reduced_residual!`](@ref) gives `r̃`, and
  [`reduced_diagonal_blocks!`](@ref) the diagonal blocks of `J̃ᵀJ̃` for a block-Jacobi preconditioner.
- **Cholesky or LDLᵀ**: [`reduced_normal_dense!`](@ref) / [`reduced_normal_sparse!`](@ref) form
  `S = J̃ᵀJ̃`. This gives the same system as the Schur route and squares the condition number again.

The gradient `Jᵀr`, the column norms `diag(JᵀJ)` and the model term `‖J v‖²` of the full problem come from
[`jtr!`](@ref), [`jtj_diag!`](@ref) and [`fast_bAb`](@ref), on the raw tables or the factored ones (`Q` is
orthogonal). The factored tables hold the damped Jacobian, so for `λ > 0` their eliminated column norms
include `λ s_e` and `‖J v‖²` includes `λ Σ s_e v_e²`; the raw tables are always kept.

## Which to use

| | Schur (Hessian path) | QR elimination (Jacobian path) |
|---|---|---|
| Works at the condition number of | `JᵀJ`: `cond(J)²` | `J`, if the reduced problem is solved by QR or LSMR |
| Reduced system | can lose positive definiteness to rounding | `J̃ᵀJ̃`, positive semi-definite by construction |
| Memory per eliminated block | `d_e² + d_e·c_e` | `(m_e + d_e)(c_e + d_e + 1)`, twice (raw and factored) |
| A change of `λ` | re-factors from the stored blocks | re-factors from the raw tables |

Choose QR elimination when conditioning matters: in single precision, or when `JᵀJ` is close to singular.
On an ill-conditioned test problem with `cond(J)` about `8·10⁸`, the Schur route's `S` is no longer
positive definite and its solution has a relative error of about `5·10⁻²`, while QR elimination followed by a
dense QR of `J̃` gets about `10⁻⁸`. Choose Schur when memory matters: a table is dense in every retained
block coupled to its eliminated block, so its size grows with the number of rows times the coupled
dimension. [`schur_jacobian_bytes`](@ref) reports the memory from the structure alone, before anything is
allocated:

```@example qr
schur_jacobian_bytes(jst)                     # raw tables + factored copy + scratch, in bytes
```

## Failure, threading, trial systems

- [`reduce_qr!`](@ref) returns `false` if an eliminated block is under-determined (fewer rows than its
  dimension), rank deficient or zero without damping, or its table holds a `NaN` or `Inf`;
  `ws.failed_block[]` is the lowest failing block, as for the Schur reduction. Damping fixes the first
  three.
- Filling: the rows of a table belong to one eliminated block, so threads that fill different tables never
  write the same memory. The reduction and back-substitution are threaded over chunks of eliminated
  blocks, the products and accumulations over tables or by retained-column ownership, with the same
  thresholds as the Schur kernels (`BlockLinearSystems.SCHUR_THREAD_MIN_WORK`).
- [`copy_structure`](@ref) gives a zeroed copy of the tables, and [`trial_workspace`](@ref) a workspace
  with its own factored tables, for a trial system.
