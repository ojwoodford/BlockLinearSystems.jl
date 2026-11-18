# Schur Complement Reduction

Many problems (bundle adjustment is one example) are *bipartite*: a few **retained** variable blocks
are coupled to many small **eliminated** blocks. Upper case names a whole group of blocks and lower case
one block: `R` holds the retained-retained blocks, `E` the eliminated blocks `E_e` (block diagonal: no two
eliminated blocks are coupled), and `C` the coupling blocks `C_er` between eliminated block `e` and
retained block `r`. The normal equations

```math
\begin{bmatrix} R & C^\top \\ C & E \end{bmatrix}
\begin{bmatrix} x_r \\ x_e \end{bmatrix} =
\begin{bmatrix} b_r \\ b_e \end{bmatrix}
```

are solved cheaply by eliminating `E`, which is block diagonal, then solving the much smaller reduced
system $S x_r = g$ with $S = R - C^\top E^{-1} C$ and $g = b_r - C^\top E^{-1} b_e$, then recovering
$x_e = E^{-1}(b_e - C x_r)$. BlockLinearSystems lays the elimination out, stores the system, and solves it
through the linear-system contract of [DampedLinearSolve.jl](https://github.com/ojwoodford/DampedLinearSolve.jl)
(see [Solving with DampedLinearSolve.jl](dampedlinearsolve.md)). The caller does three things:

1. **Describe** the problem in its own numbering: the size of each block of unknowns (a *column block*), which
   column blocks each group of terms couples (a *row block*: one cost of a least-squares problem, say), and
   which column blocks to eliminate.
2. **Lay it out and allocate it**: [`SchurStructure`](@ref) orders the blocks and derives the couplings;
   [`SchurBlockHessian`](@ref) allocates the storage.
3. **Fill it by block**, naming pairs of blocks with [`addblock!`](@ref), then build the linear system with
   DampedLinearSolve's `makehessiansystem` and solve it.

## The layout

The layout numbers the blocks retained first, then eliminated: [`blockorder`](@ref)`(st)[p]` is the caller's
number of layout block `p`, and layout blocks `1:`[`nretained`](@ref)`(st)` are retained. Retained blocks are
ordered by size, descending; eliminated blocks by size (so that blocks of one size are processed together, with
that size known statically), then by the first retained block they couple to. Every vector of the system (the
right-hand side, the gradient, the solution) is in layout order: block `p` starts at
`computestartindices(blocksizes(st))[p]`.

Blocks may have any dimension from 1 to 65535. Eliminated blocks up to `BlockLinearSystems.D_STATIC` (8) use
statically sized register arithmetic; larger ones use a LAPACK Cholesky on a scratch matrix. That scratch is
`d_max × d_max` per chunk (one per thread), `d_max` being the largest eliminated dimension, and each eliminated
block's own factor takes `d_e²` values, so very large eliminated blocks are supported but cost memory accordingly.

## A complete example

Three retained blocks of 6 and six eliminated blocks (3, 3, 3, 3, 2 and 1), in an order of the caller's own;
each eliminated block forms one row block with the retained blocks it is coupled to:

```@example schur
using BlockLinearSystems, DampedLinearSolve, LinearAlgebra, SparseArrays, Random

rng = Random.Xoshiro(1)
sizes0 = [3, 6, 3, 6, 3, 6, 3, 2, 1]            # column blocks, in the caller's order
elim = [true, false, true, false, true, false, true, true, true]
rowblocks = [[1, 2, 4], [3, 2, 4, 6], [5, 4, 6], [7, 2, 6], [8, 2, 4], [9, 6]]
st = SchurStructure(sizes0, rowblocks; eliminated = elim)
blockorder(st), nretained(st), blocksizes(st)
```

Allocate the system and fill it by block. Here each row block is a small least-squares term with four residual
rows `A`, contributing `AᵀA` over its blocks; every block also gets a unit prior. Each pair of blocks of a
contribution is added once, in either order; `addblock!` puts it wherever, and in whichever orientation, the
system stores it. A dense copy `H` is kept for checking:

```@example schur
sizes = blocksizes(st)
off = computestartindices(sizes); n = sum(sizes)
rg(p) = off[p]:off[p] + sizes[p] - 1           # the scalar range of layout block p
lay = invperm(blockorder(st))                  # the layout number of each of the caller's blocks

Hs = SchurBlockHessian{Float64, Matrix{Float64}}(st)   # dense retained part
H = zeros(n, n)
for p in 1:nblocks(st)
    addblock!(Hs, p, p, Matrix(1.0I, sizes[p], sizes[p]), 1:sizes[p], 1:sizes[p])
    H[rg(p), rg(p)] .+= I(sizes[p])
end
for rb in rowblocks
    ps = lay[rb]                               # the row block's column blocks, in layout numbers
    loc = computestartindices(sizes[ps])
    lr(i) = loc[i]:loc[i] + sizes[ps[i]] - 1   # their ranges in the contribution
    A = randn(rng, 4, sum(sizes[ps]))
    B = A' * A
    for i in eachindex(ps), j in 1:i
        addblock!(Hs, ps[i], ps[j], B, lr(i), lr(j))
    end
    cols = reduce(vcat, rg.(ps))
    H[cols, cols] .+= B
end
b = randn(rng, n)
nothing # hide
```

`dot(x, Hs, y)` is the bilinear form `xᵀ H y`, without forming `H`; `dot(v, Hs, v)`, with the same vector passed
twice, takes a faster path for the quadratic form `vᵀ H v` (the model term of a trust-region or
Levenberg–Marquardt step):

```@example schur
v, w = randn(rng, n), randn(rng, n)
dot(v, Hs, v) ≈ dot(v, H * v) && dot(v, Hs, w) ≈ dot(v, H * w)
```

Solve `(H + λ diag(s)) x = b` through the contract: build the system from the prototype, set its right-hand side,
damp it and solve. `scaling` is the damping diagonal itself (for Marquardt scaling, `abs.(diag(H))`):

```@example schur
ls = makehessiansystem(Hs, :auto)
getb(ls) .= b
lambda = 0.1
dampen!(ls, lambda, ones(n))
solve!(ls, :auto)
```

```@example schur
x_full = (H + lambda * I) \ b                  # reference: solve the whole system
norm(getx(ls) - x_full) / norm(x_full)
```

`getA(ls)` is the `SchurBlockHessian` the system reads: refill it (after `zero!`) and solve again for the next
linearization. `solve!` reports `NotPositiveDefinite` if an eliminated block is numerically singular (or not
positive definite) or the reduced system cannot be factored.

## How the reduction works

The solve reduces in two phases. Phase 1 factors every eliminated block, damped, and stores `E_e⁻¹`,
`z_e = E_e⁻¹ b_e` and `W_er = E_e⁻¹ C_er`. Phase 2 accumulates `S = R - Σ_e C_eᵀ W_e` and `g` into the
reduced system; the reduced system is solved by DampedLinearSolve (`linsolver = :auto` or `:cholesky`:
Cholesky, falling back to Bunch–Kaufman; `:ldl`: Bunch–Kaufman, or LDLᵀ when sparse); then
`x_e = E⁻¹(b_e - C x_r)` block by block.

Phase 1 is parallel over chunks of eliminated blocks of one size (`nchunks` of the structure). Phase 2 splits the
*retained* blocks into contiguous ranges; each chunk writes only its own columns of the single `S`, so there are
no per-chunk copies of `S` and no merge pass. Threading uses Polyester, a weak dependency: the loops thread only
when Polyester is loaded (`using Polyester`, or a package that loads it), more than one thread is available, the
structure has more than one chunk and its work is at least its `min_work` (both set when it is built); otherwise
they run serially. Either way nothing allocates, except a threaded call on Julia
1.10 (see [Threading](@ref)).

## Sparse reduced systems

When many pairs of retained blocks share no eliminated block, `S` is sparse. With a block-sparse retained part,
`SchurBlockHessian{Float64, SparseBlockLinearSystem{Float64}}(st)`, the system's `S` is a `SparseMatrixCSC` on
the pattern of [`build_schur_pattern`](@ref)`(st)`: the pairs of retained blocks that share an eliminated block
or a row block of the description (a relative-pose term between two cameras, say), and the diagonal blocks.
The pattern is built when the `SchurBlockHessian` is allocated and kept with it (`Hs.pattern`), so a system built
from it later does not build it again; a pattern built earlier can be given (`pattern = ...`), with entries of the
caller's own attached (`merge(pattern, (S_factor = f,))`). The pattern is shared by `copy_structure` copies:
treat it as read-only.

```@example schur
Hsp = SchurBlockHessian{Float64, SparseBlockLinearSystem{Float64}}(st)
pat = Hsp.pattern
(size(pat.S_sparse), nnz(pat.S_sparse))
```

## Trial systems

Optimizers that linearize at a trial point keep the current system for retries.
[`copy_structure`](@ref)`(::SchurBlockHessian)` allocates new zeroed `R`, `E_data` and `C_data` sharing only the
immutable structure (and the read-only `pattern`, if any); DampedLinearSolve's `copy_structure` of the linear
system gives a trial system with its own storage and caches.

## Using it with an optimizer

Passing a [`SchurBlockHessian`](@ref) as the Hessian prototype `H` to an optimizer written against
DampedLinearSolve's contract (for example SecondOrderOptimization's `HessianObjective(f, fdf!; H)`, then
`optimize!(x0, obj, options)`) is what selects elimination; there is no option to set. Whether the reduced system
is dense or sparse follows the storage of `R`, and `linsolver` names the solver applied to the reduced system:
`:auto`, `:cholesky` or `:ldl`.
