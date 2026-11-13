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
$x_e = E^{-1}(b_e - C x_r)$. BlockLinearSystems provides the data structures and the parallel kernels
for this; the optimizer-facing linear systems built on them live in
[SecondOrderOptimization.jl](https://github.com/ojwoodford/SecondOrderOptimization.jl).

## Describing the problem

A [`SchurStructure`](@ref) records the block dimensions and which retained blocks are coupled to each eliminated
block. Eliminated blocks may have any dimension from 1 to 255: up to `BlockLinearSystems.D_STATIC` (8) they
use statically sized register arithmetic, larger ones use a LAPACK Cholesky on a scratch matrix. The
numerical values live in a [`SchurBlockMatrix`](@ref): `R` (a dense `Matrix` or a
[`SparseBlockLinearSystem`](@ref)), `E_data` (the eliminated blocks `E_e`, each stored column-major) and
`C_data` (the coupling blocks `C_er`, one `de × dr` block per coupling, each stored column-major).

## A complete example

```@example schur
using BlockLinearSystems, LinearAlgebra, Random

rng = Random.Xoshiro(1)
r_dims = [6, 6, 6]                 # three retained blocks
e_dims = [3, 3, 3, 3, 2, 1]        # six eliminated blocks, of dimension 3, 3, 3, 3, 2 and 1
couplings = [[1, 2], [1, 2, 3], [2, 3], [1, 3], [1, 2], [3]]
st = SchurStructure(r_dims, e_dims, couplings)

# A random diagonally dominant (so positive definite) Hessian with exactly the structure described above:
# dense R, block-diagonal E, and a coupling block wherever a retained block is coupled to an eliminated one
n = st.nr + sum(e_dims)
H = zeros(n, n)
H[1:st.nr, 1:st.nr] .= rand(rng, st.nr, st.nr)
for e in 1:st.neblocks
    e0, de = st.nr + st.boffsets_e[e], e_dims[e]
    H[e0:e0+de-1, e0:e0+de-1] .= rand(rng, de, de)
    for c in st.eliminated_ptr[e]:st.eliminated_ptr[e+1]-1
        r0, dr = st.boffsets_r[st.entry_retained[c]], r_dims[st.entry_retained[c]]
        H[e0:e0+de-1, r0:r0+dr-1] .= rand(rng, de, dr)
    end
end
H = H + H'
H[diagind(H)] .= sum(abs, H; dims=2)[:] .+ 1
b = randn(rng, n)

# Fill R, E and C (each block column-major; C_er is the block of H in eliminated rows, retained columns)
R = H[1:st.nr, 1:st.nr]
E_data = zeros(sum(abs2, e_dims))
C_data = zeros(sum(r_dims[st.entry_retained[c]] * e_dims[e] for e in 1:st.neblocks for c in st.eliminated_ptr[e]:st.eliminated_ptr[e+1]-1))
for e in 1:st.neblocks
    de, eo, e0 = e_dims[e], st.eoffsets[e], st.nr + st.boffsets_e[e]
    E_data[eo:eo+de^2-1] .= vec(H[e0:e0+de-1, e0:e0+de-1])
    for c in st.eliminated_ptr[e]:st.eliminated_ptr[e+1]-1
        r = st.entry_retained[c]; dr = r_dims[r]; r0 = st.boffsets_r[r]
        C_data[st.coffsets[c]:st.coffsets[c]+de*dr-1] .= vec(H[e0:e0+de-1, r0:r0+dr-1])
    end
end
Hs = SchurBlockMatrix(st, R, E_data, C_data)
nothing # hide
```

Reduce, solve the small system, and back-substitute. `lambda` and `scaling` add Levenberg–Marquardt damping
`lambda * diag(scaling)` to the whole system (the scaling is the damping diagonal itself, e.g. `abs.(diag(H))`
for Marquardt scaling):

```@example schur
lambda, scaling = 0.1, ones(n)
nr = st.nr
ws = SchurReductionWorkspace{Float64}(st, 1)      # one chunk; use more for multithreading
S = zeros(nr, nr); g = zeros(nr)
reset_schur_dense!(S, Hs.R, view(scaling, 1:nr), lambda, ws)
reduce_schur_dense!(S, g, view(b, 1:nr), view(b, nr+1:n), view(scaling, nr+1:n), lambda, st, E_data, C_data, ws) ||
    error("an eliminated block is singular")
x = zeros(n)
x[1:nr] .= Symmetric(S, :U) \ g                   # only the upper triangle of S is filled
backsubstitute!(view(x, nr+1:n), view(x, 1:nr), st, C_data, ws.E_inv_data, ws.z_data)

x_full = (H + lambda * Diagonal(scaling)) \ b     # reference: solve the whole system
norm(x - x_full) / norm(x_full)
```

The reduction returns `false` if an eliminated block is numerically singular (or not positive definite), and
`ws.failed_block[]` then holds the lowest failing block index.

## Sparse reduced systems

When many pairs of retained blocks share no eliminated block, `S` is sparse. [`build_schur_pattern`](@ref) computes its
pattern, and [`build_r_to_s_map`](@ref) maps the stored blocks of a [`SparseBlockLinearSystem`](@ref) `R` to
it; [`reduce_schur_sparse!`](@ref) then fills `S.nzval` directly. Retained–retained couplings that go through
no eliminated block (odometry or loop-closure factors) need slots too: pass them as `extra_blocks`, which
[`retained_block_pairs`](@ref) extracts from `R`:

```julia
pattern = build_schur_pattern(st; extra_blocks=retained_block_pairs(R))
S = pattern.S_sparse
r_to_s = build_r_to_s_map(R, pattern, st.boffsets_r)
reset_schur_sparse!(S.nzval, R.data, r_to_s, pattern.diag_inds, scaling_r, lambda, ws)
reduce_schur_sparse!(S, g, b_r, b_e, scaling_e, lambda, st, E_data, C_data,
                     pattern.block_colptr, pattern.block_rowval, pattern.block_row_offsets, pattern.col_heights, ws)
```

If the pattern is already known when the [`SchurBlockMatrix`](@ref) is built, attach it with
`Hs = SchurBlockMatrix(st, R, E_data, C_data; pattern=pattern)`. Code that receives only `Hs` can then
read `Hs.pattern` instead of rebuilding it (`nothing` means "build it when needed"). The pattern is shared
by `copy_structure` copies, so treat it as read-only: copy `pattern.S_sparse` before using it as storage.

## How the reduction is parallelised

The reduction runs in two phases. Phase 1 factors every eliminated block (in parallel over chunks of
eliminated blocks) and stores `E_e⁻¹`, `z_e = E_e⁻¹ b_e` and `W_er = E_e⁻¹ C_er`. Phase 2 splits the *retained* blocks
into contiguous ranges (`ws.col_chunks`); each chunk writes only its own columns of the single `S`, so there
are no per-chunk copies of `S` and no merge pass. Loops run serially unless more than one thread is available
and there is enough work (`BlockLinearSystems.SCHUR_THREAD_MIN_WORK`); threading uses Polyester.

## Trial systems

Optimizers that linearize at a trial point keep the current system for retries.
[`copy_structure`](@ref)`(::SchurBlockMatrix)` allocates new zeroed `R`, `E_data` and `C_data` sharing only
the immutable structure (and the read-only `pattern`, if any), and [`trial_workspace`](@ref) gives a workspace with its own caches.

## Using it with SecondOrderOptimization.jl

Loading both `BlockLinearSystems` and `SecondOrderOptimization` activates an extension that wraps all of this
in `DenseSchurLinearSystem` and `SparseSchurLinearSystem` (selected with `linsolver = :schur`, `:dense_schur`
or `:sparse_schur`), for use with its Levenberg–Marquardt, dogleg and Newton iterators.
