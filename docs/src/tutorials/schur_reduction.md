# Schur Complement Reduction

Many problems (bundle adjustment is one example) are *bipartite*: a few **retained** variable blocks (`B`)
are coupled to many small **eliminated** blocks (`C`) through coupling blocks `E`. The normal
equations

```math
\begin{bmatrix} B & E \\ E^\top & C \end{bmatrix}
\begin{bmatrix} x_c \\ x_p \end{bmatrix} =
\begin{bmatrix} b_c \\ b_p \end{bmatrix}
```

are solved cheaply by eliminating `C`, which is block diagonal, then solving the much smaller reduced
system $S x_c = g$ with $S = B - E C^{-1} E^\top$ and $g = b_c - E C^{-1} b_p$, then recovering
$x_p = C^{-1}(b_p - E^\top x_c)$. BlockLinearSystems provides the data structures and the parallel kernels
for this; the optimizer-facing linear systems built on them live in
[SecondOrderOptimization.jl](https://github.com/ojwoodford/SecondOrderOptimization.jl).

## Describing the problem

A [`SchurStructure`](@ref) records the block dimensions and which retained blocks are coupled to each eliminated
block. Eliminated blocks may have any dimension from 1 to 255: up to `BlockLinearSystems.D_STATIC` (8) they
use statically sized register arithmetic, larger ones use a LAPACK Cholesky on a scratch matrix. The
numerical values live in a [`SchurBlockMatrix`](@ref): `B` (a dense `Matrix` or a
[`SparseBlockLinearSystem`](@ref)), `C_data` (the eliminated diagonal blocks, each stored column-major) and
`E_data` (one `dp × dc` block per coupling, stored as the transpose of the retained-by-eliminated block).

## A complete example

```@example schur
using BlockLinearSystems, LinearAlgebra, Random

rng = Random.Xoshiro(1)
c_dims = [6, 6, 6]                 # three retained blocks
p_dims = [3, 3, 3, 3, 2, 1]        # six eliminated blocks, of dimension 3, 3, 3, 3, 2 and 1
couplings = [[1, 2], [1, 2, 3], [2, 3], [1, 3], [1, 2], [3]]
st = SchurStructure(c_dims, p_dims, couplings)

# A random diagonally dominant (so positive definite) Hessian with exactly the structure described above:
# dense B, block-diagonal C, and a coupling block wherever a retained block is coupled to an eliminated one
n = st.nc + sum(p_dims)
H = zeros(n, n)
H[1:st.nc, 1:st.nc] .= rand(rng, st.nc, st.nc)
for l in 1:st.npblocks
    p0, dp = st.nc + st.boffsets_p[l], p_dims[l]
    H[p0:p0+dp-1, p0:p0+dp-1] .= rand(rng, dp, dp)
    for e in st.cpptr[l]:st.cpptr[l+1]-1
        c0, dc = st.boffsets_c[st.cpblock[e]], c_dims[st.cpblock[e]]
        H[c0:c0+dc-1, p0:p0+dp-1] .= rand(rng, dc, dp)
    end
end
H = H + H'
H[diagind(H)] .= sum(abs, H; dims=2)[:] .+ 1
b = randn(rng, n)

# Fill B, C and E (E is stored transposed: E_data[eo + (r-1)*dp + i - 1] = H[retained row r, eliminated column i])
B = H[1:st.nc, 1:st.nc]
C_data = zeros(sum(abs2, p_dims))
E_data = zeros(sum(c_dims[st.cpblock[e]] * p_dims[l] for l in 1:st.npblocks for e in st.cpptr[l]:st.cpptr[l+1]-1))
for l in 1:st.npblocks
    dp, po, p0 = p_dims[l], st.ppoffsets[l], st.nc + st.boffsets_p[l]
    C_data[po:po+dp^2-1] .= vec(H[p0:p0+dp-1, p0:p0+dp-1])
    for e in st.cpptr[l]:st.cpptr[l+1]-1
        c = st.cpblock[e]; dc = c_dims[c]; c0 = st.boffsets_c[c]
        E_data[st.cpoffsets[e]:st.cpoffsets[e]+dc*dp-1] .= vec(H[c0:c0+dc-1, p0:p0+dp-1]')
    end
end
Hs = SchurBlockMatrix(st, B, C_data, E_data)
nothing # hide
```

Reduce, solve the small system, and back-substitute. `lambda` and `scaling` add Levenberg–Marquardt damping
`lambda * diag(scaling)` to the whole system (the scaling is the damping diagonal itself, e.g. `abs.(diag(H))`
for Marquardt scaling):

```@example schur
lambda, scaling = 0.1, ones(n)
nc = st.nc
ws = SchurReductionWorkspace{Float64}(st, 1)      # one chunk; use more for multithreading
S = zeros(nc, nc); g = zeros(nc)
reset_schur_dense!(S, Hs.B, view(scaling, 1:nc), lambda, ws)
reduce_schur_dense!(S, g, view(b, 1:nc), view(b, nc+1:n), view(scaling, nc+1:n), lambda, st, C_data, E_data, ws) ||
    error("an eliminated block is singular")
x = zeros(n)
x[1:nc] .= Symmetric(S, :U) \ g                   # only the upper triangle of S is filled
backsubstitute!(view(x, nc+1:n), view(x, 1:nc), st, E_data, ws.C_inv_data, ws.z_data)

x_full = (H + lambda * Diagonal(scaling)) \ b     # reference: solve the whole system
norm(x - x_full) / norm(x_full)
```

The reduction returns `false` if an eliminated block is numerically singular (or not positive definite), and
`ws.failed_block[]` then holds the lowest failing block index.

## Sparse reduced systems

When many pairs of retained blocks share no eliminated block, `S` is sparse. [`build_schur_pattern`](@ref) computes its
pattern, and [`build_b_to_s_map`](@ref) maps the stored blocks of a [`SparseBlockLinearSystem`](@ref) `B` to
it; [`reduce_schur_sparse!`](@ref) then fills `S.nzval` directly. Retained–retained couplings that go through
no eliminated block (odometry or loop-closure factors) need slots too: pass them as `extra_blocks`, which
[`retained_block_pairs`](@ref) extracts from `B`:

```julia
pattern = build_schur_pattern(st; extra_blocks=retained_block_pairs(B))
S = pattern.S_sparse
b_to_s = build_b_to_s_map(B, pattern, st.boffsets_c)
reset_schur_sparse!(S.nzval, B.data, b_to_s, pattern.diag_inds, scaling_c, lambda, ws)
reduce_schur_sparse!(S, g, b_c, b_p, scaling_p, lambda, st, C_data, E_data,
                     pattern.block_colptr, pattern.block_rowval, pattern.block_row_offsets, pattern.col_heights, ws)
```

## How the reduction is parallelised

The reduction runs in two phases. Phase 1 factors every eliminated block (in parallel over chunks of
eliminated blocks) and stores `C⁻¹`, `z = C⁻¹ g_p` and `W = C⁻¹Eᵀ`. Phase 2 splits the *retained* blocks
into contiguous ranges (`ws.col_chunks`); each chunk writes only its own columns of the single `S`, so there
are no per-chunk copies of `S` and no merge pass. Loops run serially unless more than one thread is available
and there is enough work (`BlockLinearSystems.SCHUR_THREAD_MIN_WORK`); threading uses Polyester.

## Trial systems

Optimizers that linearize at a trial point keep the current system for retries.
[`copy_structure`](@ref)`(::SchurBlockMatrix)` allocates new zeroed `B`, `C_data` and `E_data` sharing only
the immutable structure, and [`trial_workspace`](@ref) gives a workspace with its own caches.

## Using it with SecondOrderOptimization.jl

Loading both `BlockLinearSystems` and `SecondOrderOptimization` activates an extension that wraps all of this
in `DenseSchurLinearSystem` and `SparseSchurLinearSystem` (selected with `linsolver = :schur`, `:dense_schur`
or `:sparse_schur`), for use with its Levenberg–Marquardt, dogleg and Newton iterators.
