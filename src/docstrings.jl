"""
    SparseBlockLinearSystem{T}(sparsitytransposed::SparseMatrixCSC{Bool, Int}, rowblocksizes, colblocksizes=rowblocksizes)
    SparseBlockLinearSystem{T}(nnzvals, indicestransposed, rowblocksizes, colblocksizes=rowblocksizes)

Cache-friendly block-sparse matrix storing each non-zero block contiguously in column-major order within a flat `data::Vector{T}` buffer.

The block sparsity pattern is provided transposed (`sparsitytransposed` has size `(length(colblocksizes), length(rowblocksizes))`) so that blocks along a block-row are contiguous in the CSC structure of `indicestransposed`.

# Fields
- `data::Vector{T}`: Contiguous storage for all non-zero block entries.
- `indices::SparseMatrixCSC{Int, Int}`: Cached start indices in `data` for each block `(row, col)`.
- `indicestransposed::SparseMatrixCSC{Int, Int}`: Transposed start indices in `data` (`indicestransposed[col, row]`).
- `rowblocksizes::Vector{UInt8}`: Row dimensions of each block row.
- `columnblocksizes::Vector{UInt8}`: Column dimensions of each block column.
- `m::Int`: Total number of scalar rows.
- `n::Int`: Total number of scalar columns.

# Example
```julia
using BlockLinearSystems, SparseArrays

sparsity_t = sparse([true false; true true])
bsm = SparseBlockLinearSystem{Float64}(sparsity_t, [2, 3], [2, 3])
block(bsm, 1, 1) .= [1.0 2.0; 2.0 4.0]
```
"""
SparseBlockLinearSystem

"""
    DenseBlockLinearSystem{T}(rowblockoffsets, columnblockoffsets=rowblockoffsets)
    DenseBlockLinearSystem{T}(data::AbstractMatrix{T}, rowblockoffsets, columnblockoffsets=rowblockoffsets)
    DenseBlockLinearSystem(data::AbstractMatrix{T}, rowblockoffsets, columnblockoffsets=rowblockoffsets)

Block-partitioned dense matrix backed by a matrix buffer `data` (such as `Matrix{T}`, `SubArray`, or `FixedSizeMatrix{T}`) alongside 1-based sentinel-terminated block offset vectors `rowblockoffsets` and `columnblockoffsets`.

# Fields
- `data::AbstractMatrix{T}`: Underlying matrix buffer of size `(rowblockoffsets[end]-1, columnblockoffsets[end]-1)`.
- `rowblockoffsets::Vector{Int}`: 1-based start row of each block row, terminated by `m + 1`.
- `columnblockoffsets::Vector{Int}`: 1-based start column of each block column, terminated by `n + 1`.

# Example
```julia
using BlockLinearSystems

bdm = DenseBlockLinearSystem{Float64}([1, 3, 6]) # Two blocks of sizes 2 and 3
block(bdm, 2, 1) .= reshape(1.0:6.0, 3, 2)
```
"""
DenseBlockLinearSystem

"""
    block(M, i, j)
    block(M, i, j, rows, cols)

Returns a mutable view into block `(i, j)` of `M` (`SparseBlockLinearSystem` or `DenseBlockLinearSystem`).
When `rows` and/or `cols` are passed as `StaticInt`, returns a stack-wrapped `SizedMatrix` or `HybridArray` view for SIMD-accelerated block operations.

# Example
```julia
using BlockLinearSystems, Static

bdm = DenseBlockLinearSystem{Float64}([1, 3, 5])
B = block(bdm, 1, 1, static(2), static(2))
B .= 1.0
```
"""
block

"""
    validblock(M, i, j) -> Bool

Returns `true` if block `(i, j)` is structurally present in `M` (always `true` for [`DenseBlockLinearSystem`](@ref), and `M.indicestransposed[j, i] != 0` for [`SparseBlockLinearSystem`](@ref)).

# Example
```julia
using BlockLinearSystems, SparseArrays

bsm = SparseBlockLinearSystem{Float64}(sparse([true false; true true]), [2, 2], [2, 2])
validblock(bsm, 1, 2) # true
validblock(bsm, 2, 1) # false
```
"""
validblock

"""
    zero!(M)

Fills all stored entries of `M` (`SparseBlockLinearSystem`, `DenseBlockLinearSystem`, or `AbstractArray`) with zero in-place.

# Example
```julia
using BlockLinearSystems

bdm = DenseBlockLinearSystem{Float64}([1, 3])
bdm[1, 1] = 5.0
zero!(bdm)
```
"""
zero!

"""
    copy_structure(M)

Creates a new [`SparseBlockLinearSystem`](@ref) or [`DenseBlockLinearSystem`](@ref) sharing the block offset and sparsity index structures of `M`, with a freshly allocated zeroed `data` buffer.

# Example
```julia
using BlockLinearSystems

bdm = DenseBlockLinearSystem{Float64}([1, 3, 5])
bdm_copy = copy_structure(bdm)
```
"""
copy_structure

"""
    uniformscaling!(M, k)

Adds scalar `k` to the main diagonal of square matrix `M` (`SparseBlockLinearSystem`, `DenseBlockLinearSystem`, or `AbstractMatrix`) in-place: `M[i, i] += k`.

# Example
```julia
using BlockLinearSystems

bdm = DenseBlockLinearSystem{Float64}([1, 3, 5])
uniformscaling!(bdm, 2.0)
```
"""
uniformscaling!

"""
    symmetrifyfull(M) -> Matrix{T}

Constructs a full symmetric `Matrix{T}` from a lower-triangular block matrix `M` (`SparseBlockLinearSystem` or `DenseBlockLinearSystem`).
Allocates and returns a new symmetric `Matrix{T}`. See [`symmetrifyfull!`](@ref) for the in-place version.

# Example
```julia
using BlockLinearSystems

bdm = DenseBlockLinearSystem{Float64}([1, 3, 5])
bdm[3, 1] = 4.0
mat = symmetrifyfull(bdm) # mat[1, 3] == 4.0, mat !== bdm.data
```
"""
symmetrifyfull

"""
    symmetrifyfull!(bdm::DenseBlockLinearSystem) -> AbstractMatrix{T}
    symmetrifyfull!(mat::AbstractMatrix{T}, bsm::SparseBlockLinearSystem{T}, blocks=nothing)

Symmetrifies a lower-triangular block matrix in-place:
- For [`DenseBlockLinearSystem`](@ref): mirrors the lower triangle to the upper triangle of `bdm.data` in-place and returns `bdm.data`.
- For [`SparseBlockLinearSystem`](@ref): populates preallocated dense matrix `mat` in-place. If `blocks` is a vector of block indices, extracts and symmetrifies only the principal submatrix corresponding to `blocks`.

# Example
```julia
using BlockLinearSystems, SparseArrays

bsm = SparseBlockLinearSystem{Float64}(sparse([true true; false true]), [2, 2], [2, 2])
block(bsm, 2, 1) .= 3.0
mat = zeros(4, 4)
symmetrifyfull!(mat, bsm)
```
"""
symmetrifyfull!

"""
    symmetrifysparse(bsm::SparseBlockLinearSystem{T}) -> SparseMatrixCSC{T, Int}

Converts a block-lower-triangular [`SparseBlockLinearSystem`](@ref) into a full symmetric `SparseMatrixCSC{T, Int}` by mirroring off-diagonal blocks to the upper triangle.

# Example
```julia
using BlockLinearSystems, SparseArrays

bsm = SparseBlockLinearSystem{Float64}(sparse([true true; false true]), [2, 2], [2, 2])
block(bsm, 2, 1) .= 5.0
S = symmetrifysparse(bsm)
```
"""
symmetrifysparse

"""
    makesparseindices(bsm::SparseBlockLinearSystem, symmetrify::Bool=false) -> SparseMatrixCSC{Int, Int}

Precomputes a `SparseMatrixCSC{Int, Int}` whose `nzval` holds the 1-based indices into `bsm.data` for every scalar non-zero entry of `sparse(bsm)` (or `symmetrifysparse(bsm)` when `symmetrify=true`).

This enables zero-allocation copying from `bsm.data` into an existing `SparseMatrixCSC` via `H.nzval[i] = bsm.data[sparseindices.nzval[i]]`.

# Example
```julia
using BlockLinearSystems, SparseArrays

bsm = SparseBlockLinearSystem{Float64}(sparse([true true; false true]), [2, 2], [2, 2])
idx_csc = makesparseindices(bsm, true)
S = sparse(bsm, idx_csc)
```
"""
makesparseindices

"""
    blocksizes(sys, dim) -> Vector{UInt8}
    blocksizes(sys) -> Tuple{Vector{UInt8}, Vector{UInt8}}

Returns the block dimensions along dimension `dim` (1 for rows, 2 for columns), or a tuple of `(rowblocksizes, columnblocksizes)`.

# Example
```julia
using BlockLinearSystems

sys = DenseBlockLinearSystem{Float64}([2, 3])
blocksizes(sys, 1) # [2, 3]
```
"""
blocksizes

"""
    nblocks(sys, dim) -> Int
    nblocks(sys) -> Tuple{Int, Int}

Returns the number of blocks along dimension `dim`, or `(n_rowblocks, n_colblocks)`.

# Example
```julia
using BlockLinearSystems

sys = DenseBlockLinearSystem{Float64}([2, 3])
nblocks(sys, 1) # 2
nblocks(sys)    # (2, 2)
```
"""
nblocks

"""
    blockdata(sys) -> Array

Returns the underlying data storage buffer (a contiguous flat `Vector{T}` for [`SparseBlockLinearSystem`](@ref) or `AbstractMatrix{T}` for [`DenseBlockLinearSystem`](@ref)).
"""
blockdata

"""
    eachstoredblock(sys)

Returns an iterator yielding `(i, j, block_view)` for every structurally non-zero block in `sys`.

# Example
```julia
using BlockLinearSystems

sys = DenseBlockLinearSystem{Float64}([2, 2])
for (i, j, B) in eachstoredblock(sys)
    B .= 1.0
end
```
"""
eachstoredblock

"""
    block_sparse_nnz(sparsity, blocksizes) -> Int

Computes the total number of scalar non-zeros in a block-sparse matrix given its block `sparsity::SparseMatrixCSC` pattern and `blocksizes` vector.

# Example
```julia
using BlockLinearSystems, SparseArrays

sparsity = sparse([true true; false true])
block_sparse_nnz(sparsity, [2, 3]) # 2*2 + 2*3 + 3*3 = 19
```
"""
block_sparse_nnz

"""
    computestartindices(blocksizes) -> Vector{Int}

Computes the 1-based starting scalar index for each block given a vector of `blocksizes`.

# Example
```julia
using BlockLinearSystems

computestartindices([2, 3, 4]) # [1, 3, 6]
```
"""
computestartindices

"""
    find_diag_inds(A::SparseMatrixCSC) -> Vector{Int}

Finds the index `k` in `A.nzval` corresponding to the diagonal entry `A[col, col]` for each column `col in 1:size(A, 2)`.

# Example
```julia
using BlockLinearSystems, SparseArrays

S = sparse([1, 2, 2], [1, 1, 2], [10.0, 20.0, 30.0], 2, 2)
find_diag_inds(S) # [1, 3]
```
"""
find_diag_inds

"""
    sparse_with_map(sparse_sys::SparseBlockLinearSystem{T}; symmetric::Bool=false) -> (SparseMatrixCSC{T, Int}, Vector{Int})
    sparse_with_map(sparse_sys::SparseBlockLinearSystem{T}, sparse_idx::SparseMatrixCSC{Int, Int}) -> (SparseMatrixCSC{T, Int}, Vector{Int})

Precomputes a standard `SparseMatrixCSC{T, Int}` alongside a `scatter_map::Vector{Int}`.
The `scatter_map` maps entries from `sparse_sys.data` to the CSC nonzeros vector.

Subsequent updates can be performed in-place with **zero allocations** via:
```julia
copyto!(csc_mat, sparse_sys, scatter_map)
```

# Example
```julia
using BlockLinearSystems, SparseArrays

sys = SparseBlockLinearSystem{Float64}(sparse([true true; true true]), [2, 2])
block(sys, 1, 1) .= 1.0
csc_mat, scatter_map = sparse_with_map(sys; symmetric=false)

# In-place update in solver loop (0 allocations):
block(sys, 1, 1) .= 2.0
copyto!(csc_mat, sys, scatter_map)
```
"""
sparse_with_map

"""
    BlockRLE(runlengths, blocksizes)
    BlockRLE(runs::Vector{Pair})
    BlockRLE(vec::AbstractVector{<:Integer})

Run-Length Encoded (RLE) block layout metadata capturing contiguous runs of uniform block sizes.
Subtypes `AbstractVector{Int}` (storing the sizes compactly, normally as `UInt8`, but always returning `Int`) for transparent interop with all linear systems while providing
\$O(1)\$ matrix dimension lookup, \$O(1)\$ run indexing, and \$O(\\log R)\$ block-to-coordinate mapping.

# Fields
- `runlengths::Vector{Int}`: Number of blocks in each run.
- `blocksizes::Vector{UInt8}`: Block dimension in each run.
- `cumblocks::Vector{Int}`: 1-based start block index for each run, terminated by `total_blocks + 1`.
- `cummatrixdims::Vector{Int}`: 1-based start scalar matrix coordinate for each run, terminated by `total_dimension + 1`.

# Example
```julia
# 100 blocks of dimension 6 followed by 50,000 blocks of dimension 3:
rle = BlockRLE([100, 50_000], [6, 3])
length(rle)            # 50100
matrixdimension(rle)   # 150600
nruns(rle)             # 2
```
"""
BlockRLE

"""
    nruns(rle::BlockRLE) -> Int

Returns the number of uniform block runs in `rle`.
"""
nruns

"""
    runlength(rle::BlockRLE, k::Integer) -> Int

Returns the number of blocks in run `k`.
"""
runlength

"""
    runblocksize(rle::BlockRLE, k::Integer) -> Int

Returns the block dimension for blocks in run `k`.
"""
runblocksize

"""
    matrixdimension(rle::BlockRLE) -> Int

Returns the total scalar matrix row/column dimension represented by `rle` (equivalent to `sum(rle)`).
"""
matrixdimension

"""
    runblockrange(rle::BlockRLE, k::Integer) -> UnitRange{Int}

Returns the 1-based range of block indices belonging to run `k`.
"""
runblockrange

"""
    runmatrixrange(rle::BlockRLE, k::Integer) -> UnitRange{Int}

Returns the 1-based range of scalar rows/columns in the represented matrix belonging to run `k`.
"""
runmatrixrange

"""
    eachrun(rle::BlockRLE)

Iterator yielding named tuples `(run = k, blockrange = r, blocksize = s, matrixrange = m)`
for each uniform run in `rle`.
"""
eachrun

"""
    sort_blocks_hierarchical(roles::AbstractVector, dimensions::AbstractVector{<:Integer}; rev_dim::Bool=true)
        -> (permutation::Vector{Int}, rle::BlockRLE)

Sorts blocks using a two-tier hierarchy `(Role, Dimension)` to cleanly separate Schur-eliminated
variables from retained variables while grouping identical block dimensions into contiguous RLE runs.

Returns the permutation vector `permutation` and the resulting `BlockRLE` metadata.
"""
sort_blocks_hierarchical

"""
    blockrange(first, len) -> AbstractUnitRange

Index range of a block of `len` entries starting at index `first`. A `StaticInt` `len` gives a statically
sized range; any other integer `len` gives a plain `UnitRange{Int}`, so views built with it are type stable
and allocation free even when the size is only known at run time. Integer sizes of any type (including
the `UInt8` sizes stored by `BlockRLE`) are widened to `Int` before use.
"""
blockrange

"""
    SchurJacobianStructure(structure::SchurStructure, e_rows; free_rows=Int[], free_retained=Vector{Int}[], nchunks)
    SchurJacobianStructure(r_dims, e_dims, couplings, e_rows; free_rows=Int[], free_retained=Vector{Int}[], nchunks=1)

Layout of a tall linear least-squares problem `min ‖J x - r‖²`, `x = [x_r; x_e]` (retained variables first),
partitioned like `structure` (same retained and eliminated blocks, same couplings, same block order).
It is the Jacobian-path counterpart of a [`SchurStructure`](@ref), and holds one, unchanged, in its
`schur` field.

Every row of `J` involves at most one eliminated block (the Schur condition). The rows that involve
eliminated block `e` form its *table*, `e_rows[e]` rows by the columns of every retained block coupled to
`e`, then `e`'s own columns, then the residual:

```
T_e = [ J_r | J_e | r ]        (e_rows[e] + d_e) × (c_e + d_e + 1)
```

where `c_e` is the sum of the coupled retained dimensions (at most `structure.max_coupled_dim`). The last
`d_e` rows are damping rows, zero in the raw storage and written by [`reduce_qr!`](@ref). Rows that
involve no eliminated block are given as *free groups*: group `g` has `free_rows[g]` rows and the retained
blocks `free_retained[g]`, and its table is `[ J_r | r ]`.

Tables are numbered `t = 1:ntables`: table `e` (`e <= structure.neblocks`) belongs to eliminated block
`e`, table `structure.neblocks + g` to free group `g`. A table's retained blocks are its *entries*,
numbered globally: entries `1:length(structure.entry_retained)` are the coupling entries of `structure`
(same index `c`, same order, sorted by retained block within each eliminated block), followed by the
entries of the free groups (retained blocks sorted within each group). [`coupling_entry`](@ref) finds an
entry from its table and retained block.

`nchunks` sets how many disjoint ranges of retained blocks (`col_chunks`) the column-ownership kernels are
threaded over; it defaults to the number of chunks of `structure`.

Throws `DimensionMismatch` if `e_rows` (or `free_retained`) has the wrong length, and `ArgumentError` for
a table with no rows, or a free group that references a retained block that does not exist or lists one
twice.

# Example
```julia
using BlockLinearSystems
# Two retained blocks (6 and 3), two eliminated blocks (3 and 3); 4 and 6 rows each; one free group of 2 rows
jst = SchurJacobianStructure([6, 3], [3, 3], [[1, 2], [2]], [4, 6]; free_rows=[2], free_retained=[[1]])
jst.ntables          # 3
reduced_nrows(jst)   # 4 + 6 + 2
```
"""
SchurJacobianStructure

"""
    SchurBlockJacobian{T}(structure::SchurJacobianStructure)
    SchurBlockJacobian(data::Vector{T}, structure::SchurJacobianStructure)

Raw storage of a tall least-squares problem: the tables of `structure` (see
[`SchurJacobianStructure`](@ref)), each column-major, in one flat vector `data` at
`structure.table_offsets`. It is the Jacobian prototype a caller fills, the analogue of
[`SchurBlockMatrix`](@ref) on the Hessian path. `T` may be any floating-point type; `Float64` and
`Float32` are tested.

Fill it through [`coupling_block`](@ref), [`eliminated_block`](@ref) and [`residual_block`](@ref). The
rows of a table belong to one eliminated block (or one free group), so threads that fill different
tables never write the same memory and need no per-thread copies. The damping rows are not reachable
through the views and stay zero. [`reduce_qr!`](@ref) reads these tables and never modifies them, so a
change of the damping re-factors from them.

[`zero!`](@ref) zeroes every table; [`copy_structure`](@ref) returns a new, zeroed `SchurBlockJacobian`
sharing only the (immutable) structure. [`schur_jacobian_bytes`](@ref) gives the memory needed before
anything is allocated.
"""
SchurBlockJacobian

"""
    schur_jacobian_bytes(structure::SchurJacobianStructure, T=Float64; workspace=true) -> Int

Bytes of element data a [`SchurBlockJacobian{T}`](@ref SchurBlockJacobian) for `structure` allocates
and, with `workspace=true`, its [`QRReductionWorkspace{T}`](@ref QRReductionWorkspace) too (a factored
copy of every table, plus scratch for eliminated blocks above `BlockLinearSystems.D_STATIC`
dimensions). Computed from the structure alone, so a caller can refuse, or choose the Schur route,
before allocating anything.

Each table is dense in every retained block coupled to its eliminated block, so its size is
`(m_e + d_e) × (c_e + d_e + 1)`: the memory grows with the number of rows times the coupled retained
dimension, which for an eliminated block seen by many residuals is far more than the Schur route's
`d_e × c_e` coupling blocks. The index arrays of the structure itself are not counted.
"""
schur_jacobian_bytes

"""
    coupling_block(J::SchurBlockJacobian, u)
    coupling_block(J::SchurBlockJacobian, u, row, nrows[, ncols])

Mutable view of entry `u`'s retained block in its table (`u` indexes entries as described in
[`SchurJacobianStructure`](@ref); use [`coupling_entry`](@ref) to find it): all of the table's rows, or
rows `row:row+nrows-1` (counted from 1 within the table). The view has `nrows × d_r` entries; `ncols`, if
given, must equal `d_r`.

`nrows` and `ncols` may be `Int`, `StaticInt` or `Val`. With both static the view is a `SizedMatrix`,
with one static a `HybridArray`, as for [`block`](@ref); all forms are type stable and allocation free.
Throws `BoundsError` for an unknown entry or rows outside the table's data rows (the damping rows are
never reachable), and `DimensionMismatch` for a wrong `ncols`.

# Example
```julia
using BlockLinearSystems, Static, StaticArrays
J = SchurBlockJacobian{Float64}(SchurJacobianStructure([6, 3], [3, 3], [[1, 2], [2]], [4, 6]))
u = coupling_entry(J.structure, 1, 2)                 # retained block 2 in eliminated block 1's table
coupling_block(J, u, 3, static(2), static(3)) .= @SMatrix rand(2, 3)   # rows 3:4 of that residual
```
"""
coupling_block

"""
    eliminated_block(J::SchurBlockJacobian, e)
    eliminated_block(J::SchurBlockJacobian, e, row, nrows[, ncols])

Mutable view of the eliminated columns `J_e` of table `e` (an eliminated block, so `e <=
J.structure.schur.neblocks`): all of its data rows, or rows `row:row+nrows-1`. Sizes and errors as for
[`coupling_block`](@ref), with `ncols == d_e`.
"""
eliminated_block

"""
    residual_block(J::SchurBlockJacobian, t)
    residual_block(J::SchurBlockJacobian, t, row, nrows)

Mutable view of the residual column of table `t` (an eliminated block's table or a free group's): all of
its data rows, or rows `row:row+nrows-1`. A `SizedVector` for a static `nrows`, otherwise a plain view.
The residual is `r` of `min ‖J x - r‖²`: callers that hold `f(x) ≈ f(x0) + J dx` store `r = -f(x0)`, or
store `f(x0)` and negate the step, as with the Schur systems.
"""
residual_block

"""
    coupling_entry(structure::SchurJacobianStructure, t, r) -> Int

Index of the entry of retained block `r` in table `t` (binary search). Throws `ArgumentError` if `r` is
not coupled to table `t`, `BoundsError` for an unknown table.
"""
coupling_entry

"""
    QRReductionWorkspace{T}(structure::SchurJacobianStructure)
    QRReductionWorkspace(J::SchurBlockJacobian)

Storage for QR elimination, the analogue of [`SchurReductionWorkspace`](@ref): `factored`, a factored
copy of every table in the layout of the raw storage (free tables are copied unchanged), per-chunk
scratch for back-substitution of eliminated blocks above `BlockLinearSystems.D_STATIC` dimensions, and
the failure flag `failed_block` (an `Atomic{Int}`: 0, or the lowest failing eliminated block).

After [`reduce_qr!`](@ref), the top `d_e` rows of table `e` hold `[B | U | q]` with `U` upper
triangular, and the rows below hold `[J̃_e | 0 | r̃_e]`. The Householder reflectors are not kept (nothing
needs `Q` afterwards), so the eliminated columns hold exactly `[U; 0]` and each factored table equals
`Qᵀ` times the damped raw table. All the kernels of the reduced problem read these tables.

[`trial_workspace`](@ref) gives a second workspace for a trial system with the same structure.
"""
QRReductionWorkspace

"""
    trial_workspace(ws::QRReductionWorkspace)

A workspace for a second system with the same structure: its own factored tables and failure flag (the
factored tables are needed for back-substitution after the solve), with the scratch shared.
"""
trial_workspace(::QRReductionWorkspace)

"""
    reduce_qr!(ws::QRReductionWorkspace{T}, J::SchurBlockJacobian{T}, scaling_e, lambda) -> Bool

Eliminate every eliminated block of `J` by QR, into `ws`. For each eliminated block `e`:

1. copy its raw table to `ws.factored`;
2. write the damping rows `[0 | √(lambda * scaling_e[e]) | 0]` (a diagonal `d_e × d_e` block), so that the
   factored table represents `[J; √λ·D]` with `D = Diagonal(√scaling_e)`;
3. Householder-factor the eliminated columns, applying `Qᵀ` to the whole table in place.

This solves the eliminated part of `(JᵀJ + λ·Diagonal(s)) x = Jᵀr`; `scaling_e` (length
`sum(e_dims)`) is the eliminated part of `s`, in the units of `diag(JᵀJ)` (for Marquardt scaling,
[`jtj_diag!`](@ref)). The retained part of the damping is the caller's business: append rows to `J̃`, or
add `λ·s_r` to the diagonal of `J̃ᵀJ̃`. `lambda = 0` is undamped.

Returns `false` if an eliminated block is numerically singular — `|U_kk| ≤ M·eps(T)·‖column k‖`, with
`M` the table's row count (the rounding left on an exactly dependent column grows with the number of
rows), which catches under-determined (`e_rows[e] < d_e`), rank-deficient and zero blocks without damping
— or its table contains a non-finite value; `ws.failed_block[]` then holds the lowest failing block, as
for the Schur reduction. Free tables are copied, not checked.

Statically sized in the eliminated dimension (unrolled over the columns) up to
`BlockLinearSystems.D_STATIC`, dynamic above, through the `e_groups` and `chunks` of the structure.
Threaded over the chunks of eliminated blocks when there is enough work
(`BlockLinearSystems.SCHUR_THREAD_MIN_WORK`); tables are independent, so there is no merge step. The
raw tables are untouched, so calling it again with another `lambda` re-factors from them; it allocates
nothing on the serial path (a threaded launch costs Polyester a small constant allocation).
"""
reduce_qr!

"""
    backsubstitute!(x_e, x_r, ws::QRReductionWorkspace)

Recover the eliminated variables from the retained ones after [`reduce_qr!`](@ref): `x_e = U⁻¹ (q - B x_r)`
for each eliminated block, from the top rows of its factored table. Equals the Schur
`x_e = E⁻¹ (b_e - C x_r)` of `H = JᵀJ + λ·Diagonal(s)`. Static up to `D_STATIC` dimensions, dynamic above
(using the workspace scratch), threaded over chunks of eliminated blocks; allocation free on the serial path.
"""
backsubstitute!(::AbstractVector, ::AbstractVector, ::QRReductionWorkspace)

"""
    reduced_nrows(structure::SchurJacobianStructure) -> Int

Number of rows of the reduced Jacobian `J̃`: `e_rows[e]` rows per eliminated block, then the free rows.
Fixed by the structure, whatever `λ` is: each table has `e_rows[e] + d_e` rows, of which `d_e` become `U`.
When `λ = 0` the damping rows are zero and remain so, so `d_e` of each block's reduced rows are zero (and
contribute nothing).
"""
reduced_nrows

"""
    free_block_pairs(structure::SchurJacobianStructure) -> Vector{Tuple{Int, Int}}

The pairs `(r1, r2)`, `r1 < r2`, of retained blocks coupled through a free group. Pass them as
`extra_blocks` to [`build_schur_pattern`](@ref), so that the sparse reduced system has a slot for them:
`build_schur_pattern(structure.schur; extra_blocks=free_block_pairs(structure))`.
"""
free_block_pairs

"""
    reduced_mul!(y, ws::QRReductionWorkspace, v) -> y

`y = J̃ v`, with `v` over the retained variables (length `nr`) and `y` over the reduced rows (length
[`reduced_nrows`](@ref)), in the documented order: the bottom rows of eliminated block 1's table, block 2's,
…, then each free group's rows. Threaded over tables, allocation free on the serial path.
"""
reduced_mul!

"""
    reduced_tmul!(g, ws::QRReductionWorkspace, w) -> g

`g = J̃ᵀ w`, with `w` over the reduced rows (in the order of [`reduced_mul!`](@ref)). Threaded by retained
column ownership (`structure.col_chunks`), so no two threads write the same entry.
"""
reduced_tmul!

"""
    reduced_residual!(rt, ws::QRReductionWorkspace) -> rt

`r̃`, the right-hand side of the reduced problem `min ‖J̃ x_r - r̃‖²`, in the row order of
[`reduced_mul!`](@ref).
"""
reduced_residual!

"""
    reduced_jacobian!(Jt::AbstractMatrix, rt::AbstractVector, ws::QRReductionWorkspace) -> Jt

Write the reduced problem densely: `Jt = J̃` (size `(reduced_nrows(structure), nr)`, overwritten) and
`rt = r̃`, for a dense QR of the reduced problem. Solving that by QR, then [`backsubstitute!`](@ref),
keeps the whole solve at the condition number of `J`; forming `J̃ᵀJ̃` squares it.
"""
reduced_jacobian!

"""
    reduced_normal_dense!(S::Matrix, g_S, ws::QRReductionWorkspace) -> true

Overwrite `S` with the upper triangle of `J̃ᵀJ̃` (the strict lower triangle is zeroed) and `g_S` with
`J̃ᵀr̃`. These equal the Schur complement `S` and right-hand side `g_S` that [`reduce_schur_dense!`](@ref)
produces from `H = JᵀJ` and `b = Jᵀr`, apart from the retained damping, which the Schur route adds and this
one leaves to the caller (`S[i, i] += λ * s_r[i]`). Threaded by retained column ownership, as
`accumulate_dense_columns!` is.
"""
reduced_normal_dense!

"""
    reduced_normal_sparse!(S::SparseMatrixCSC, g_S, ws::QRReductionWorkspace, pattern) -> true

Sparse counterpart of [`reduced_normal_dense!`](@ref): overwrite `S.nzval` (upper triangle) and `g_S`.
`pattern` and `S` come from `build_schur_pattern(structure.schur, T; extra_blocks=free_block_pairs(structure))`
(copy `pattern.S_sparse` for `S`); add the retained damping at `pattern.diag_inds`. Throws an
`AssertionError` if the pattern lacks a needed block.
"""
reduced_normal_sparse!

"""
    reduced_diagonal_blocks!(D::AbstractVector, ws::QRReductionWorkspace) -> D

The diagonal blocks of `J̃ᵀJ̃`, one full (both triangles) column-major `d_r × d_r` block per retained
block, block `r` at `D[structure.doffsets[r]:structure.doffsets[r+1]-1]`; for block-Jacobi
preconditioning of an iterative solve on `J̃`.
"""
reduced_diagonal_blocks!

"""
    jtr!(g, src::Union{SchurBlockJacobian, QRReductionWorkspace}) -> g

The gradient `g = Jᵀr` of the full problem, `[g_r; g_e]` (length `nr + sum(e_dims)`), from the raw tables
or the factored ones: `Q` is orthogonal and the damping rows have zero residual, so both give the same
result (to rounding), for any `λ`.
"""
jtr!

"""
    jtj_diag!(d, src::Union{SchurBlockJacobian, QRReductionWorkspace}) -> d

The squared column norms of `J`, i.e. `diag(JᵀJ)`, `[d_r; d_e]`, for Marquardt scaling and for the
largest diagonal entry. From the raw tables this is `diag(JᵀJ)`; from the factored tables it is that of the
damped Jacobian `[J; √(λ s_e)]`, so the eliminated entries include `λ * s_e` (equal at `λ = 0`).
"""
jtj_diag!

"""
    fast_bAb(src::Union{SchurBlockJacobian, QRReductionWorkspace}, v) -> Real

`‖J v‖² = vᵀ(JᵀJ)v` for `v = [v_r; v_e]`, from the tables, without forming anything: the analogue of
[`fast_bAb`](@ref)`(::SchurBlockMatrix, dx)` for the Hessian `JᵀJ`. From the factored tables the damped
Jacobian is used, so the result includes `λ * Σ s_e v_e²` (equal at `λ = 0`). Any element type of `v`;
the result has the promoted type.
"""
fast_bAb(::Union{SchurBlockJacobian, QRReductionWorkspace}, ::AbstractVector)
