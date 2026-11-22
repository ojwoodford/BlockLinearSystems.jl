"""
    SparseBlockLinearSystem{T}(sparsitytransposed::SparseMatrixCSC{Bool, Int}, rowblocksizes, colblocksizes=rowblocksizes)
    SparseBlockLinearSystem{T}(nnzvals, indicestransposed, rowblocksizes, colblocksizes=rowblocksizes)

Cache-friendly block-sparse matrix storing each non-zero block contiguously in column-major order within a flat `data::Vector{T}` buffer.

The block sparsity pattern is provided transposed (`sparsitytransposed` has size `(length(colblocksizes), length(rowblocksizes))`) so that blocks along a block-row are contiguous in the CSC structure of `indicestransposed`.

# Fields
- `data::Vector{T}`: Contiguous storage for all non-zero block entries.
- `indices::SparseMatrixCSC{Int, Int}`: Cached start indices in `data` for each block `(row, col)`.
- `indicestransposed::SparseMatrixCSC{Int, Int}`: Transposed start indices in `data` (`indicestransposed[col, row]`).
- `rowblocksizes::BlockRLE`: Row dimensions of each block row (1 to 65535).
- `columnblocksizes::BlockRLE`: Column dimensions of each block column (1 to 65535).
- `m::Int`: Total number of scalar rows.
- `n::Int`: Total number of scalar columns.

# Example
```jldoctest
julia> using BlockLinearSystems, SparseArrays

julia> bsm = SparseBlockLinearSystem{Float64}(sparse([true false; true true]), [2, 3], [2, 3]);

julia> block(bsm, 1, 1) .= [1.0 2.0; 2.0 4.0];

julia> size(bsm.data)   # blocks (1, 1), (2, 1) and (2, 2): 4 + 6 + 9 entries
(19,)
```
"""
SparseBlockLinearSystem

"""
    DenseBlockLinearSystem{T}(rowblocksizes, columnblocksizes=rowblocksizes)
    DenseBlockLinearSystem{T}(data::AbstractMatrix{T}, rowblocksizes, columnblocksizes=rowblocksizes)
    DenseBlockLinearSystem(data::AbstractMatrix{T}, rowblocksizes, columnblocksizes=rowblocksizes)
    DenseBlockLinearSystem(data::AbstractMatrix{T}, like::DenseBlockLinearSystem)

Block-partitioned dense matrix backed by a matrix buffer `data` (such as `Matrix{T}`, `SubArray`, or `FixedSizeMatrix{T}`), partitioned into blocks of the sizes given (each 1 to 65535). Without `data`, a zeroed `Matrix{T}` is allocated.

With `like`, the system takes the block layout of `like`, which it shares rather than copies, and allocates nothing: use it to give a matrix you are handed the layout of a system you already have. `data` must have the size of `like`, and may have any element type and storage; it is used as it is, not zeroed.

# Fields
- `data::AbstractMatrix{T}`: Underlying matrix buffer of size `(sum(rowblocksizes), sum(columnblocksizes))`.
- `rowblockoffsets::Vector{Int}`: 1-based start row of each block row, terminated by `m + 1`.
- `columnblockoffsets::Vector{Int}`: 1-based start column of each block column, terminated by `n + 1`.
- `rowblocksizes::BlockRLE`: Row dimensions of each block row.
- `columnblocksizes::BlockRLE`: Column dimensions of each block column.

# Example
```jldoctest
julia> using BlockLinearSystems

julia> bdm = DenseBlockLinearSystem{Float64}([2, 3]);   # two blocks, of sizes 2 and 3

julia> block(bdm, 2, 1) .= reshape(1.0:6.0, 3, 2);

julia> bdm.data[3:5, 1:2]
3×2 Matrix{Float64}:
 1.0  4.0
 2.0  5.0
 3.0  6.0

julia> H = zeros(5, 5);

julia> block(DenseBlockLinearSystem(H, bdm), 2, 2) .= 1.0;   # H, with the layout of bdm

julia> H[3:5, 3]
3-element Vector{Float64}:
 1.0
 1.0
 1.0
```
"""
DenseBlockLinearSystem

"""
    block(M, i, j)
    block(M, i, j, rows, cols)
    block(J::SchurBlockJacobian, k, entry, nrows, ncols)

Returns a mutable view into block `(i, j)` of `M` (`SparseBlockLinearSystem` or `DenseBlockLinearSystem`).
When `rows` and/or `cols` are passed as `StaticInt`, returns a stack-wrapped `SizedMatrix` or `HybridArray` view for SIMD-accelerated block operations.

For a [`SchurBlockJacobian`](@ref) whose structure was built from a description, the view of row block `k`'s
rows of `J` in the columns of `entry`, a handle from [`blockentry`](@ref)`(J.structure, k, j)` for layout column
block `j`: `nrows × ncols`, which must be the row block's number of rows and the column block's size. Sizes may
be `Int`, `StaticInt` or `Val`; both static give a `SizedMatrix`, one a `HybridArray`. Sizes and the handle are
checked under `@boundscheck`; the view allocates nothing. The rows of different row blocks are disjoint, so a
fill may be threaded over row blocks.

# Example
```jldoctest
julia> using BlockLinearSystems, Static

julia> bdm = DenseBlockLinearSystem{Float64}([2, 2]);

julia> B = block(bdm, 1, 1, static(2), static(2));

julia> B .= 1.0;

julia> bdm.data[1:2, 1:2] == ones(2, 2)
true
```
"""
block

"""
    validblock(M, i, j) -> Bool

Returns `true` if block `(i, j)` is structurally present in `M` (always `true` for [`DenseBlockLinearSystem`](@ref), and `M.indicestransposed[j, i] != 0` for [`SparseBlockLinearSystem`](@ref)).

# Example
```jldoctest
julia> using BlockLinearSystems, SparseArrays

julia> bsm = SparseBlockLinearSystem{Float64}(sparse([true false; true true]), [2, 2], [2, 2]);

julia> BlockLinearSystems.validblock(bsm, 2, 1)
true

julia> BlockLinearSystems.validblock(bsm, 1, 2)
false
```
"""
validblock

"""
    zero!(M)

Fills all stored entries of `M` (`SparseBlockLinearSystem`, `DenseBlockLinearSystem`, or `AbstractArray`) with zero in-place.

# Example
```jldoctest
julia> using BlockLinearSystems

julia> bdm = DenseBlockLinearSystem{Float64}([2]);

julia> bdm[1, 1] = 5.0;

julia> zero!(bdm);

julia> bdm[1, 1]
0.0
```
"""
zero!

"""
    copy_structure(M)

Creates a new [`SparseBlockLinearSystem`](@ref) or [`DenseBlockLinearSystem`](@ref) sharing the block offset and sparsity index structures of `M`, with a freshly allocated zeroed `data` buffer.

# Example
```jldoctest
julia> using BlockLinearSystems

julia> bdm = DenseBlockLinearSystem{Float64}([2, 2]);

julia> bdm[1, 1] = 5.0;

julia> bdm_copy = copy_structure(bdm);

julia> bdm_copy[1, 1], blocksizes(bdm_copy, 1)
(0.0, [2, 2])
```
"""
copy_structure

"""
    uniformscaling!(M, k)

Adds scalar `k` to the main diagonal of square matrix `M` (`SparseBlockLinearSystem`, `DenseBlockLinearSystem`, or `AbstractMatrix`) in-place: `M[i, i] += k`.

# Example
```jldoctest
julia> using BlockLinearSystems

julia> bdm = DenseBlockLinearSystem{Float64}([2, 2]);

julia> uniformscaling!(bdm, 2.0);

julia> bdm[3, 3], bdm[3, 1]
(2.0, 0.0)
```
"""
uniformscaling!

"""
    symmetrifyfull(M) -> Matrix{T}

Constructs a full symmetric `Matrix{T}` from a lower-triangular block matrix `M` (`SparseBlockLinearSystem` or `DenseBlockLinearSystem`).
Allocates and returns a new symmetric `Matrix{T}`. See [`symmetrifyfull!`](@ref) for the in-place version.

# Example
```jldoctest
julia> using BlockLinearSystems

julia> bdm = DenseBlockLinearSystem{Float64}([2, 2]);

julia> bdm[3, 1] = 4.0;

julia> mat = symmetrifyfull(bdm);

julia> mat[1, 3], mat !== bdm.data
(4.0, true)
```
"""
symmetrifyfull

"""
    symmetrifyfull!(bdm::DenseBlockLinearSystem) -> AbstractMatrix{T}
    symmetrifyfull!(mat::AbstractMatrix{T}, bsm::SparseBlockLinearSystem{T}, blocks=nothing)

Symmetrifies a lower-triangular block matrix in-place:
- For [`DenseBlockLinearSystem`](@ref): mirrors the lower triangle to the upper triangle of `bdm.data` in-place and returns `bdm.data`.
- For [`SparseBlockLinearSystem`](@ref): populates preallocated dense matrix `mat` in-place. If `blocks` is a vector of block indices, extracts and symmetrifies only the principal submatrix corresponding to `blocks`, with the blocks in the order listed (a block listed twice is an `ArgumentError`).

# Example
```jldoctest
julia> using BlockLinearSystems, SparseArrays

julia> bsm = SparseBlockLinearSystem{Float64}(sparse([true false; true true]), [2, 2], [2, 2]);

julia> block(bsm, 2, 1) .= 3.0;

julia> mat = zeros(4, 4);

julia> symmetrifyfull!(mat, bsm);

julia> mat[1, 3], mat[3, 1]
(3.0, 3.0)
```
"""
symmetrifyfull!

"""
    symmetrifysparse(bsm::SparseBlockLinearSystem{T}) -> SparseMatrixCSC{T, Int}

Converts a block-lower-triangular [`SparseBlockLinearSystem`](@ref) into a full symmetric `SparseMatrixCSC{T, Int}` by mirroring off-diagonal blocks to the upper triangle.

# Example
```jldoctest
julia> using BlockLinearSystems, SparseArrays

julia> bsm = SparseBlockLinearSystem{Float64}(sparse([true false; true true]), [2, 2], [2, 2]);

julia> block(bsm, 2, 1) .= 5.0;

julia> S = BlockLinearSystems.symmetrifysparse(bsm);

julia> S[1, 3], S[3, 1], nnz(S)
(5.0, 5.0, 16)
```
"""
symmetrifysparse

"""
    makesparseindices(bsm::SparseBlockLinearSystem, symmetrify::Bool=false) -> SparseMatrixCSC{Int, Int}

Precomputes a `SparseMatrixCSC{Int, Int}` whose `nzval` holds the 1-based indices into `bsm.data` for every scalar non-zero entry of `sparse(bsm)` (or of its symmetric completion when `symmetrify=true`).

This enables zero-allocation copying from `bsm.data` into an existing `SparseMatrixCSC` via `H.nzval[i] = bsm.data[sparseindices.nzval[i]]`.

# Example
```jldoctest
julia> using BlockLinearSystems, SparseArrays

julia> bsm = SparseBlockLinearSystem{Float64}(sparse([true false; true true]), [2, 2], [2, 2]);

julia> idx_csc = BlockLinearSystems.makesparseindices(bsm, true);

julia> S = sparse(bsm, idx_csc);

julia> size(S), nnz(S)
((4, 4), 16)
```
"""
makesparseindices

"""
    blocksizes(sys, dim) -> BlockRLE
    blocksizes(sys) -> Tuple{BlockRLE, BlockRLE}
    blocksizes(structure::Union{SchurStructure, SchurJacobianStructure}) -> Vector{Int}

Returns the block dimensions along dimension `dim` (1 for rows, 2 for columns), or a tuple of `(rowblocksizes, columnblocksizes)`.
For a [`SchurStructure`](@ref) or [`SchurJacobianStructure`](@ref), the sizes of its column blocks in layout order
(retained, then eliminated), as a new `Vector{Int}`: for setup.

# Example
```jldoctest
julia> using BlockLinearSystems

julia> sys = DenseBlockLinearSystem{Float64}([2, 3]);

julia> blocksizes(sys, 1)
2-element BlockRLE{UInt16, Int64}:
 2
 3
```
"""
blocksizes

"""
    nblocks(sys, dim) -> Int
    nblocks(sys) -> Tuple{Int, Int}
    nblocks(structure::Union{SchurStructure, SchurJacobianStructure}) -> Int

Returns the number of blocks along dimension `dim`, or `(n_rowblocks, n_colblocks)`. For a
[`SchurStructure`](@ref) or [`SchurJacobianStructure`](@ref), the number of its column blocks, retained and
eliminated.

# Example
```jldoctest
julia> using BlockLinearSystems

julia> sys = DenseBlockLinearSystem{Float64}([2, 3]);

julia> nblocks(sys, 1)
2

julia> nblocks(sys)
(2, 2)
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
```jldoctest
julia> using BlockLinearSystems

julia> sys = DenseBlockLinearSystem{Float64}([2, 2]);

julia> for (i, j, B) in BlockLinearSystems.eachstoredblock(sys)
           B .= i + j
       end

julia> sys.data[3, 1], sys.data[4, 4]
(3.0, 4.0)
```
"""
eachstoredblock

"""
    block_sparse_nnz(sparsity, blocksizes) -> Int

Computes the total number of scalar non-zeros in a block-sparse matrix given its block `sparsity::SparseMatrixCSC` pattern and `blocksizes` vector.

# Example
```jldoctest
julia> using BlockLinearSystems, SparseArrays

julia> block_sparse_nnz(sparse([true true; false true]), [2, 3])   # 2*2 + 2*3 + 3*3
19
```
"""
block_sparse_nnz

"""
    computestartindices(blocksizes) -> Vector{Int}

Computes the 1-based starting scalar index for each block given a vector of `blocksizes`.

# Example
```jldoctest
julia> using BlockLinearSystems

julia> computestartindices([2, 3, 4])
3-element Vector{Int64}:
 1
 3
 6
```
"""
computestartindices

"""
    find_diag_inds(A::SparseMatrixCSC) -> Vector{Int}

Finds the index `k` in `A.nzval` corresponding to the diagonal entry `A[col, col]` for each column `col in 1:size(A, 2)`.
Throws `ArgumentError`, naming the column, if a column has no stored diagonal entry.

# Example
```jldoctest
julia> using BlockLinearSystems, SparseArrays

julia> find_diag_inds(sparse([1, 2, 2], [1, 1, 2], [10.0, 20.0, 30.0], 2, 2))
2-element Vector{Int64}:
 1
 3
```
"""
find_diag_inds

"""
    sparse_with_map(sparse_sys::SparseBlockLinearSystem{T}; symmetric::Bool=false) -> (SparseMatrixCSC{T, Int}, Vector{Int})
    sparse_with_map(sparse_sys::SparseBlockLinearSystem{T}, sparse_idx::SparseMatrixCSC{Int, Int}) -> (SparseMatrixCSC{T, Int}, Vector{Int})

Precomputes a standard `SparseMatrixCSC{T, Int}` alongside a `scatter_map::Vector{Int}`.
The `scatter_map` maps entries from `sparse_sys.data` to the CSC nonzeros vector.

Subsequent updates can be performed in-place with **zero allocations** via
`copyto!(csc_mat, sparse_sys, scatter_map)`.

# Example
```jldoctest
julia> using BlockLinearSystems, SparseArrays

julia> sys = SparseBlockLinearSystem{Float64}(sparse([true true; true true]), [2, 2]);

julia> block(sys, 1, 1) .= 1.0;

julia> csc_mat, scatter_map = sparse_with_map(sys; symmetric=false);

julia> block(sys, 1, 1) .= 2.0;   # in the solver loop

julia> copyto!(csc_mat, sys, scatter_map);   # in place, without allocating

julia> csc_mat[1, 1]
2.0
```
"""
sparse_with_map

"""
    BlockRLE(runlengths, blocksizes)
    BlockRLE(runs::Vector{Pair})
    BlockRLE(vec::AbstractVector{<:Integer})

Run-Length Encoded (RLE) block layout metadata capturing contiguous runs of uniform block sizes.
Subtypes `AbstractVector{Int}` (storing the sizes compactly, as `UInt16` by default, but always returning `Int`) for transparent interop with all linear systems while providing
\$O(1)\$ matrix dimension lookup, \$O(1)\$ run indexing, and \$O(\\log R)\$ block-to-coordinate mapping.

Block dimensions must be between 1 and 65535 (for a `BlockRLE{T}` with another storage type, 1 and `typemax(T)`).
Any other size is an `ArgumentError` that names the block, or the run for run-length input; zero-length runs are
skipped without their size being checked.

# Fields
- `runlengths::Vector{Int}`: Number of blocks in each run.
- `blocksizes::Vector{T}`: Block dimension in each run, stored as `T` (`UInt16` by default).
- `cumblocks::Vector{Int}`: 1-based start block index for each run, terminated by `total_blocks + 1`.
- `cummatrixdims::Vector{Int}`: 1-based start scalar matrix coordinate for each run, terminated by `total_dimension + 1`.

# Example
```jldoctest
julia> rle = BlockRLE([100, 50_000], [6, 3]);   # 100 blocks of dimension 6, then 50,000 of dimension 3

julia> length(rle)
50100

julia> sum(rle)   # the matrix dimension
150600
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
    blockrange(first, len) -> AbstractUnitRange

Index range of a block of `len` entries starting at index `first`. A `StaticInt` `len` gives a statically
sized range; any other integer `len` gives a plain `UnitRange{Int}`, so views built with it are type stable
and allocation free even when the size is only known at run time. Integer sizes of any type (including
the `UInt16` sizes stored by `BlockRLE`) are widened to `Int` before use.
"""
blockrange

"""
    SchurBlockJacobian{T, RType}(structure::SchurJacobianStructure)
    SchurBlockJacobian{T, RType}(data::Vector{T}, r::Vector{Float64}, structure::SchurJacobianStructure)

Raw storage of a least-squares problem laid out by `structure` (see [`SchurJacobianStructure`](@ref)):
one dense table of `J` per eliminated block and per free group, in one flat vector `data` of element type `T`,
and the residual in a vector `r` of its own, always `Float64`. It is the Jacobian prototype a caller fills, the analogue of
[`SchurBlockHessian`](@ref) on the Hessian path. `T` may be any floating-point type; `Float64` and
`Float32` are tested.

`RType` is the storage of the reduced system in the retained variables, `J̃` and `S = J̃ᵀJ̃`, that the linear
system built from `J` (DampedLinearSolve's `makejacobiansystem`) solves after eliminating: `Matrix{T}` (dense) or
`SparseMatrixCSC{T, Int}` (sparse); anything else is an `ArgumentError`. It plays the part the type of `R` plays in
`SchurBlockHessian{T, RType}`, and is chosen by whoever builds the prototype (a modelling package, from its own
storage option), so that code receiving only `J` can dispatch on `J isa SchurBlockJacobian{T, Matrix{T}}`. There
is no default. The tables are the same for both.

The data-taking form uses `data` and `r` as the storage (no copy); their lengths must be the structure's, else
`DimensionMismatch`.

Fill it by row block, through [`blockentry`](@ref), [`block`](@ref) and [`residual_block`](@ref). The rows of
different row blocks are disjoint, so threads that fill different row blocks never write the same memory and need
no per-thread copies. The damping rows are not reachable through the views and stay zero. The elimination reads
these tables and never modifies them, so a change of the damping re-factors from them.

**Fields.** `data`, `r` and `structure` may be read for operations on the whole system: zero it, copy it, compare
structures by identity. Where a block lives inside `data` or `r` is not public: address it by row block with
[`blockentry`](@ref), [`block`](@ref) and [`residual_block`](@ref).

`copy(J)` returns a `SchurBlockJacobian` with its own copy of the tables and the residual and the same structure,
and `copyto!(dest, src)` copies both of one into another of the same structure.
[`zero!`](@ref) zeroes every table and the residual; [`copy_structure`](@ref) returns a new, zeroed `SchurBlockJacobian`
of the same type, sharing only the (immutable) structure. Each table is dense in every retained block coupled to its
eliminated block, `(m_e + d_e) × (c_e + d_e)` for `m_e` rows, `d_e` eliminated and `c_e` coupled retained
columns (with `m_e + d_e` residual entries), and the elimination keeps a factored copy of each table with its
residual, `(m_e + d_e) × (c_e + d_e + 1)` in `T`: far more than the Hessian path's `d_e² + d_e c_e` for an
eliminated block seen by many rows.
"""
SchurBlockJacobian

"""
    schur_jacobian_bytes(structure::SchurJacobianStructure, T=Float64; workspace=true) -> Int

Bytes of element data a [`SchurBlockJacobian{T}`](@ref SchurBlockJacobian) (of either reduced storage) for `structure` allocates
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
```jldoctest
julia> using BlockLinearSystems, Static, StaticArrays

julia> J = SchurBlockJacobian{Float64, Matrix{Float64}}(SchurJacobianStructure([6, 3], [3, 3], [[1, 2], [2]], [4, 6]));

julia> u = BlockLinearSystems.coupling_entry(J.structure, 1, 2)   # retained block 2 in eliminated block 1's table
2

julia> BlockLinearSystems.coupling_block(J, u, 3, static(2), static(3)) .= @SMatrix [1.0 2.0 3.0; 4.0 5.0 6.0];   # rows 3:4 of that table

julia> Matrix(BlockLinearSystems.coupling_block(J, u))   # all four data rows
4×3 Matrix{Float64}:
 0.0  0.0  0.0
 0.0  0.0  0.0
 1.0  2.0  3.0
 4.0  5.0  6.0
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
    residual_block(J::SchurBlockJacobian, k[, nrows])

Mutable view of row block `k`'s rows of the residual, in the `Float64` vector `J.r`, for a [`SchurBlockJacobian`](@ref) whose
structure was built from a description. `nrows`, if given, must be the row block's number of rows (checked
under `@boundscheck`); a static `nrows` gives a `SizedVector`, otherwise a plain view. It allocates nothing.
The residual is `r` of `min ‖J x - r‖²`: callers that hold `f(x) ≈ f(x0) + J dx` store `r = -f(x0)`, or store
`f(x0)` and negate the step, as with the Schur systems.
"""
residual_block

"""
    blockentry(structure::SchurJacobianStructure, k, j) -> Int

A handle for layout column block `j` in row block `k` of a structure built from a description, for
[`block`](@ref)`(J, k, entry, nrows, ncols)`. For setup: keep one per (row block, column block) to be written.
What the integer encodes is internal. Throws `ArgumentError` if row block `k` does not touch column block `j`,
`BoundsError` for an unknown row block.

# Example
```jldoctest
julia> using BlockLinearSystems, Static, StaticArrays

julia> jst = SchurJacobianStructure([3, 6], [[1, 2], [2]], [2, 6]; eliminated=[true, false]);   # a point, a camera

julia> J = SchurBlockJacobian{Float64, Matrix{Float64}}(jst);

julia> p = invperm(blockorder(jst));   # the layout number of each of the caller's blocks

julia> block(J, 1, blockentry(jst, 1, p[1]), static(2), static(3)) .= @SMatrix [1.0 2.0 3.0; 4.0 5.0 6.0];

julia> residual_block(J, 1) .= [0.5, -0.5];

julia> Matrix(block(J, 1, blockentry(jst, 1, p[1]), 2, 3))
2×3 Matrix{Float64}:
 1.0  2.0  3.0
 4.0  5.0  6.0
```
"""
blockentry

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
add `λ·s_r` to the diagonal of `J̃ᵀJ̃`. `lambda = 0` is undamped. `lambda` and `scaling_e` must be finite and
non-negative: anything else throws `ArgumentError` before any table is written.

Returns `false` if an eliminated block is numerically singular — `|U_kk| ≤ M·eps(T)·‖column k‖`, with
`M` the table's row count (the rounding left on an exactly dependent column grows with the number of
rows), which catches under-determined (`e_rows[e] < d_e`), rank-deficient and zero blocks without damping
— or its table contains a non-finite value; `ws.failed_block[]` then holds the lowest failing block, as
for the Schur reduction. Free tables are copied, not checked.

Statically sized in the eliminated dimension (unrolled over the columns) up to
`BlockLinearSystems.D_STATIC`, dynamic above, over the runs of the structure's `e_dims` and their `chunks`.
Threaded over the chunks of eliminated blocks when Polyester is loaded and the structure's work is at least its
`min_work`, serial otherwise; tables are independent, so there is no
merge step. The raw tables are untouched, so calling it again with another `lambda` re-factors from them.
It allocates nothing, serial or threaded (on Julia 1.10, a threaded call allocates a few hundred
bytes per chunk; see [Threading](@ref)).
"""
reduce_qr!

"""
    backsubstitute!(x_e, x_r, ws::QRReductionWorkspace)

Recover the eliminated variables from the retained ones after [`reduce_qr!`](@ref): `x_e = U⁻¹ (q - B x_r)`
for each eliminated block, from the top rows of its factored table. Equals the Schur
`x_e = E⁻¹ (b_e - C x_r)` of `H = JᵀJ + λ·Diagonal(s)`. Static up to `D_STATIC` dimensions, dynamic above
(using the workspace scratch), threaded over chunks of eliminated blocks; allocation free (but see [Threading](@ref) for Julia 1.10).
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
…, then each free group's rows. Threaded over tables, allocation free (but see [Threading](@ref) for Julia 1.10).
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
    reduced_jacobian!(Jt::SparseMatrixCSC, rt::AbstractVector, ws::QRReductionWorkspace) -> Jt

Write the reduced problem: `Jt = J̃` (size `(reduced_nrows(structure), nr)`) and `rt = r̃`, in the row order of
[`reduced_mul!`](@ref), for a QR of the reduced problem. Solving that by QR, then [`backsubstitute!`](@ref),
keeps the whole solve at the condition number of `J`; forming `J̃ᵀJ̃` squares it.

A dense `Jt` is overwritten entirely. A `SparseMatrixCSC` `Jt` must have the pattern of
[`build_reduced_jacobian_pattern`](@ref), and only its `nzval` is written: the same matrix as the dense form,
entry for entry, filled in place. Its pattern is checked first (column heights and the first row of each table's
run), and an `ArgumentError` is thrown if it differs. The sparse form allocates nothing and is threaded by retained
column ownership, so no two threads write the same entry (but see [Threading](@ref) for Julia 1.10).

The retained damping is left to the caller, as for every reduced kernel: rows `√(λ s_r)` appended to `J̃`.
"""
reduced_jacobian!

"""
    build_reduced_jacobian_pattern(structure::SchurJacobianStructure, ::Type{T}=Float64) -> SparseMatrixCSC{T, Int}

The sparse `J̃`, of size `(reduced_nrows(structure), nr)`, with every stored value zero, for
[`reduced_jacobian!`](@ref) to fill. Its rows are the reduced rows of every table, in the order of
[`reduced_mul!`](@ref). Each column of retained block `r` stores the reduced rows of every table that couples
to `r`, whether an eliminated block's or a free group's. Build it once per structure; each fill writes only `nzval`.
A table whose bottom rows are numerically zero (`λ = 0`, see [`reduced_nrows`](@ref)) still has its entries stored.
"""
build_reduced_jacobian_pattern

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
`ArgumentError` if the pattern lacks a needed block.
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
    factor_diagonal_blocks!(D::AbstractVector, dims) -> Bool

Factor in place each square block of the flat array `D` by Cholesky, `B = L Lᵀ`, writing `L` into the block's
lower triangle (its strict upper triangle is not read or written). Block `b`, of dimension `dims[b]`, is stored
column-major after the blocks before it, as [`reduced_diagonal_blocks!`](@ref) lays them out. Returns `true` if
every block is positive definite and finite; otherwise `false`, with `D` partly factored. Plain loops, for small
blocks: no BLAS, no allocation.
"""
factor_diagonal_blocks!

"""
    solve_diagonal_blocks!(y, D::AbstractVector, dims, x) -> y

`y = B⁻¹ x` for the block-diagonal matrix whose blocks [`factor_diagonal_blocks!`](@ref) has factored in `D`:
block `b` of `x` and `y` holds entries after those of the blocks before it. `y` may be `x`. Plain loops, no
allocation.
"""
solve_diagonal_blocks!

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
    jtj_quadform(src::Union{SchurBlockJacobian, QRReductionWorkspace}, v) -> Real

`‖J v‖² = vᵀ(JᵀJ)v` for `v = [v_r; v_e]`, from the tables, without forming anything: the analogue of
`dot(v, H, v)` for a [`SchurBlockHessian`](@ref) holding the Hessian `H = JᵀJ`. From the factored tables
the damped Jacobian is used, so the result includes `λ * Σ s_e v_e²` (equal at `λ = 0`). Any element type of `v`;
the result has the promoted type.
"""
jtj_quadform

"""
    dot(x::AbstractVector{<:Real}, H::SchurBlockHessian{<:Real}, y::AbstractVector{<:Real}) -> Real

The bilinear form `xᵀ H y`, with `x` and `y` over `[retained; eliminated]` (length `nr + sum(e_dims)`), in
O(N) operations without materializing `H`. The result has the promoted element type of `H`, `x` and `y`. A
`DimensionMismatch` is thrown if a length is wrong. Real element types only.

`H` is symmetric: one triangle of a dense `R` is read, and the stored blocks of a sparse `R`.

When `x === y` this is the quadratic form `xᵀ H x`, computed by a faster kernel that reads each term once and
doubles the off-diagonal and coupling terms. The eliminated part (`E` and the couplings `C`), where nearly all
the work is, is processed per group of eliminated blocks of uniform dimension with that dimension known
statically (3.4x faster than per-block run-time sizes on a bundle-adjustment-like benchmark with 200 retained
and 20000 eliminated blocks). Pass the same vector twice to get it.
"""
LinearAlgebra.dot(::AbstractVector{<:Real}, ::SchurBlockHessian{<:Real}, ::AbstractVector{<:Real})

"""
    retained_block(H::SchurBlockHessian, i, j[, rows, cols])

Mutable view of block `(i, j)` of `R`, `d_i × d_j`. For a dense `R` (a `Matrix`) any block may be
viewed; for a sparse `R` (a [`SparseBlockLinearSystem`](@ref)) only stored blocks (the lower triangle,
`i >= j`), else `ArgumentError`. `rows` and `cols` may be `Int`, `StaticInt` or `Val` and must equal the
block's size (`DimensionMismatch` otherwise); static sizes give a `SizedMatrix`, as for [`block`](@ref).

With [`eliminated_block`](@ref)`(H, e)` and [`coupling_block`](@ref)`(H, c)` this lets a caller assemble a
[`SchurBlockHessian`](@ref) without offset arithmetic. All forms are type stable and allocation free.
"""
retained_block

"""
    eliminated_block(H::SchurBlockHessian, e[, rows, cols])

Mutable view of the eliminated block `E_e` (`d_e × d_e`, column-major in `H.E_data`). Sizes and errors
as for [`retained_block`](@ref).
"""
eliminated_block(::SchurBlockHessian, ::Integer)

"""
    coupling_block(H::SchurBlockHessian, c[, rows, cols])

Mutable view of the coupling block `C_er` (`d_e × d_r`, column-major in `H.C_data`) of coupling entry `c`
(an index into `H.structure.entry_retained`; [`coupling_entry`](@ref)`(H.structure, e, r)` finds it).
Sizes and errors as for [`retained_block`](@ref).
"""
coupling_block(::SchurBlockHessian, ::Integer)

"""
    coupling_entry(structure::SchurStructure, e, r) -> Int

Index of the coupling entry of retained block `r` in eliminated block `e` (binary search). Throws
`ArgumentError` if `r` is not coupled to `e`, `BoundsError` for an unknown eliminated block.
"""
coupling_entry(::SchurStructure, ::Integer, ::Integer)

"""
    threadargs(x) -> ThreadArgs

`x` as an isbits value, for passing into a `Polyester.@batch` loop without allocating; rebuild it inside
the loop with [`fromthreadargs`](@ref). Polyester hands the threads an isbits value as it is but boxes any
other captured value, one allocation per launch. `threadargs` walks `x`'s immutable fields (a tuple, a
struct, a closure) into a flat tuple of leaves: each isbits field stays as it is, and each object held by
reference (a mutable object such as an `Array`, or a field of abstract type) becomes its address.

The addresses do not keep their objects alive: keep `x` alive with `GC.@preserve x` for as long as the
loop runs. The methods come from the Polyester package extension, so `threadargs` has none unless
Polyester is loaded.

```jldoctest
julia> using BlockLinearSystems, Polyester

julia> function scale!(ys::Vector{Vector{Float64}}, a)
           args = threadargs((ys, a))
           GC.@preserve ys begin
               @batch for i in eachindex(ys)
                   ys_, a_ = fromthreadargs(args)
                   ys_[i] .*= a_
               end
           end
           return ys
       end;

julia> scale!([[1.0, 2.0], [3.0]], 2.0)
2-element Vector{Vector{Float64}}:
 [2.0, 4.0]
 [6.0]
```

This is the mechanism every threaded kernel of this package uses. It is here until it can be offered to
Polyester; moving it there will remove `threadargs` and `fromthreadargs` from this package's API, in a
minor version.
"""
threadargs

"""
    fromthreadargs(args) -> x

Rebuild, inside a `Polyester.@batch` loop, the value `x` that [`threadargs`](@ref) turned into `args`;
the result has the type of `x`. Bind the result to a local variable before using it (calling it directly
as `fromthreadargs(args)(i)` made Polyester fail to compile the loop on Julia 1.13). Needs Polyester
loaded, as `threadargs` does, and may move to Polyester with it.
"""
fromthreadargs
