# Sized views into the storage of a SchurBlockHessian, so that callers assemble R, E and C without offset
# arithmetic: the Hessian-path counterparts of the views into a SchurBlockJacobian's tables.

# A d1 x d2 column-major block stored contiguously from data[off]
@inline flat_block(data::Vector, off::Int, ::StaticInt{R}, ::StaticInt{C}) where {R, C} =
    SizedMatrix{R, C}(view(data, SR(0, R * C - 1) .+ off))
@inline flat_block(data::Vector, off::Int, r::Integer, c::Integer) =
    reshape(view(data, blockrange(off, Int(r) * Int(c))), Int(r), Int(c))
@inline flat_block(data::Vector, off::Int, r::StaticInt{R}, c::Integer) where R =
    HybridArray{Tuple{R, StaticArrays.Dynamic()}}(flat_block(data, off, Int(R), c))
@inline flat_block(data::Vector, off::Int, r::Integer, c::StaticInt{C}) where C =
    HybridArray{Tuple{StaticArrays.Dynamic(), C}}(flat_block(data, off, r, Int(C)))

function check_block_size(what::String, index, expected_rows::Integer, expected_cols::Integer, rows::Integer, cols::Integer)
    (rows == expected_rows && cols == expected_cols) ||
        throw(DimensionMismatch("$what $index has size ($expected_rows, $expected_cols), but ($rows, $cols) was requested"))
    return nothing
end

# eliminated_block(H, e[, rows, cols]): view of E_e (d_e x d_e)
@inline function eliminated_block(H::SchurBlockHessian, e::Integer, rows, cols)
    st = H.structure
    @boundscheck (1 <= e <= st.neblocks) || throw(BoundsError(H, e))
    r_, c_ = as_size(rows), as_size(cols)
    de = @inbounds Int(st.e_dims[e])
    @boundscheck check_block_size("Eliminated block", e, de, de, dynamic(r_), dynamic(c_))
    return flat_block(H.E_data, @inbounds(st.eoffsets[e]), r_, c_)
end
@inline function eliminated_block(H::SchurBlockHessian, e::Integer)
    st = H.structure
    @boundscheck (1 <= e <= st.neblocks) || throw(BoundsError(H, e))
    de = @inbounds Int(st.e_dims[e])
    return eliminated_block(H, e, de, de)
end

# coupling_block(H, c[, rows, cols]): view of C_er (d_e x d_r) for coupling entry c
@inline function coupling_block(H::SchurBlockHessian, c::Integer, rows, cols)
    st = H.structure
    @boundscheck (1 <= c <= length(st.entry_retained)) || throw(BoundsError(H, c))
    r_, c_ = as_size(rows), as_size(cols)
    e = coupling_eliminated(st, Int(c))
    de = @inbounds Int(st.e_dims[e])
    dr = @inbounds Int(st.r_dims[st.entry_retained[c]])
    @boundscheck check_block_size("Coupling entry", c, de, dr, dynamic(r_), dynamic(c_))
    return flat_block(H.C_data, @inbounds(st.coffsets[c]), r_, c_)
end
@inline function coupling_block(H::SchurBlockHessian, c::Integer)
    st = H.structure
    @boundscheck (1 <= c <= length(st.entry_retained)) || throw(BoundsError(H, c))
    e = coupling_eliminated(st, Int(c))
    return coupling_block(H, c, @inbounds(Int(st.e_dims[e])), @inbounds(Int(st.r_dims[st.entry_retained[c]])))
end

# Eliminated block of coupling entry c (binary search in eliminated_ptr)
@inline coupling_eliminated(st::SchurStructure, c::Int) = searchsortedlast(st.eliminated_ptr, c)

# coupling_entry(structure::SchurStructure, e, r): coupling entry of retained block r in eliminated block e
function coupling_entry(st::SchurStructure, e::Integer, r::Integer)
    1 <= e <= st.neblocks || throw(BoundsError(st, e))
    rng = st.eliminated_ptr[e]:st.eliminated_ptr[e + 1] - 1
    k = searchsortedfirst(view(st.entry_retained, rng), r)
    (k <= length(rng) && st.entry_retained[rng[k]] == r) || throw(ArgumentError("Retained block $r is not coupled to eliminated block $e"))
    return rng[k]
end

# retained_block(H, i, j[, rows, cols]): view of block (i, j) of R
@inline function retained_block(H::SchurBlockHessian{T, <:Matrix}, i::Integer, j::Integer, rows, cols) where T
    st = H.structure
    @boundscheck (1 <= i <= st.nrblocks && 1 <= j <= st.nrblocks) || throw(BoundsError(H, (i, j)))
    r_, c_ = as_size(rows), as_size(cols)
    @boundscheck check_block_size("Retained block", (i, j), @inbounds(Int(st.r_dims[i])), @inbounds(Int(st.r_dims[j])), dynamic(r_), dynamic(c_))
    return @inbounds sized_view(H.R, st.boffsets_r[i], r_, st.boffsets_r[j], c_)
end
@inline function retained_block(H::SchurBlockHessian{T, <:SparseBlockLinearSystem}, i::Integer, j::Integer, rows, cols) where T
    @boundscheck validblock(H.R, i, j) || throw(ArgumentError("Block ($i, $j) of R is not stored"))
    return block(H.R, i, j, as_size(rows), as_size(cols))
end
@inline function retained_block(H::SchurBlockHessian, i::Integer, j::Integer)
    st = H.structure
    @boundscheck (1 <= i <= st.nrblocks && 1 <= j <= st.nrblocks) || throw(BoundsError(H, (i, j)))
    return retained_block(H, i, j, @inbounds(Int(st.r_dims[i])), @inbounds(Int(st.r_dims[j])))
end

# ------------------------------------------------------------------------------------------------------------------
# Filling by block: addblock!
# ------------------------------------------------------------------------------------------------------------------

# The size of an index range: static for a statically sized one (`blockrange` with a `StaticInt` length), so that the
# loops and views it sizes are static
@inline range_size(::StaticArrays.StaticVector{N}) where N = StaticInt{N}()
@inline range_size(r::AbstractVector) = length(r)

# Coupling entry of retained block r in eliminated block e (0 if they are not coupled)
@inline function find_coupling(st::SchurStructure, e::Int, r::Int)
    lo = @inbounds st.eliminated_ptr[e]
    hi = @inbounds st.eliminated_ptr[e + 1] - 1
    c = searchsortedfirst(@inbounds(view(st.entry_retained, lo:hi)), r) + lo - 1
    return (c <= hi && @inbounds(st.entry_retained[c]) == r) ? c : 0
end

"""
    addblock!(H::SchurBlockHessian, i, j, B, rows, cols) -> H

Add a contribution to the block of layout column blocks `(i, j)` of `H` (`d_i × d_j`): `B[rows, cols]` is that
block and `B[cols, rows]` the block `(j, i)`, so `B` holds a symmetric contribution, of which the system reads
the one block it stores, in whichever orientation it stores the pair (the retained part's stored triangle, an
eliminated block, or a coupling block). Only that block is read, so the result is the same whichever order the
call names the pair in, even for a `B` that is symmetric only to rounding. `rows` and `cols` are index vectors
into `B`, static (from [`blockrange`](@ref) with a `StaticInt` length) or run-time; a static one gives a
statically sized loop.

- Add each unordered pair `{i, j}` once per contribution, in either order; `i == j` adds the full block.
- A pair the system has no storage for is an error under `@boundscheck` (`ArgumentError`): two eliminated
  blocks; a retained and an eliminated block that share no row block; two retained blocks that share no row
  block or eliminated block, when the retained part is sparse (a dense one stores every pair). An unknown block
  is a `BoundsError`, a wrong size a `DimensionMismatch`.
- It allocates nothing.
- Calls for pairs that involve different eliminated blocks write disjoint storage; calls for two retained
  blocks do not. That is what a caller needs to thread its fill: give each thread whole eliminated blocks, and
  each thread its own retained part ([`SchurBlockHessian`](@ref)`(R, like)`).

# Example
```jldoctest
julia> using BlockLinearSystems

julia> st = SchurStructure([3, 6], [[1, 2]]; eliminated=[true, false]);   # a point and a camera

julia> H = SchurBlockHessian{Float64, Matrix{Float64}}(st);

julia> p = invperm(blockorder(st));   # the layout number of each of the caller's blocks

julia> B = ones(9, 9);   # the contribution of the one row block, over [point; camera]

julia> addblock!(H, p[1], p[1], B, 1:3, 1:3); addblock!(H, p[2], p[2], B, 4:9, 4:9); addblock!(H, p[2], p[1], B, 4:9, 1:3);

julia> sum(H.E_data), sum(H.C_data), sum(H.R)
(9.0, 18.0, 36.0)
```
"""
@inline function addblock!(H::SchurBlockHessian, i::Integer, j::Integer, B::AbstractMatrix, rows, cols)
    st = H.structure
    nr = st.nrblocks
    i, j = Int(i), Int(j)
    @boundscheck check_addblock(H, i, j, B, rows, cols)
    if i == j
        if i <= nr
            @inbounds add_retained_diagonal!(H.R, st, i, B, rows)
        else
            @inbounds add_eliminated!(H.E_data, st, i - nr, B, rows)
        end
    elseif i <= nr && j <= nr
        @inbounds add_retained_pair!(H.R, st, i, j, B, rows, cols)
    elseif i <= nr
        @inbounds add_coupling!(H.C_data, st, j - nr, i, B, rows, cols)
    else
        @inbounds add_coupling!(H.C_data, st, i - nr, j, B, cols, rows)
    end
    return H
end

@inline function check_addblock(H::SchurBlockHessian, i::Int, j::Int, B, rows, cols)
    st = H.structure
    n = st.nrblocks + st.neblocks
    (1 <= i <= n && 1 <= j <= n) || throw_block_bounds(H, i, j)
    di = i <= st.nrblocks ? Int(st.r_dims[i]) : Int(st.e_dims[i - st.nrblocks])
    dj = j <= st.nrblocks ? Int(st.r_dims[j]) : Int(st.e_dims[j - st.nrblocks])
    (length(rows) == di && length(cols) == dj) || throw_block_size(i, j, di, dj, length(rows), length(cols))
    checkbounds(B, rows, cols)
    checkbounds(B, cols, rows)
    if i != j
        if i > st.nrblocks && j > st.nrblocks
            throw_no_storage(i, j, "are both eliminated")
        elseif i <= st.nrblocks && j <= st.nrblocks
            H.R isa SparseBlockLinearSystem && !validblock(H.R, max(i, j), min(i, j)) && throw_no_storage(i, j, "share no row block")
        else
            e, r = max(i, j) - st.nrblocks, min(i, j)
            find_coupling(st, e, r) == 0 && throw_no_storage(i, j, "share no row block")
        end
    end
    return nothing
end

@noinline throw_block_bounds(H, i, j) = throw(BoundsError(H, (i, j)))
@noinline throw_block_size(i, j, di, dj, m, n) =
    throw(DimensionMismatch("Block ($i, $j) has size ($di, $dj), but ($m, $n) was given"))
@noinline throw_no_storage(i, j, why) = throw(ArgumentError("Blocks $i and $j $why: the system has no storage for the pair"))

# The diagonal block of a retained block: all of B[rows, rows]
@inline function add_retained_diagonal!(R::Matrix, st::SchurStructure, i::Int, B, rows)
    ro = @inbounds st.boffsets_r[i]
    n = length(rows)
    @inbounds for col in 1:n, row in 1:n
        R[ro + row - 1, ro + col - 1] += B[rows[row], rows[col]]
    end
    return nothing
end
@inline function add_retained_diagonal!(R::SparseBlockLinearSystem, st::SchurStructure, i::Int, B, rows)
    n = range_size(rows)
    @inbounds block(R, i, i, n, n) .+= view(B, rows, rows)
    return nothing
end

# A pair of retained blocks. The dense retained part stores the upper triangle (block row <= block column) and the
# sparse one the lower: each reads the block of B in the orientation it stores, B[rows, cols] for (i, j) or B[cols, rows]
# for (j, i), and copies it without transposing.
@inline function add_retained_pair!(R::Matrix, st::SchurStructure, i::Int, j::Int, B, rows, cols)
    a, ra, b, rb = i < j ? (i, rows, j, cols) : (j, cols, i, rows)
    ro_a = @inbounds st.boffsets_r[a]
    ro_b = @inbounds st.boffsets_r[b]
    na, nb = length(ra), length(rb)
    @inbounds for col in 1:nb, row in 1:na
        R[ro_a + row - 1, ro_b + col - 1] += B[ra[row], rb[col]]
    end
    return nothing
end
@inline function add_retained_pair!(R::SparseBlockLinearSystem, st::SchurStructure, i::Int, j::Int, B, rows, cols)
    if i > j
        @inbounds block(R, i, j, range_size(rows), range_size(cols)) .+= view(B, rows, cols)
    else
        @inbounds block(R, j, i, range_size(cols), range_size(rows)) .+= view(B, cols, rows)
    end
    return nothing
end

# An eliminated block: all of B[rows, rows], column-major in E_data
@inline function add_eliminated!(E_data::Vector, st::SchurStructure, e::Int, B, rows)
    idx = @inbounds st.eoffsets[e]
    n = length(rows)
    @inbounds for col in 1:n, row in 1:n
        E_data[idx] += B[rows[row], rows[col]]
        idx += 1
    end
    return nothing
end

# The coupling block of eliminated block e and retained block r, from B[rr, re] (retained rows, eliminated columns),
# stored transposed (d_e x d_r, column-major) in C_data
@inline function add_coupling!(C_data::Vector, st::SchurStructure, e::Int, r::Int, B, rr, re)
    c = find_coupling(st, e, r)
    co = @inbounds st.coffsets[c]
    dr, de = length(rr), length(re)
    @inbounds for col in 1:de, row in 1:dr
        C_data[co + (row - 1) * de + col - 1] += B[rr[row], re[col]]
    end
    return nothing
end
