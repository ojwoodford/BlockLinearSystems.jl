using Static, StaticArrays, HybridArrays

# Sized views into the storage of a SchurBlockMatrix, so that callers assemble R, E and C without offset
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
@inline function eliminated_block(H::SchurBlockMatrix, e::Integer, rows, cols)
    st = H.structure
    @boundscheck (1 <= e <= st.neblocks) || throw(BoundsError(H, e))
    r_, c_ = as_size(rows), as_size(cols)
    de = @inbounds Int(st.e_dims[e])
    @boundscheck check_block_size("Eliminated block", e, de, de, dynamic(r_), dynamic(c_))
    return flat_block(H.E_data, @inbounds(st.eoffsets[e]), r_, c_)
end
@inline function eliminated_block(H::SchurBlockMatrix, e::Integer)
    st = H.structure
    @boundscheck (1 <= e <= st.neblocks) || throw(BoundsError(H, e))
    de = @inbounds Int(st.e_dims[e])
    return eliminated_block(H, e, de, de)
end

# coupling_block(H, c[, rows, cols]): view of C_er (d_e x d_r) for coupling entry c
@inline function coupling_block(H::SchurBlockMatrix, c::Integer, rows, cols)
    st = H.structure
    @boundscheck (1 <= c <= length(st.entry_retained)) || throw(BoundsError(H, c))
    r_, c_ = as_size(rows), as_size(cols)
    e = coupling_eliminated(st, Int(c))
    de = @inbounds Int(st.e_dims[e])
    dr = @inbounds Int(st.r_dims[st.entry_retained[c]])
    @boundscheck check_block_size("Coupling entry", c, de, dr, dynamic(r_), dynamic(c_))
    return flat_block(H.C_data, @inbounds(st.coffsets[c]), r_, c_)
end
@inline function coupling_block(H::SchurBlockMatrix, c::Integer)
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
@inline function retained_block(H::SchurBlockMatrix{T, <:Matrix}, i::Integer, j::Integer, rows, cols) where T
    st = H.structure
    @boundscheck (1 <= i <= st.nrblocks && 1 <= j <= st.nrblocks) || throw(BoundsError(H, (i, j)))
    r_, c_ = as_size(rows), as_size(cols)
    @boundscheck check_block_size("Retained block", (i, j), @inbounds(Int(st.r_dims[i])), @inbounds(Int(st.r_dims[j])), dynamic(r_), dynamic(c_))
    return @inbounds sized_view(H.R, st.boffsets_r[i], r_, st.boffsets_r[j], c_)
end
@inline function retained_block(H::SchurBlockMatrix{T, <:SparseBlockLinearSystem}, i::Integer, j::Integer, rows, cols) where T
    @boundscheck validblock(H.R, i, j) || throw(ArgumentError("Block ($i, $j) of R is not stored"))
    return block(H.R, i, j, as_size(rows), as_size(cols))
end
@inline function retained_block(H::SchurBlockMatrix, i::Integer, j::Integer)
    st = H.structure
    @boundscheck (1 <= i <= st.nrblocks && 1 <= j <= st.nrblocks) || throw(BoundsError(H, (i, j)))
    return retained_block(H, i, j, @inbounds(Int(st.r_dims[i])), @inbounds(Int(st.r_dims[j])))
end
