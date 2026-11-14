using Static, StaticArrays, SparseArrays, LinearAlgebra

# QR elimination: the Jacobian-path counterpart of the Schur complement.
#
# The linear least-squares problem min ||J x - r||^2 over x = [x_r; x_e] (retained first) is stored as one dense
# table per eliminated block e, holding the rows of J that involve e:
#
#     T_e = [ J_r | J_e | r ]        (m_e + d_e) x (c_e + d_e + 1), column-major
#
# with the retained columns of every retained block coupled to e (in coupling-entry order), then the d_e
# eliminated columns, then the residual. The last d_e rows are damping rows, zero in the raw storage and written
# by the reduction. Rows that involve no eliminated block live in "free" tables [ J_r | r ], one per free group.
#
# The reduction applies a Householder QR of the (damped) eliminated columns to the whole table. The top d_e rows
# become [B | U | q] (U upper triangular), giving x_e = U^-1 (q - B x_r); the bottom m_e rows [J~_e | 0 | r~_e]
# do not involve x_e. The bottom rows of every table, then the free rows, form the reduced problem
# min ||J~ x_r - r~||^2, with J~'J~ the Schur complement of J'J (without the retained damping).

# SchurJacobianStructure(structure::SchurStructure, e_rows; free_rows=Int[], free_retained=Vector{Int}[], nchunks)
# SchurJacobianStructure(r_dims, e_dims, couplings, e_rows; free_rows, free_retained, nchunks=1)
# Layout of a tall (Jacobian) least-squares problem partitioned like `structure`.
struct SchurJacobianStructure
    schur::SchurStructure            # Partition, block offsets, couplings, e_groups and chunks
    e_rows::Vector{Int}              # Number of residual rows m_e of each eliminated block's table
    free_rows::Vector{Int}           # Number of rows of each free group
    ntables::Int                     # neblocks + number of free groups; table t <= neblocks is eliminated block t
    table_nrows::Vector{Int}         # Rows of each table (m_e + d_e for eliminated blocks, the group's rows for free ones)
    table_ncols::Vector{Int}         # Columns of each table (retained columns, eliminated columns, residual)
    table_top::Vector{Int}           # Rows of each table that do not belong to the reduced problem (d_e, or 0)
    table_offsets::Vector{Int}       # Start of each table in the flat data, length ntables + 1
    table_ptr::Vector{Int}           # Table-major CSR pointer into the entries, length ntables + 1
    entry_retained::Vector{Int}      # Retained block of each entry: the coupling entries of `schur`, then the free entries
    entry_col::Vector{Int}           # First column of each entry's retained block inside its table
    entry_table::Vector{Int}         # Table of each entry
    retained_ptr::Vector{Int}        # Retained-block-major CSR pointer into retained_entry
    retained_entry::Vector{Int}      # Entries of each retained block, ascending
    reduced_offsets::Vector{Int}     # First row of each table's rows in the reduced problem, length ntables + 1
    doffsets::Vector{Int}            # Start of each retained block's diagonal block in a flat d_r x d_r array, length nrblocks + 1
    col_chunks::Vector{UnitRange{Int}} # Disjoint ranges of retained blocks, for threading by column ownership
end

function SchurJacobianStructure(
    structure::SchurStructure, e_rows::AbstractVector{<:Integer};
    free_rows::AbstractVector{<:Integer}=Int[],
    free_retained::AbstractVector{<:AbstractVector{<:Integer}}=Vector{Int}[],
    nchunks::Int=maximum(length, structure.chunks; init=1)
)
    st = structure
    ne = st.neblocks
    nrb = st.nrblocks
    nfree = length(free_rows)
    length(e_rows) == ne || throw(DimensionMismatch("Number of eliminated row counts ($(length(e_rows))) must match the number of eliminated blocks ($ne)"))
    length(free_retained) == nfree || throw(DimensionMismatch("Number of free groups' retained lists ($(length(free_retained))) must match the number of free row counts ($nfree)"))
    nchunks >= 1 || throw(ArgumentError("nchunks must be at least 1, got $nchunks"))
    for (e, m) in enumerate(e_rows)
        m >= 1 || throw(ArgumentError("Eliminated block $e has $m rows; every eliminated block needs at least 1 row"))
    end
    for (g, m) in enumerate(free_rows)
        m >= 1 || throw(ArgumentError("Free group $g has $m rows; every free group needs at least 1 row"))
    end
    seen = zeros(Int, nrb)
    for (g, rs) in enumerate(free_retained)
        for r in rs
            1 <= r <= nrb || throw(ArgumentError("Free group $g references retained block $r, but there are only $nrb retained blocks"))
            seen[r] == g && throw(ArgumentError("Retained block $r is listed more than once in free group $g"))
            seen[r] = g
        end
    end

    ntables = ne + nfree
    ncoup = length(st.entry_retained)
    sorted_free = [sort(collect(Int, rs)) for rs in free_retained]
    nentries = ncoup + sum(length, sorted_free; init=0)

    table_nrows = Vector{Int}(undef, ntables)
    table_ncols = Vector{Int}(undef, ntables)
    table_top = Vector{Int}(undef, ntables)
    table_ptr = Vector{Int}(undef, ntables + 1)
    entry_retained = Vector{Int}(undef, nentries)
    entry_col = Vector{Int}(undef, nentries)
    entry_table = Vector{Int}(undef, nentries)

    u = 1
    table_ptr[1] = 1
    for t in 1:ntables
        rs = t <= ne ? view(st.entry_retained, st.eliminated_ptr[t]:st.eliminated_ptr[t + 1] - 1) : sorted_free[t - ne]
        col = 1
        for r in rs
            entry_retained[u] = r
            entry_col[u] = col
            entry_table[u] = t
            col += Int(st.r_dims[r])
            u += 1
        end
        table_ptr[t + 1] = u
        de = t <= ne ? Int(st.e_dims[t]) : 0
        table_top[t] = de
        table_nrows[t] = t <= ne ? Int(e_rows[t]) + de : Int(free_rows[t - ne])
        table_ncols[t] = col - 1 + de + 1
    end

    table_offsets = Vector{Int}(undef, ntables + 1)
    reduced_offsets = Vector{Int}(undef, ntables + 1)
    table_offsets[1] = 1
    reduced_offsets[1] = 1
    for t in 1:ntables
        table_offsets[t + 1] = table_offsets[t] + table_nrows[t] * table_ncols[t]
        reduced_offsets[t + 1] = reduced_offsets[t] + table_nrows[t] - table_top[t]
    end

    # Retained-block-major index of all entries (ascending entry index within each retained block)
    counts = zeros(Int, nrb + 1)
    for u in 1:nentries
        counts[entry_retained[u] + 1] += 1
    end
    retained_ptr = cumsum(vcat(1, counts[2:end]))
    retained_entry = Vector{Int}(undef, nentries)
    fill_pos = copy(retained_ptr)
    for u in 1:nentries
        r = entry_retained[u]
        retained_entry[fill_pos[r]] = u
        fill_pos[r] += 1
    end

    doffsets = Vector{Int}(undef, nrb + 1)
    doffsets[1] = 1
    for r in 1:nrb
        doffsets[r + 1] = doffsets[r] + Int(st.r_dims[r])^2
    end

    return SchurJacobianStructure(
        st, collect(Int, e_rows), collect(Int, free_rows), ntables,
        table_nrows, table_ncols, table_top, table_offsets, table_ptr,
        entry_retained, entry_col, entry_table, retained_ptr, retained_entry,
        reduced_offsets, doffsets, partition_columns(retained_ptr, nchunks)
    )
end

function SchurJacobianStructure(
    r_dims::AbstractVector{<:Integer}, e_dims::AbstractVector{<:Integer},
    couplings::AbstractVector{<:AbstractVector{<:Integer}}, e_rows::AbstractVector{<:Integer};
    nchunks::Int=1, kwargs...
)
    return SchurJacobianStructure(SchurStructure(r_dims, e_dims, couplings; nchunks), e_rows; nchunks, kwargs...)
end

# Total number of stored table entries, the work measure for the threading decision
@inline table_length(jst::SchurJacobianStructure) = jst.table_offsets[end] - 1
@inline qr_threaded(jst::SchurJacobianStructure, nitems::Integer) = use_threads(nitems, table_length(jst))

# coupling_entry(structure::SchurJacobianStructure, t, r) -> Int
# The entry of retained block `r` in table `t`.
function coupling_entry(jst::SchurJacobianStructure, t::Integer, r::Integer)
    1 <= t <= jst.ntables || throw(BoundsError(jst, t))
    rng = jst.table_ptr[t]:jst.table_ptr[t + 1] - 1
    k = searchsortedfirst(view(jst.entry_retained, rng), r)
    (k <= length(rng) && jst.entry_retained[rng[k]] == r) || throw(ArgumentError("Retained block $r is not coupled to table $t"))
    return rng[k]
end

# SchurBlockJacobian{T}(structure::SchurJacobianStructure)
# Raw table storage of a tall least-squares problem.
struct SchurBlockJacobian{T}
    data::Vector{T}                  # Every table, column-major, at structure.table_offsets
    structure::SchurJacobianStructure
    function SchurBlockJacobian{T}(data::Vector{T}, jst::SchurJacobianStructure) where T
        length(data) == table_length(jst) || throw(DimensionMismatch("Data has length $(length(data)), the structure needs $(table_length(jst))"))
        return new{T}(data, jst)
    end
end

SchurBlockJacobian{T}(jst::SchurJacobianStructure) where T = SchurBlockJacobian{T}(zeros(T, table_length(jst)), jst)
SchurBlockJacobian(data::Vector{T}, jst::SchurJacobianStructure) where T = SchurBlockJacobian{T}(data, jst)

zero!(J::SchurBlockJacobian) = (fill!(J.data, zero(eltype(J.data))); J)

# copy_structure(J::SchurBlockJacobian)
# A new `SchurBlockJacobian` with the same (shared, immutable) structure and freshly allocated, zeroed tables.
copy_structure(J::SchurBlockJacobian{T}) where T = SchurBlockJacobian{T}(zeros(T, length(J.data)), J.structure)

# ------------------------------------------------------------------------------------------------------------------
# Assembly views
# ------------------------------------------------------------------------------------------------------------------

@inline table_matrix(data::Vector, jst::SchurJacobianStructure, t::Int) =
    @inbounds reshape(view(data, jst.table_offsets[t]:jst.table_offsets[t + 1] - 1), jst.table_nrows[t], jst.table_ncols[t])

@inline sized_view(A, row, ::StaticInt{R}, col, ::StaticInt{C}) where {R, C} =
    SizedMatrix{R, C}(view(A, SR(0, R - 1) .+ row, SR(0, C - 1) .+ col))
@inline sized_view(A, row, nrows::Integer, col, ncols::Integer) = view(A, blockrange(row, nrows), blockrange(col, ncols))
@inline sized_view(A, row, ::StaticInt{R}, col, ncols::Integer) where R =
    HybridArray{Tuple{R, StaticArrays.Dynamic()}}(view(A, blockrange(row, Int(R)), blockrange(col, ncols)))
@inline sized_view(A, row, nrows::Integer, col, ::StaticInt{C}) where C =
    HybridArray{Tuple{StaticArrays.Dynamic(), C}}(view(A, blockrange(row, nrows), blockrange(col, Int(C))))
@inline sized_vview(A, row, ::StaticInt{R}, col) where R = SizedVector{R}(view(A, SR(0, R - 1) .+ row, col))
@inline sized_vview(A, row, nrows::Integer, col) = view(A, blockrange(row, nrows), col)

@inline as_size(n::Integer) = n
@inline as_size(n::StaticInt) = n
@inline as_size(::Val{N}) where N = StaticInt{N}()

# Number of data rows (excluding damping rows) of table t
@inline data_rows(jst::SchurJacobianStructure, t::Int) = @inbounds jst.table_nrows[t] - (t <= jst.schur.neblocks ? jst.table_top[t] : 0)

function check_rows(J, t::Int, row::Integer, nrows::Integer)
    m = data_rows(J.structure, t)
    (1 <= row && row + nrows - 1 <= m && nrows >= 0) || throw(BoundsError(J, (t, row:row + nrows - 1)))
    return nothing
end

function check_cols(what::String, index::Integer, expected::Integer, ncols::Integer)
    ncols == expected || throw(DimensionMismatch("$what $index has $expected columns, but $ncols were requested"))
    return nothing
end

# coupling_block(J::SchurBlockJacobian, u[, row, nrows[, ncols]])
# View of the rows `row:row+nrows-1` of entry `u`'s retained block in its table.
@inline function coupling_block(J::SchurBlockJacobian, u::Integer, row::Integer, nrows, ncols)
    jst = J.structure
    @boundscheck (1 <= u <= length(jst.entry_retained)) || throw(BoundsError(J, u))
    t = @inbounds jst.entry_table[u]
    dr = @inbounds Int(jst.schur.r_dims[jst.entry_retained[u]])
    nr_, nc_ = as_size(nrows), as_size(ncols)
    @boundscheck (check_rows(J, t, row, dynamic(nr_)); check_cols("Entry", u, dr, dynamic(nc_)))
    return sized_view(table_matrix(J.data, jst, t), Int(row), nr_, @inbounds(jst.entry_col[u]), nc_)
end
@inline function coupling_block(J::SchurBlockJacobian, u::Integer, row::Integer, nrows)
    jst = J.structure
    @boundscheck (1 <= u <= length(jst.entry_retained)) || throw(BoundsError(J, u))
    return coupling_block(J, u, row, nrows, @inbounds Int(jst.schur.r_dims[jst.entry_retained[u]]))
end
@inline function coupling_block(J::SchurBlockJacobian, u::Integer)
    jst = J.structure
    @boundscheck (1 <= u <= length(jst.entry_retained)) || throw(BoundsError(J, u))
    return coupling_block(J, u, 1, data_rows(jst, @inbounds jst.entry_table[u]))
end

# eliminated_block(J::SchurBlockJacobian, e[, row, nrows[, ncols]])
# View of the rows `row:row+nrows-1` of the eliminated columns of table `e`.
@inline function eliminated_block(J::SchurBlockJacobian, e::Integer, row::Integer, nrows, ncols)
    jst = J.structure
    @boundscheck (1 <= e <= jst.schur.neblocks) || throw(BoundsError(J, e))
    de = @inbounds Int(jst.schur.e_dims[e])
    nr_, nc_ = as_size(nrows), as_size(ncols)
    @boundscheck (check_rows(J, Int(e), row, dynamic(nr_)); check_cols("Eliminated block", e, de, dynamic(nc_)))
    return sized_view(table_matrix(J.data, jst, Int(e)), Int(row), nr_, @inbounds(jst.table_ncols[e]) - de, nc_)
end
@inline function eliminated_block(J::SchurBlockJacobian, e::Integer, row::Integer, nrows)
    jst = J.structure
    @boundscheck (1 <= e <= jst.schur.neblocks) || throw(BoundsError(J, e))
    return eliminated_block(J, e, row, nrows, @inbounds Int(jst.schur.e_dims[e]))
end
@inline function eliminated_block(J::SchurBlockJacobian, e::Integer)
    jst = J.structure
    @boundscheck (1 <= e <= jst.schur.neblocks) || throw(BoundsError(J, e))
    return eliminated_block(J, e, 1, data_rows(jst, Int(e)))
end

# residual_block(J::SchurBlockJacobian, t[, row, nrows])
# View of the rows `row:row+nrows-1` of the residual column of table `t`.
@inline function residual_block(J::SchurBlockJacobian, t::Integer, row::Integer, nrows)
    jst = J.structure
    @boundscheck (1 <= t <= jst.ntables) || throw(BoundsError(J, t))
    nr_ = as_size(nrows)
    @boundscheck check_rows(J, Int(t), row, dynamic(nr_))
    return sized_vview(table_matrix(J.data, jst, Int(t)), Int(row), nr_, @inbounds(jst.table_ncols[t]))
end
@inline function residual_block(J::SchurBlockJacobian, t::Integer)
    jst = J.structure
    @boundscheck (1 <= t <= jst.ntables) || throw(BoundsError(J, t))
    return residual_block(J, t, 1, data_rows(jst, Int(t)))
end

# Element data of the workspace: a factored copy of the tables, and per-chunk scratch for blocks above D_STATIC
qr_nscratch(jst::SchurJacobianStructure) = maximum(length, jst.schur.chunks; init=1)
qr_scratch_length(jst::SchurJacobianStructure) = (d = maximum(jst.schur.e_sizes; init=0); d > D_STATIC ? d : 0)

# schur_jacobian_bytes(structure::SchurJacobianStructure, T=Float64; workspace=true) -> Int
# Bytes of element data allocated by a `SchurBlockJacobian{T}` (and its `QRReductionWorkspace{T}`).
function schur_jacobian_bytes(jst::SchurJacobianStructure, ::Type{T}=Float64; workspace::Bool=true) where T
    n = table_length(jst)
    workspace && (n += table_length(jst) + qr_nscratch(jst) * qr_scratch_length(jst))
    return n * sizeof(T)
end
