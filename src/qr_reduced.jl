using Static, StaticArrays, SparseArrays, LinearAlgebra

# reduced_nrows(structure::SchurJacobianStructure) -> Int
# Number of rows of the reduced Jacobian `J̃`.
reduced_nrows(jst::SchurJacobianStructure) = jst.reduced_offsets[end] - 1

# free_block_pairs(structure::SchurJacobianStructure) -> Vector{Tuple{Int, Int}}
# Pairs of retained blocks coupled through a free group.
function free_block_pairs(jst::SchurJacobianStructure)
    pairs = Tuple{Int, Int}[]
    for t in jst.schur.neblocks + 1:jst.ntables
        rng = jst.table_ptr[t]:jst.table_ptr[t + 1] - 1
        for u2 in rng, u1 in first(rng):u2 - 1
            push!(pairs, (jst.entry_retained[u1], jst.entry_retained[u2]))
        end
    end
    return unique!(pairs)
end

# ------------------------------------------------------------------------------------------------------------------
# Back-substitution
# ------------------------------------------------------------------------------------------------------------------

function qr_backsubstitute_group!(
    x_e::AbstractVector{T}, x_r::AbstractVector, F::Vector{T}, jst::SchurJacobianStructure,
    ::Vector{T}, e_chunk::UnitRange{Int}, s_de::StaticInt{De}
) where {T, De}
    st = jst.schur
    @inbounds for e in e_chunk
        M = jst.table_nrows[e]
        C = jst.table_ncols[e]
        off = jst.table_offsets[e]
        ce = C - De - 1
        t = make_svec(F, off + (C - 1) * M, s_de)                    # q
        for u in jst.table_ptr[e]:jst.table_ptr[e + 1] - 1
            r = jst.entry_retained[u]
            ro = st.boffsets_r[r]
            cb = off + (jst.entry_col[u] - 1) * M
            for j in 1:Int(st.r_dims[r])
                t -= make_svec(F, cb + (j - 1) * M, s_de) * x_r[ro + j - 1]
            end
        end
        U = SMatrix{De, De, T, De * De}(ntuple(Val(De * De)) do idx
            i = (idx - 1) % De + 1
            j = (idx - 1) ÷ De + 1
            F[off + (ce + j - 1) * M + i - 1]
        end)
        xe = UpperTriangular(U) \ t
        eo = st.boffsets_e[e]
        for i in 1:De
            x_e[eo + i - 1] = xe[i]
        end
    end
    return nothing
end

function qr_backsubstitute_group!(
    x_e::AbstractVector{T}, x_r::AbstractVector, F::Vector{T}, jst::SchurJacobianStructure,
    tbuf::Vector{T}, e_chunk::UnitRange{Int}, de::Int
) where T
    st = jst.schur
    @inbounds for e in e_chunk
        M = jst.table_nrows[e]
        C = jst.table_ncols[e]
        off = jst.table_offsets[e]
        ce = C - de - 1
        qb = off + (C - 1) * M
        for i in 1:de
            tbuf[i] = F[qb + i - 1]
        end
        for u in jst.table_ptr[e]:jst.table_ptr[e + 1] - 1
            r = jst.entry_retained[u]
            ro = st.boffsets_r[r]
            cb = off + (jst.entry_col[u] - 1) * M
            for j in 1:Int(st.r_dims[r])
                xj = x_r[ro + j - 1]
                for i in 1:de
                    tbuf[i] -= F[cb + (j - 1) * M + i - 1] * xj
                end
            end
        end
        eo = st.boffsets_e[e]
        for i in de:-1:1
            acc = tbuf[i]
            for j in i + 1:de
                acc -= F[off + (ce + j - 1) * M + i - 1] * x_e[eo + j - 1]
            end
            x_e[eo + i - 1] = acc / F[off + (ce + i - 1) * M + i - 1]
        end
    end
    return nothing
end

# backsubstitute!(x_e, x_r, ws::QRReductionWorkspace)
# Recover the eliminated variables `x_e = U⁻¹ (q - B x_r)` from the retained ones, per table.
function backsubstitute!(x_e::AbstractVector{T}, x_r::AbstractVector, ws::QRReductionWorkspace{T}) where T
    jst = ws.structure
    st = jst.schur
    threaded = qr_threaded(jst, st.neblocks)
    for (g_idx, e_range) in enumerate(st.e_groups)
        chunks = st.chunks[g_idx]
        dispatch_dim(st.e_sizes[g_idx]) do s_de
            foreach_index(length(chunks), threaded) do k
                qr_backsubstitute_group!(x_e, x_r, ws.factored, jst, ws.scratch[k], chunks[k], s_de)
            end
        end
    end
    return nothing
end

# ------------------------------------------------------------------------------------------------------------------
# The reduced problem
# ------------------------------------------------------------------------------------------------------------------

# Call f(r) for every retained block, threaded over the disjoint column chunks
@inline function foreach_retained(f::F, jst::SchurJacobianStructure) where F
    chunks = jst.col_chunks
    foreach_index(length(chunks), qr_threaded(jst, length(chunks))) do k
        for r in chunks[k]
            f(r)
        end
    end
    return nothing
end

# Call f(i) for i in 1:n, threaded over K contiguous ranges. Polyester only sees the wrapper (f, n, K), which keeps
# its per-launch allocation small and constant (handing it f directly allocated per item on Julia 1.10).
@inline function foreach_range(f::F, n::Int, K::Int, threaded::Bool) where F
    K = max(min(K, n), 1)
    foreach_index(K, threaded) do k
        for i in (div((k - 1) * n, K) + 1):div(k * n, K)
            f(i)
        end
    end
    return nothing
end

@inline foreach_table(f::F, jst::SchurJacobianStructure) where F =
    foreach_range(f, jst.ntables, Threads.nthreads(), qr_threaded(jst, jst.ntables))

# dot of n contiguous entries of a and b
@inline function dot_run(a::AbstractVector{T}, ia::Int, b::AbstractVector, ib::Int, n::Int) where T
    acc = zero(promote_type(T, eltype(b)))
    @inbounds for i in 0:n - 1
        acc += a[ia + i] * b[ib + i]
    end
    return acc
end

# reduced_mul!(y, ws::QRReductionWorkspace, v)
# `y = J̃ v`.
function reduced_mul!(y::AbstractVector, ws::QRReductionWorkspace, v::AbstractVector)
    jst = ws.structure
    st = jst.schur
    F = ws.factored
    length(y) == reduced_nrows(jst) || throw(DimensionMismatch("y has length $(length(y)), expected $(reduced_nrows(jst))"))
    length(v) == st.nr || throw(DimensionMismatch("v has length $(length(v)), expected $(st.nr)"))
    foreach_table(jst) do t
        @inbounds begin
            M = jst.table_nrows[t]
            top = jst.table_top[t]
            off = jst.table_offsets[t]
            yo = jst.reduced_offsets[t]
            n = M - top
            for i in 0:n - 1
                y[yo + i] = zero(eltype(y))
            end
            for u in jst.table_ptr[t]:jst.table_ptr[t + 1] - 1
                r = jst.entry_retained[u]
                ro = st.boffsets_r[r]
                cb = off + (jst.entry_col[u] - 1) * M + top
                for j in 1:Int(st.r_dims[r])
                    vj = v[ro + j - 1]
                    c0 = cb + (j - 1) * M
                    for i in 0:n - 1
                        y[yo + i] += F[c0 + i] * vj
                    end
                end
            end
        end
    end
    return y
end

# reduced_tmul!(g, ws::QRReductionWorkspace, w)
# `g = J̃ᵀ w`.
function reduced_tmul!(g::AbstractVector, ws::QRReductionWorkspace, w::AbstractVector)
    jst = ws.structure
    st = jst.schur
    F = ws.factored
    length(g) == st.nr || throw(DimensionMismatch("g has length $(length(g)), expected $(st.nr)"))
    length(w) == reduced_nrows(jst) || throw(DimensionMismatch("w has length $(length(w)), expected $(reduced_nrows(jst))"))
    foreach_retained(jst) do r
        @inbounds begin
            ro = st.boffsets_r[r]
            dr = Int(st.r_dims[r])
            for j in 0:dr - 1
                g[ro + j] = zero(eltype(g))
            end
            for k in jst.retained_ptr[r]:jst.retained_ptr[r + 1] - 1
                u = jst.retained_entry[k]
                t = jst.entry_table[u]
                M = jst.table_nrows[t]
                top = jst.table_top[t]
                cb = jst.table_offsets[t] + (jst.entry_col[u] - 1) * M + top
                wo = jst.reduced_offsets[t]
                for j in 1:dr
                    g[ro + j - 1] += dot_run(F, cb + (j - 1) * M, w, wo, M - top)
                end
            end
        end
    end
    return g
end

# reduced_residual!(rt, ws::QRReductionWorkspace)
# `r̃`, in the row order of the reduced problem.
function reduced_residual!(rt::AbstractVector, ws::QRReductionWorkspace)
    jst = ws.structure
    F = ws.factored
    length(rt) == reduced_nrows(jst) || throw(DimensionMismatch("rt has length $(length(rt)), expected $(reduced_nrows(jst))"))
    foreach_table(jst) do t
        @inbounds begin
            M = jst.table_nrows[t]
            top = jst.table_top[t]
            cb = jst.table_offsets[t] + (jst.table_ncols[t] - 1) * M + top
            ro = jst.reduced_offsets[t]
            for i in 0:M - top - 1
                rt[ro + i] = F[cb + i]
            end
        end
    end
    return rt
end

# reduced_jacobian!(Jt::AbstractMatrix, rt::AbstractVector, ws::QRReductionWorkspace)
# Write the reduced problem densely: `Jt = J̃` (`reduced_nrows × nr`) and `rt = r̃`.
function reduced_jacobian!(Jt::AbstractMatrix, rt::AbstractVector, ws::QRReductionWorkspace)
    jst = ws.structure
    st = jst.schur
    F = ws.factored
    size(Jt) == (reduced_nrows(jst), st.nr) || throw(DimensionMismatch("Jt has size $(size(Jt)), expected $((reduced_nrows(jst), st.nr))"))
    fill!(Jt, zero(eltype(Jt)))
    reduced_residual!(rt, ws)
    foreach_table(jst) do t
        @inbounds begin
            M = jst.table_nrows[t]
            top = jst.table_top[t]
            off = jst.table_offsets[t]
            yo = jst.reduced_offsets[t]
            for u in jst.table_ptr[t]:jst.table_ptr[t + 1] - 1
                r = jst.entry_retained[u]
                ro = st.boffsets_r[r]
                cb = off + (jst.entry_col[u] - 1) * M + top
                for j in 1:Int(st.r_dims[r]), i in 0:M - top - 1
                    Jt[yo + i, ro + j - 1] = F[cb + (j - 1) * M + i]
                end
            end
        end
    end
    return Jt
end

# Contributions of entry u2 (table t, retained block r2) to g_S[r2] and to block column r2 of S, through every
# entry u1 <= u2 of the same table (entries are sorted by retained block within a table, so r1 <= r2).
# `slot(u1, row, col)` returns the index in `Sv` of S[row of r1, col of r2].
@inline function accumulate_qr_entry!(slot::Fs, Sv::AbstractArray{T}, g_S::AbstractVector{T}, F::Vector{T},
                                      jst::SchurJacobianStructure, u2::Int) where {Fs, T}
    st = jst.schur
    @inbounds begin
        t = jst.entry_table[u2]
        M = jst.table_nrows[t]
        top = jst.table_top[t]
        n = M - top
        off = jst.table_offsets[t] + top
        r2 = jst.entry_retained[u2]
        dr2 = Int(st.r_dims[r2])
        ro2 = st.boffsets_r[r2]
        cb2 = off + (jst.entry_col[u2] - 1) * M
        rb = off + (jst.table_ncols[t] - 1) * M
        for j in 1:dr2
            g_S[ro2 + j - 1] += dot_run(F, cb2 + (j - 1) * M, F, rb, n)
        end
        for u1 in jst.table_ptr[t]:u2
            dr1 = Int(st.r_dims[jst.entry_retained[u1]])
            cb1 = off + (jst.entry_col[u1] - 1) * M
            base = slot(u1)
            for col in 1:dr2
                nrow = u1 == u2 ? col : dr1      # diagonal block: upper triangle only
                for row in 1:nrow
                    Sv[base(row, col)] += dot_run(F, cb1 + (row - 1) * M, F, cb2 + (col - 1) * M, n)
                end
            end
        end
    end
    return nothing
end

# reduced_normal_dense!(S, g_S, ws::QRReductionWorkspace)
# Overwrite the upper triangle of `S` with `J̃ᵀJ̃` and `g_S` with `J̃ᵀr̃`.
function reduced_normal_dense!(S::Matrix{T}, g_S::AbstractVector{T}, ws::QRReductionWorkspace{T}) where T
    jst = ws.structure
    st = jst.schur
    F = ws.factored
    size(S) == (st.nr, st.nr) || throw(DimensionMismatch("S has size $(size(S)), expected $((st.nr, st.nr))"))
    length(g_S) == st.nr || throw(DimensionMismatch("g_S has length $(length(g_S)), expected $(st.nr)"))
    nr = st.nr
    foreach_retained(jst) do r2
        @inbounds begin
            ro2 = st.boffsets_r[r2]
            dr2 = Int(st.r_dims[r2])
            for col in ro2:ro2 + dr2 - 1
                for row in 1:nr
                    S[row, col] = zero(T)
                end
                g_S[col] = zero(T)
            end
            for k in jst.retained_ptr[r2]:jst.retained_ptr[r2 + 1] - 1
                u2 = jst.retained_entry[k]
                accumulate_qr_entry!(S, g_S, F, jst, u2) do u1
                    base = (ro2 - 2) * nr + st.boffsets_r[jst.entry_retained[u1]] - 1
                    (row, col) -> base + (col * nr) + row     # linear index of S[ro1 + row - 1, ro2 + col - 1]
                end
            end
        end
    end
    return true
end

# reduced_normal_sparse!(S::SparseMatrixCSC, g_S, ws::QRReductionWorkspace, pattern)
# Sparse counterpart of [`reduced_normal_dense!`](@ref), writing `S.nzval`.
function reduced_normal_sparse!(S::SparseMatrixCSC{T}, g_S::AbstractVector{T}, ws::QRReductionWorkspace{T}, pattern::NamedTuple) where T
    jst = ws.structure
    st = jst.schur
    F = ws.factored
    size(S) == (st.nr, st.nr) || throw(DimensionMismatch("S has size $(size(S)), expected $((st.nr, st.nr))"))
    length(g_S) == st.nr || throw(DimensionMismatch("g_S has length $(length(g_S)), expected $(st.nr)"))
    nzval = S.nzval
    S_colptr = S.colptr
    block_colptr = pattern.block_colptr
    block_rowval = pattern.block_rowval
    block_row_offsets = pattern.block_row_offsets
    col_heights = pattern.col_heights
    foreach_retained(jst) do r2
        @inbounds begin
            ro2 = st.boffsets_r[r2]
            dr2 = Int(st.r_dims[r2])
            for i in S_colptr[ro2]:S_colptr[ro2 + dr2] - 1
                nzval[i] = zero(T)
            end
            for col in ro2:ro2 + dr2 - 1
                g_S[col] = zero(T)
            end
            col_base = Int(S_colptr[ro2])
            stride = col_heights[r2]
            b_lo = Int(block_colptr[r2])
            b_hi = Int(block_colptr[r2 + 1]) - 1
            for k in jst.retained_ptr[r2]:jst.retained_ptr[r2 + 1] - 1
                u2 = jst.retained_entry[k]
                accumulate_qr_entry!(nzval, g_S, F, jst, u2) do u1
                    r1 = jst.entry_retained[u1]
                    rng = b_lo:b_hi
                    idx = searchsortedfirst(view(block_rowval, rng), r1)
                    (idx <= length(rng) && block_rowval[rng[idx]] == r1) || throw(AssertionError("Sparse pattern missing entry ($r1, $r2)"))
                    row_offset = block_row_offsets[rng[idx]]
                    (row, col) -> col_base + (col - 1) * stride + row_offset + row - 1
                end
            end
        end
    end
    return true
end

# reduced_diagonal_blocks!(D::AbstractVector, ws::QRReductionWorkspace)
# The diagonal blocks of `J̃ᵀJ̃`, one full `d_r × d_r` block per retained block.
function reduced_diagonal_blocks!(D::AbstractVector{T}, ws::QRReductionWorkspace{T}) where T
    jst = ws.structure
    st = jst.schur
    F = ws.factored
    length(D) == jst.doffsets[end] - 1 || throw(DimensionMismatch("D has length $(length(D)), expected $(jst.doffsets[end] - 1)"))
    foreach_retained(jst) do r
        @inbounds begin
            dr = Int(st.r_dims[r])
            d0 = jst.doffsets[r]
            for i in d0:d0 + dr * dr - 1
                D[i] = zero(T)
            end
            for k in jst.retained_ptr[r]:jst.retained_ptr[r + 1] - 1
                u = jst.retained_entry[k]
                t = jst.entry_table[u]
                M = jst.table_nrows[t]
                top = jst.table_top[t]
                cb = jst.table_offsets[t] + top + (jst.entry_col[u] - 1) * M
                for col in 1:dr, row in 1:col
                    D[d0 + (col - 1) * dr + row - 1] += dot_run(F, cb + (row - 1) * M, F, cb + (col - 1) * M, M - top)
                end
            end
            for col in 1:dr, row in col + 1:dr
                D[d0 + (col - 1) * dr + row - 1] = D[d0 + (row - 1) * dr + col - 1]
            end
        end
    end
    return D
end

# ------------------------------------------------------------------------------------------------------------------
# Quantities of the full problem, from the raw or the factored tables
# ------------------------------------------------------------------------------------------------------------------

table_data(J::SchurBlockJacobian) = J.data
table_data(ws::QRReductionWorkspace) = ws.factored

const SchurTables{T} = Union{SchurBlockJacobian{T}, QRReductionWorkspace{T}}

# f(col_base_index) summed over every column of retained block r, in every table, over all rows of the table
@inline function sum_retained_columns!(out::AbstractVector, src, r::Int, colfun::Fc) where Fc
    jst = src.structure
    st = jst.schur
    @inbounds begin
        ro = st.boffsets_r[r]
        dr = Int(st.r_dims[r])
        for j in 0:dr - 1
            out[ro + j] = zero(eltype(out))
        end
        for k in jst.retained_ptr[r]:jst.retained_ptr[r + 1] - 1
            u = jst.retained_entry[k]
            t = jst.entry_table[u]
            M = jst.table_nrows[t]
            off = jst.table_offsets[t]
            cb = off + (jst.entry_col[u] - 1) * M
            rb = off + (jst.table_ncols[t] - 1) * M
            for j in 1:dr
                out[ro + j - 1] += colfun(cb + (j - 1) * M, rb, M)
            end
        end
    end
    return nothing
end

@inline function foreach_eliminated_column(f::Ff, src) where Ff
    jst = src.structure
    st = jst.schur
    foreach_range(st.neblocks, Threads.nthreads(), qr_threaded(jst, st.neblocks)) do e
        @inbounds begin
            M = jst.table_nrows[e]
            C = jst.table_ncols[e]
            off = jst.table_offsets[e]
            de = Int(st.e_dims[e])
            eo = st.nr + st.boffsets_e[e]
            for k in 1:de
                f(eo + k - 1, off + (C - de - 2 + k) * M, off + (C - 1) * M, M)
            end
        end
    end
    return nothing
end

# jtr!(g, src::Union{SchurBlockJacobian, QRReductionWorkspace})
# The gradient `g = Jᵀr` of the full problem, `[g_r; g_e]`.
function jtr!(g::AbstractVector, src::SchurTables)
    jst = src.structure
    D = table_data(src)
    length(g) == jst.schur.nr + sum(Int, jst.schur.e_dims; init=0) || throw(DimensionMismatch("g has length $(length(g)), expected $(jst.schur.nr + sum(Int, jst.schur.e_dims; init=0))"))
    colfun = (cb, rb, M) -> dot_run(D, cb, D, rb, M)
    foreach_retained(r -> sum_retained_columns!(g, src, r, colfun), jst)
    foreach_eliminated_column((i, cb, rb, M) -> (@inbounds g[i] = dot_run(D, cb, D, rb, M)), src)
    return g
end

# jtj_diag!(d, src::Union{SchurBlockJacobian, QRReductionWorkspace})
# The squared column norms of `J`, i.e. `diag(JᵀJ)`, `[d_r; d_e]`.
function jtj_diag!(dg::AbstractVector, src::SchurTables)
    jst = src.structure
    D = table_data(src)
    length(dg) == jst.schur.nr + sum(Int, jst.schur.e_dims; init=0) || throw(DimensionMismatch("d has length $(length(dg)), expected $(jst.schur.nr + sum(Int, jst.schur.e_dims; init=0))"))
    colfun = (cb, rb, M) -> dot_run(D, cb, D, cb, M)
    foreach_retained(r -> sum_retained_columns!(dg, src, r, colfun), jst)
    foreach_eliminated_column((i, cb, rb, M) -> (@inbounds dg[i] = dot_run(D, cb, D, cb, M)), src)
    return dg
end

# fast_bAb(src::Union{SchurBlockJacobian, QRReductionWorkspace}, v) -> Real
# `‖J v‖² = vᵀ(JᵀJ)v` for `v = [v_r; v_e]`, from the tables.
function fast_bAb(src::SchurTables{T}, v::AbstractVector) where T
    jst = src.structure
    st = jst.schur
    D = table_data(src)
    total = zero(promote_type(T, eltype(v)))
    for t in 1:jst.ntables
        total += table_quadratic(D, jst, v, t, total)
    end
    return total
end

function table_quadratic(D::Vector{T}, jst::SchurJacobianStructure, v::AbstractVector, t::Int, z::Tacc) where {T, Tacc}
    st = jst.schur
    part = zero(z)
    @inbounds begin
        M = jst.table_nrows[t]
        C = jst.table_ncols[t]
        off = jst.table_offsets[t]
        de = t <= st.neblocks ? Int(st.e_dims[t]) : 0
        eo = t <= st.neblocks ? st.nr + st.boffsets_e[t] : 0
        urng = jst.table_ptr[t]:jst.table_ptr[t + 1] - 1
        for i in 1:M
            yi = zero(z)
            for u in urng
                r = jst.entry_retained[u]
                ro = st.boffsets_r[r]
                cb = off + (jst.entry_col[u] - 1) * M + i - 1
                for j in 1:Int(st.r_dims[r])
                    yi += D[cb + (j - 1) * M] * v[ro + j - 1]
                end
            end
            cb = off + (C - de - 1) * M + i - 1
            for k in 1:de
                yi += D[cb + (k - 1) * M] * v[eo + k - 1]
            end
            part += abs2(yi)
        end
    end
    return part
end
