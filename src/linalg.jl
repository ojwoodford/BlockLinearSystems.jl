# schur_quadform(H::SchurBlockHessian, dx::AbstractVector) -> Real
# The quadratic form `dx' * H * dx` in O(N) operations without materializing the full Hessian: the fast path of
# `dot(dx, H, dx)`. `dx` may have any element type; the result has the promoted type.
#
# The retained part (`R`) is walked block by block, reading one triangle and doubling the off-diagonal terms. The
# eliminated part (`E` and the couplings `C`) is where nearly all the work is, and is processed per group of
# eliminated blocks of uniform dimension with that dimension known statically (3.4x faster than per-block run-time
# sizes on a bundle-adjustment-like benchmark with 200 retained and 20000 eliminated blocks). Specialising the
# retained part by run of retained block sizes was measured too: it saved nothing measurable (that part takes
# ~0.01 ms) and cost 0.2-2 s of extra compilation, so it is not done.
function schur_quadform(H::SchurBlockHessian{T}, dx::AbstractVector) where T
    Tacc = promote_type(T, eltype(dx))
    structure = H.structure
    nr = structure.nr
    dx_r = view(dx, 1:nr)
    dx_e = view(dx, nr+1:length(dx))
    
    # 1. dx_r' * R * dx_r
    total = zero(Tacc)
    if H.R isa Matrix{T}
        R = H.R
        @inbounds for j in 1:nr
            val = zero(Tacc)
            for i in 1:j
                r_val = R[i, j]
                if i == j
                    val += r_val * dx_r[i]
                else
                    val += r_val * dx_r[i]
                    total += r_val * dx_r[j] * dx_r[i]
                end
            end
            total += dx_r[j] * val
        end
    elseif H.R isa SparseBlockLinearSystem{T}
        # Walk the stored block pattern directly (column i of indicestransposed lists the stored
        # column-blocks j of block row i); each block is column-major at data[ptr:ptr+drow*dcol-1].
        R_sys = H.R
        boffsets = structure.boffsets_r
        data = R_sys.data
        colptr = R_sys.indicestransposed.colptr
        rowval = R_sys.indicestransposed.rowval
        nzval = R_sys.indicestransposed.nzval
        @inbounds for i in 1:length(R_sys.rowblocksizes)
            row_start = boffsets[i]
            drow = R_sys.rowblocksizes[i]
            for k in colptr[i]:(colptr[i + 1] - 1)
                j = rowval[k]
                ptr = nzval[k]
                col_start = boffsets[j]
                dcol = R_sys.columnblocksizes[j]
                blk_total = zero(Tacc)
                for jc in 1:dcol
                    dx_jc = dx_r[col_start + jc - 1]
                    col_ptr = ptr + (jc - 1) * drow - 1
                    col_total = zero(Tacc)
                    for ir in 1:drow
                        col_total += data[col_ptr + ir] * dx_r[row_start + ir - 1]
                    end
                    blk_total += col_total * dx_jc
                end
                total += (i == j ? blk_total : 2 * blk_total)
            end
        end
    end

    # 2. 2 * dx_e' * C * dx_r and 3. dx_e' * E * dx_e, per group of eliminated blocks of uniform dimension.
    # (`z`, a typed zero, is captured by the closure rather than the type `Tacc`: Julia 1.10 widens a captured
    # type to `DataType`, which made the result type inferred as `Any` and boxed it on every call.)
    z = zero(Tacc)
    for g_idx in 1:nruns(structure.e_dims)
        e_range = runblockrange(structure.e_dims, g_idx)
        total += dispatch_dim(runblocksize(structure.e_dims, g_idx)) do s_de
            schur_quadform_eliminated(H, dx_r, dx_e, e_range, s_de, z)
        end
    end

    return total
end

# C and E contributions of the eliminated blocks in `e_range`, all of dimension `s_de` (static up to D_STATIC)
function schur_quadform_eliminated(H::SchurBlockHessian{T}, dx_r, dx_e, e_range::UnitRange{Int}, s_de::StaticInt, z::Tacc) where {T, Tacc}
    structure = H.structure
    D = dimval(s_de)
    part = z
    @inbounds for e in e_range
        e_off = structure.boffsets_e[e]
        xp = make_svec(dx_e, e_off, s_de)
        part += dot(xp, make_smat(H.E_data, structure.eoffsets[e], s_de) * xp)
        for c in structure.eliminated_ptr[e]:(structure.eliminated_ptr[e + 1] - 1)
            r = structure.entry_retained[c]
            dr = Int(structure.r_dims[r])
            ro = structure.boffsets_r[r]
            co = structure.coffsets[c]
            for ir in 1:dr
                part += 2 * dx_r[ro + ir - 1] * dot(make_svec(H.C_data, co + (ir - 1) * D, s_de), xp)
            end
        end
    end
    return part
end

# Blocks above D_STATIC dimensions: the same computation with run-time loops
function schur_quadform_eliminated(H::SchurBlockHessian{T}, dx_r, dx_e, e_range::UnitRange{Int}, de::Int, z::Tacc) where {T, Tacc}
    structure = H.structure
    part = z
    @inbounds for e in e_range
        e_off = structure.boffsets_e[e]
        eo = structure.eoffsets[e]
        for col in 1:de
            xcol = dx_e[e_off + col - 1]
            for row in 1:de
                part += H.E_data[eo + (col - 1) * de + row - 1] * xcol * dx_e[e_off + row - 1]
            end
        end
        for c in structure.eliminated_ptr[e]:(structure.eliminated_ptr[e + 1] - 1)
            r = structure.entry_retained[c]
            dr = Int(structure.r_dims[r])
            ro = structure.boffsets_r[r]
            co = structure.coffsets[c]
            for ir in 1:dr
                acc = zero(Tacc)
                for i in 1:de
                    acc += H.C_data[co + (ir - 1) * de + i - 1] * dx_e[e_off + i - 1]
                end
                part += 2 * dx_r[ro + ir - 1] * acc
            end
        end
    end
    return part
end

function LinearAlgebra.dot(x::AbstractVector{<:Real}, H::SchurBlockHessian{<:Real}, y::AbstractVector{<:Real})
    st = H.structure
    n = st.nr + (st.neblocks == 0 ? 0 : st.boffsets_e[end] + Int(st.e_dims[end]) - 1)
    (length(x) == n && length(y) == n) ||
        throw(DimensionMismatch("x has length $(length(x)) and y has length $(length(y)); H has dimension $n"))
    x === y && return schur_quadform(H, x)
    return schur_bilinear(H, x, y)
end

# schur_bilinear(H::SchurBlockHessian, x, y) -> Real
# The bilinear form `x' * H * y` for any two vectors, by the walk of `schur_quadform`: H is symmetric and one
# triangle of R is stored or read, so each doubled term `2·a·b` of the quadratic form is `aₓ·b_y + a_y·bₓ` here.
function schur_bilinear(H::SchurBlockHessian{T}, x::AbstractVector, y::AbstractVector) where T
    Tacc = promote_type(T, eltype(x), eltype(y))
    structure = H.structure
    nr = structure.nr
    x_r, x_e = view(x, 1:nr), view(x, nr+1:length(x))
    y_r, y_e = view(y, 1:nr), view(y, nr+1:length(y))

    # 1. x_r' * R * y_r, from the upper triangle of a dense R or the stored blocks of a sparse R
    total = zero(Tacc)
    if H.R isa Matrix{T}
        R = H.R
        @inbounds for j in 1:nr
            xj, yj = x_r[j], y_r[j]
            for i in 1:j-1
                r_val = R[i, j]   # multiplied first, so the products are in the promoted type
                total += r_val * x_r[i] * yj + r_val * xj * y_r[i]
            end
            total += R[j, j] * xj * yj
        end
    elseif H.R isa SparseBlockLinearSystem{T}
        R_sys = H.R
        boffsets = structure.boffsets_r
        data = R_sys.data
        colptr = R_sys.indicestransposed.colptr
        rowval = R_sys.indicestransposed.rowval
        nzval = R_sys.indicestransposed.nzval
        @inbounds for i in 1:length(R_sys.rowblocksizes)
            row_start = boffsets[i]
            drow = R_sys.rowblocksizes[i]
            for k in colptr[i]:(colptr[i + 1] - 1)
                j = rowval[k]
                ptr = nzval[k]
                col_start = boffsets[j]
                dcol = R_sys.columnblocksizes[j]
                for jc in 1:dcol
                    col_ptr = ptr + (jc - 1) * drow - 1
                    bx = zero(Tacc)   # column jc of block (i, j) times x_i
                    by = zero(Tacc)   # ... and times y_i
                    for ir in 1:drow
                        b = data[col_ptr + ir]
                        bx += b * x_r[row_start + ir - 1]
                        by += b * y_r[row_start + ir - 1]
                    end
                    total += bx * y_r[col_start + jc - 1]                       # x_i' B y_j
                    i == j || (total += by * x_r[col_start + jc - 1])          # y_i' B x_j
                end
            end
        end
    end

    # 2. The coupling terms x_e' C y_r + y_e' C x_r and 3. x_e' E y_e, per group of eliminated blocks of uniform
    # dimension (`z` is captured rather than `Tacc`; see `schur_quadform`)
    z = zero(Tacc)
    for g_idx in 1:nruns(structure.e_dims)
        e_range = runblockrange(structure.e_dims, g_idx)
        total += dispatch_dim(runblocksize(structure.e_dims, g_idx)) do s_de
            schur_bilinear_eliminated(H, x_r, x_e, y_r, y_e, e_range, s_de, z)
        end
    end
    return total
end

# C and E contributions of the eliminated blocks in `e_range`, all of dimension `s_de` (static up to D_STATIC)
function schur_bilinear_eliminated(H::SchurBlockHessian{T}, x_r, x_e, y_r, y_e, e_range::UnitRange{Int}, s_de::StaticInt, z::Tacc) where {T, Tacc}
    structure = H.structure
    D = dimval(s_de)
    part = z
    @inbounds for e in e_range
        e_off = structure.boffsets_e[e]
        xp = make_svec(x_e, e_off, s_de)
        yp = make_svec(y_e, e_off, s_de)
        part += dot(xp, make_smat(H.E_data, structure.eoffsets[e], s_de) * yp)
        for c in structure.eliminated_ptr[e]:(structure.eliminated_ptr[e + 1] - 1)
            r = structure.entry_retained[c]
            dr = Int(structure.r_dims[r])
            ro = structure.boffsets_r[r]
            co = structure.coffsets[c]
            for ir in 1:dr
                cv = make_svec(H.C_data, co + (ir - 1) * D, s_de)
                part += x_r[ro + ir - 1] * dot(cv, yp) + y_r[ro + ir - 1] * dot(cv, xp)
            end
        end
    end
    return part
end

# Blocks above D_STATIC dimensions: the same computation with run-time loops
function schur_bilinear_eliminated(H::SchurBlockHessian{T}, x_r, x_e, y_r, y_e, e_range::UnitRange{Int}, de::Int, z::Tacc) where {T, Tacc}
    structure = H.structure
    part = z
    @inbounds for e in e_range
        e_off = structure.boffsets_e[e]
        eo = structure.eoffsets[e]
        for col in 1:de
            ycol = y_e[e_off + col - 1]
            for row in 1:de
                part += H.E_data[eo + (col - 1) * de + row - 1] * x_e[e_off + row - 1] * ycol
            end
        end
        for c in structure.eliminated_ptr[e]:(structure.eliminated_ptr[e + 1] - 1)
            r = structure.entry_retained[c]
            dr = Int(structure.r_dims[r])
            ro = structure.boffsets_r[r]
            co = structure.coffsets[c]
            for ir in 1:dr
                cx = zero(Tacc)
                cy = zero(Tacc)
                for i in 1:de
                    cval = H.C_data[co + (ir - 1) * de + i - 1]
                    cx += cval * x_e[e_off + i - 1]
                    cy += cval * y_e[e_off + i - 1]
                end
                part += x_r[ro + ir - 1] * cy + y_r[ro + ir - 1] * cx
            end
        end
    end
    return part
end
