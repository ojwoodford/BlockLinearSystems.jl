# The full matrix [R C'; C E] a SchurBlockHessian represents, read and written as a dense matrix (for
# check_linear_system). R is a dense Matrix, or a SparseBlockLinearSystem storing the lower-triangle
# blocks; each E_e is stored column-major; each C_er (eliminated rows, retained columns) column-major.
function schur_blocks(f, H::SchurBlockHessian)
    st = H.structure
    rrange(r) = st.boffsets_r[r]:(st.boffsets_r[r] + Int(st.r_dims[r]) - 1)
    erange(e) = st.nr + st.boffsets_e[e] .+ (0:Int(st.e_dims[e]) - 1)
    if H.R isa Matrix
        f(H.R, 1:st.nr, 1:st.nr)
    else
        for r2 in 1:st.nrblocks, r1 in r2:st.nrblocks
            BlockLinearSystems.validblock(H.R, r1, r2) && f(BlockLinearSystems.block(H.R, r1, r2), rrange(r1), rrange(r2))
        end
    end
    for e in 1:st.neblocks
        de = Int(st.e_dims[e])
        f(reshape(view(H.E_data, st.eoffsets[e] .+ (0:de * de - 1)), de, de), erange(e), erange(e))
        for c in st.eliminated_ptr[e]:(st.eliminated_ptr[e + 1] - 1)
            r = st.entry_retained[c]
            f(reshape(view(H.C_data, st.coffsets[c] .+ (0:de * Int(st.r_dims[r]) - 1)), de, Int(st.r_dims[r])), erange(e), rrange(r))
        end
    end
    return nothing
end
function DampedLinearSolve.conformance_matrix(H::SchurBlockHessian)
    n = H.structure.nr + sum(Int, H.structure.e_dims; init=0)
    M = zeros(n, n)
    schur_blocks(H) do blk, rows, cols
        M[rows, cols] .= blk
        rows != cols && (M[cols, rows] .= blk')
    end
    return M
end
function DampedLinearSolve.conformance_fill!(H::SchurBlockHessian, M::AbstractMatrix)
    schur_blocks((blk, rows, cols) -> (blk .= view(M, rows, cols)), H)
    return H
end

# ------------------------------------------------------------------------------------------------------------------
# Conformance (check_linear_system): the dense J and r the tables stand for, and filling the tables from a dense
# factor of the test matrix, wherever the structure has an entry
# ------------------------------------------------------------------------------------------------------------------

# Calls f(table, rows, cols) for each block of the tables: rows are the table's data rows, cols the columns of
# the full problem they hold, with the block's columns inside the table
function foreach_table_block(f, J::SchurBlockJacobian)
    jst = J.structure
    st = jst.schur
    row0 = 0
    for t in 1:jst.ntables
        m = data_rows(jst, t)
        rows = row0 .+ (1:m)
        for u in jst.table_ptr[t]:jst.table_ptr[t+1]-1
            r = jst.entry_retained[u]
            f(t, rows, st.boffsets_r[r] .+ (0:st.r_dims[r]-1), jst.entry_col[u] .+ (0:st.r_dims[r]-1))
        end
        if t <= st.neblocks
            de = Int(st.e_dims[t])
            C = jst.table_ncols[t]
            f(t, rows, st.nr .+ st.boffsets_e[t] .+ (0:de-1), (C - de):(C - 1))
        end
        row0 += m
    end
    return nothing
end
table_view(J::SchurBlockJacobian, t) = reshape(view(J.data, J.structure.table_offsets[t]:J.structure.table_offsets[t+1]-1),
                                               J.structure.table_nrows[t], J.structure.table_ncols[t])

function dense_jacobian(J::SchurBlockJacobian)
    Jd = zeros(jacobian_nrows(J), jacobian_ncols(J))
    foreach_table_block(J) do t, rows, cols, tcols
        Jd[rows, cols] .= table_view(J, t)[1:length(rows), tcols]
    end
    return Jd
end
function dense_residual(J::SchurBlockJacobian)
    jst = J.structure
    return reduce(vcat, [table_view(J, t)[1:data_rows(jst, t), end] for t in 1:jst.ntables])
end

DampedLinearSolve.conformance_matrix(A::SchurJacobianHessian) = conformance_gram(dense_jacobian(A.J))
# JᵀJ, summed in row order (as the column norms are summed), so its diagonal equals them exactly
function conformance_gram(J::AbstractMatrix)
    n = size(J, 2)
    M = Matrix{Float64}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        total = 0.0
        for k in axes(J, 1)
            total += J[k, i] * J[k, j]
        end
        M[i, j] = total
    end
    return M
end
# A dense m × n factor J with JᵀJ = M (its largest min(m, n) eigenpairs, through a fixed orthonormal basis; a
# non-finite diagonal entry of M is written into the first row of J), as DampedLinearSolve's Jacobian storage is filled
function conformance_factor(M::AbstractMatrix, m::Integer, n::Integer)
    k = min(m, n)
    Mf = [isfinite(M[i, j]) ? Float64(M[i, j]) : 0.0 for i in 1:n, j in 1:n]
    E = eigen(Symmetric((Mf + Mf') / 2))
    idx = n-k+1:n
    W = Matrix(qr([sin(0.9 * i + 1.1 * j + 0.2 * i * j) for i in 1:m, j in 1:k]).Q)[:, 1:k]
    J = W * Diagonal(sqrt.(max.(E.values[idx], 0.0))) * E.vectors[:, idx]'
    for i in 1:n
        isfinite(M[i, i]) || (J[1, i] = M[i, i])
    end
    return J
end
# A factor Jd with JᵀJ = M that the tables can hold exactly, when M has the pattern of JᵀJ and is positive
# definite: the Cholesky factor U of M with the eliminated variables first. Its rows for eliminated block e are
# [coupling | U_e], nonzero only in e's coupled retained blocks, and go in the first d_e rows of e's table; its
# last n_r rows (the Schur complement's factor) go in the first free table, which must couple to every retained
# block. Returns nothing when the structure or M does not allow it.
function exact_factor(J::SchurBlockJacobian, M::AbstractMatrix)
    jst = J.structure
    st = jst.schur
    nr, n = st.nr, jacobian_ncols(J)
    all(isfinite, M) || return nothing
    all(e -> data_rows(jst, e) >= st.e_dims[e], 1:st.neblocks) || return nothing
    t = st.neblocks + 1
    (t <= jst.ntables && data_rows(jst, t) >= nr && jst.table_ptr[t+1] - jst.table_ptr[t] == st.nrblocks) || return nothing
    perm = [nr+1:n; 1:nr]
    F = cholesky(Symmetric(Matrix{Float64}(M[perm, perm])); check = false)
    issuccess(F) || return nothing
    U = zeros(n, n)
    U[:, perm] .= F.U                          # columns back in the problem's order
    Jd = zeros(jacobian_nrows(J), n)
    row0 = 0
    for tt in 1:jst.ntables
        m = data_rows(jst, tt)
        if tt <= st.neblocks
            de = Int(st.e_dims[tt]); e0 = st.boffsets_e[tt]
            Jd[row0 .+ (1:de), :] .= U[e0 .+ (0:de-1), :]
        elseif tt == t
            Jd[row0 .+ (1:nr), :] .= U[n-nr+1:n, :]
        end
        row0 += m
    end
    return Jd
end

function DampedLinearSolve.conformance_fill!(A::SchurJacobianHessian, M::AbstractMatrix)
    J = A.J
    Jd = something(exact_factor(J, M), conformance_factor(M, jacobian_nrows(J), jacobian_ncols(J)))
    foreach_table_block(J) do t, rows, cols, tcols
        table_view(J, t)[1:length(rows), tcols] .= Jd[rows, cols]
    end
    # a non-finite diagonal entry of M goes into the first row that holds its column
    placed = falses(size(M, 1))
    foreach_table_block(J) do t, rows, cols, tcols
        for (k, c) in enumerate(cols)
            if !isfinite(M[c, c]) && !placed[c]
                table_view(J, t)[1, tcols[k]] = M[c, c]
                placed[c] = true
            end
        end
    end
    return A
end
function DampedLinearSolve.conformance_fillb!(ls::QRSchurLinearSystem, b::AbstractVector)
    J = ls.A.J
    Jd = dense_jacobian(J)
    ls.b .= b
    r = all(isfinite, Jd) ? transpose(Jd) \ b : fill(NaN, size(Jd, 1))
    jst = J.structure
    row0 = 0
    for t in 1:jst.ntables
        m = data_rows(jst, t)
        table_view(J, t)[1:m, end] .= r[row0 .+ (1:m)]
        row0 += m
    end
    return ls
end
DampedLinearSolve.conformance_traits(A::SchurJacobianHessian) = (exact = false, maxrank = min(jacobian_nrows(A.J), size(A, 1)), indefinite = false, accumulates = false)
