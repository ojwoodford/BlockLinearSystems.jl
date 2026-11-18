# QR test problems (QRTestProblem and its builders) and the helpers that compare QR elimination with the Schur
# reduction and with dense least squares. Loaded once per session into Main by fixtures/load.jl; no testsets.
module QRFixtures

using Random, LinearAlgebra, SparseArrays, BlockLinearSystems
using BlockLinearSystems: SchurReductionWorkspace, build_r_to_s_map, retained_block_pairs, reset_schur_dense!, reset_schur_sparse!, reduce_schur_dense!, reduce_schur_sparse!, backsubstitute!, QRReductionWorkspace, reduce_qr!, reduced_nrows, free_block_pairs, reduced_jacobian!   # internal: the kernels, withdrawn from the exports by EL6
using Main.SchurFixtures
using BlockLinearSystems: eliminated_block, coupling_block   # internal: withdrawn from the exports by EL4

export QRTestProblem, qr_problems, qr_scaling, qr_dense, qr_dense_r, relerr, schur_hessian, schur_reduced, qr_solve, dense_ls, qr_rtol, sparse_reduced_problems

"""
    QRTestProblem

Random tables for a tall problem `min ‖J x - r‖²`, filled through the assembly API, together with the same `J`
and `r` assembled densely (rows in table order: eliminated tables, then free groups; columns `[x_r; x_e]`).
"""
struct QRTestProblem{T}
    name::String
    r_dims::Vector{Int}
    e_dims::Vector{Int}
    couplings::Vector{Vector{Int}}
    J::SchurBlockJacobian{T, Matrix{T}}
    Jf::Matrix{T}
    rf::Vector{T}
end

function QRTestProblem(name, rng, ::Type{T}, r_dims, e_dims, couplings, e_rows;
                       free_rows=Int[], free_retained=Vector{Int}[], nchunks=1, min_work=BlockLinearSystems.DEFAULT_MIN_WORK) where T
    jst = SchurJacobianStructure(r_dims, e_dims, couplings, e_rows; free_rows, free_retained, nchunks, min_work)
    J = SchurBlockJacobian{T, Matrix{T}}(jst)
    st = jst.schur
    nr, ne = sum(r_dims), sum(e_dims)
    m = sum(e_rows) + sum(free_rows; init=0)
    Jf = zeros(T, m, nr + ne)
    rf = zeros(T, m)
    row = 0
    for t in 1:jst.ntables
        mt = t <= st.neblocks ? e_rows[t] : free_rows[t - st.neblocks]
        rows = row+1:row+mt
        for u in jst.table_ptr[t]:jst.table_ptr[t+1]-1
            r = jst.entry_retained[u]
            B = randn(rng, T, mt, r_dims[r])
            coupling_block(J, u) .= B
            Jf[rows, st.boffsets_r[r]:st.boffsets_r[r]+r_dims[r]-1] .= B
        end
        if t <= st.neblocks
            E = randn(rng, T, mt, e_dims[t])
            eliminated_block(J, t) .= E
            Jf[rows, nr .+ (st.boffsets_e[t]:st.boffsets_e[t]+e_dims[t]-1)] .= E
        end
        rv = randn(rng, T, mt)
        BlockLinearSystems.table_residual_block(J, t) .= rv
        rf[rows] .= rv
        row += mt
    end
    return QRTestProblem{T}(name, collect(r_dims), collect(e_dims), [collect(c) for c in couplings], J, Jf, rf)
end

function qr_problems(::Type{T}; nchunks=1, seed=1, min_work=BlockLinearSystems.DEFAULT_MIN_WORK) where T
    rng = Random.Xoshiro(seed)
    probs = QRTestProblem{T}[]
    # Bundle-adjustment-like: many small eliminated blocks, each coupled to a few retained blocks, 2 rows per coupling
    ba_c = [sort(randperm(rng, 8)[1:rand(rng, 2:4)]) for _ in 1:60]
    push!(probs, QRTestProblem("bundle adjustment", rng, T, fill(6, 8), fill(3, 60), ba_c, 2 .* length.(ba_c); nchunks, min_work))
    # Mixed eliminated dimensions, one above D_STATIC and two of dimension 1
    mixed_e = [12, 9, 8, 5, 3, 2, 1, 1]
    mixed_c = [[1, 3], [2, 4], [1, 2, 4], [3], [1, 4], [2, 3], [1], [4]]
    push!(probs, QRTestProblem("mixed dimensions", rng, T, [6, 3, 9, 6], mixed_e, mixed_c, mixed_e .+ 4 .+ rand(rng, 0:6, 8); nchunks, min_work))
    # Free groups: rows that involve no eliminated block, one of them the only rows of retained block 5
    push!(probs, QRTestProblem("free groups", rng, T, [6, 6, 6, 6, 3], [3, 3, 3, 2, 2, 1], [[1, 2], [2, 3], [3, 4], [1, 4], [2], [1, 3]],
                               [8, 8, 7, 6, 5, 4]; free_rows=[7, 3, 6], free_retained=[[2, 1], [3], [5, 4, 2]], nchunks, min_work))
    # An eliminated block coupled to every retained block
    push!(probs, QRTestProblem("coupled to all", rng, T, [6, 6, 3, 6], [3, 3, 3, 3], [[1, 2, 3, 4], [1, 2], [3, 4], [2, 4]],
                               [24, 8, 8, 8]; nchunks, min_work))
    # A single eliminated block
    push!(probs, QRTestProblem("single eliminated block", rng, T, [6, 3], [4], [[1, 2]], [20]; nchunks, min_work))
    return probs
end

# Damping diagonal: Marquardt-like (diag(J'J)) but deliberately non-uniform
qr_scaling(p::QRTestProblem{T}, rng) where T = vec(sum(abs2, p.Jf; dims=1)) .* T.(0.5 .+ rand(rng, size(p.Jf, 2)))

# Dense J and r from the tables (rows in table order), for problems whose tables were modified after construction
function qr_dense(J::SchurBlockJacobian{T}) where T
    jst = J.structure
    st = jst.schur
    nr, ne = st.nr, sum(Int, st.e_dims)
    blocks = Matrix{T}[]
    for t in 1:jst.ntables
        m = t <= st.neblocks ? jst.e_rows[t] : jst.free_rows[t - st.neblocks]
        A = zeros(T, m, nr + ne)
        for u in jst.table_ptr[t]:jst.table_ptr[t+1]-1
            r = jst.entry_retained[u]
            A[:, st.boffsets_r[r] .+ (0:Int(st.r_dims[r])-1)] .= coupling_block(J, u)
        end
        if t <= st.neblocks
            A[:, nr .+ st.boffsets_e[t] .+ (0:Int(st.e_dims[t])-1)] .= eliminated_block(J, t)
        end
        push!(blocks, A)
    end
    return reduce(vcat, blocks)
end
qr_dense_r(J::SchurBlockJacobian) = reduce(vcat, [collect(BlockLinearSystems.table_residual_block(J, t)) for t in 1:J.structure.ntables])

relerr(a, b) = norm(Float64.(a) - Float64.(b)) / norm(Float64.(b))

# The Hessian side: H = J'J and b = J'r as a SchurBlockHessian (dense or sparse R), in element type T
function schur_hessian(p::QRTestProblem{T}, kind) where T
    J64 = Float64.(p.Jf)
    H = J64' * J64
    b = J64' * Float64.(p.rf)
    jst = p.J.structure
    rr = [(r2, r1) for (r1, r2) in free_block_pairs(jst)]   # lower-triangle block coordinates
    sp = SchurTestProblem(H, b, p.r_dims, p.e_dims, p.couplings, rr)
    Hs = kind === :dense ? dense_schur_hessian(sp) : sparse_schur_hessian(sp)
    R = Hs.R isa Matrix ? T.(Hs.R) :
        SparseBlockLinearSystem{T}(T.(Hs.R.data), Hs.R.indices, Hs.R.indicestransposed, Hs.R.rowblocksizes, Hs.R.columnblocksizes, Hs.R.m, Hs.R.n)
    return SchurBlockHessian(Hs.structure, R, T.(Hs.E_data), T.(Hs.C_data)), T.(b)
end

# Existing Schur reduction: (S, g_S, ws, ok, pattern) with S including the retained damping
function schur_reduced(Hs::SchurBlockHessian{T}, b, sc, lambda) where T
    st = Hs.structure
    nr, n = st.nr, length(b)
    ws = SchurReductionWorkspace{T}(st, 1)
    g = zeros(T, nr)
    lambda = T(lambda)
    b_r, b_e, sc_r, sc_e = b[1:nr], b[nr+1:n], sc[1:nr], sc[nr+1:n]
    if Hs.R isa Matrix
        S = zeros(T, nr, nr)
        reset_schur_dense!(S, Hs.R, sc_r, lambda, ws)
        ok = reduce_schur_dense!(S, g, b_r, b_e, sc_e, lambda, st, Hs.E_data, Hs.C_data, ws)
        return (S, g, ws, ok, nothing)
    else
        pattern = build_schur_pattern(st, T; extra_blocks=retained_block_pairs(Hs.R))
        S = copy(pattern.S_sparse)
        reset_schur_sparse!(S.nzval, Hs.R.data, build_r_to_s_map(Hs.R, pattern, st.boffsets_r), pattern.diag_inds, sc_r, lambda, ws)
        ok = reduce_schur_sparse!(S, g, b_r, b_e, sc_e, lambda, st, Hs.E_data, Hs.C_data,
                                  pattern.block_colptr, pattern.block_rowval, pattern.block_row_offsets, pattern.col_heights, ws)
        return (S, g, ws, ok, pattern)
    end
end

# The QR route to the full solution: eliminate, dense QR of [J̃; √(λ s_r)], back-substitute
function qr_solve(p::QRTestProblem{T}, sc, lambda) where T
    J = p.J
    jst = J.structure
    nr = jst.schur.nr
    n = size(p.Jf, 2)
    ws = QRReductionWorkspace(J)
    reduce_qr!(ws, J, sc[nr+1:n], T(lambda)) || return (zeros(T, n), false, ws)
    Jt = zeros(T, reduced_nrows(jst), nr)
    rt = zeros(T, reduced_nrows(jst))
    reduced_jacobian!(Jt, rt, ws)
    A = [Jt; Diagonal(sqrt.(T(lambda) .* sc[1:nr]))]
    x_r = qr(A) \ [rt; zeros(T, nr)]
    x_e = zeros(T, n - nr)
    backsubstitute!(x_e, x_r, ws)
    return ([x_r; x_e], true, ws)
end

# Dense reference: min ‖J x - r‖² + λ Σ s_i x_i², by QR in Float64
function dense_ls(p::QRTestProblem, sc, lambda)
    J64 = Float64.(p.Jf)
    n = size(J64, 2)
    A = [J64; Diagonal(sqrt.(lambda .* Float64.(sc)))]
    return qr(A) \ [Float64.(p.rf); zeros(n)]
end

qr_rtol(::Type{Float64}) = 1e-9
qr_rtol(::Type{Float32}) = 2e-3

# The problems of R4: those of qr_problems (no free group, blocks above D_STATIC, free groups), the tutorial's
# structure, an under-determined eliminated block with a free group, and no eliminated block at all (with a retained
# block that no row touches, so a column of J̃ stores nothing)
function sparse_reduced_problems(::Type{T}; nchunks=1, min_work=BlockLinearSystems.DEFAULT_MIN_WORK) where T
    rng = Random.Xoshiro(40)
    probs = qr_problems(T; nchunks, min_work)
    push!(probs, QRTestProblem("tutorial", rng, T, [6, 6, 6], [3, 3, 3, 3, 2, 1],
                               [[1, 2], [1, 2, 3], [2, 3], [1, 3], [1, 2], [3]], [8, 10, 6, 6, 4, 3];
                               free_rows=[12], free_retained=[[1, 3]], nchunks, min_work))
    push!(probs, QRTestProblem("under-determined", rng, T, [6, 6], [12, 12, 12], [[1, 2], [1], [2]], [28, 15, 10];
                               free_rows=[8], free_retained=[[1, 2]], nchunks, min_work))
    push!(probs, QRTestProblem("no eliminated block", rng, T, [3, 2, 4], Int[], Vector{Int}[], Int[];
                               free_rows=[5, 4], free_retained=[[1, 2], [2]], nchunks, min_work))
    return probs
end

end # module QRFixtures
