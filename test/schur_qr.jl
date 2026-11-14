using Random, LinearAlgebra, SparseArrays, StaticArrays, Static, Test, BlockLinearSystems

# QR elimination on the Jacobian path, tested against the Schur reduction (schur_problems.jl must be included
# first: its SchurTestProblem and dense/sparse_schur_matrix build the Hessian side) and against dense least squares.
# The test problems come from schur_jacobian.jl.

# The Hessian side: H = J'J and b = J'r as a SchurBlockMatrix (dense or sparse R), in element type T
function schur_matrix(p::QRTestProblem{T}, kind) where T
    J64 = Float64.(p.Jf)
    H = J64' * J64
    b = J64' * Float64.(p.rf)
    jst = p.J.structure
    rr = [(r2, r1) for (r1, r2) in free_block_pairs(jst)]   # lower-triangle block coordinates
    sp = SchurTestProblem(H, b, p.r_dims, p.e_dims, p.couplings, rr)
    Hs = kind === :dense ? dense_schur_matrix(sp) : sparse_schur_matrix(sp)
    R = Hs.R isa Matrix ? T.(Hs.R) :
        SparseBlockLinearSystem{T}(T.(Hs.R.data), Hs.R.indices, Hs.R.indicestransposed, Hs.R.rowblocksizes, Hs.R.columnblocksizes, Hs.R.m, Hs.R.n)
    return SchurBlockMatrix(Hs.structure, R, T.(Hs.E_data), T.(Hs.C_data)), T.(b)
end

# Existing Schur reduction: (S, g_S, ws, ok, pattern) with S including the retained damping
function schur_reduced(Hs::SchurBlockMatrix{T}, b, sc, lambda) where T
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

@testset "QR elimination equals the Schur reduction of J'J (E4)" begin
    for T in (Float64, Float32)
        rng = Random.Xoshiro(7)
        for p in qr_problems(T), kind in (:dense, :sparse), lambda in (0.0, 0.3, 4.0)
            @testset "$T $(p.name) $kind λ=$lambda" begin
                jst = p.J.structure
                st = jst.schur
                nr, n = st.nr, size(p.Jf, 2)
                sc = qr_scaling(p, rng)
                Hs, b = schur_matrix(p, kind)
                S_ref, g_ref, sws, ok_ref, pattern = schur_reduced(Hs, b, sc, lambda)
                @test ok_ref

                ws = QRReductionWorkspace(p.J)
                @test reduce_qr!(ws, p.J, sc[nr+1:n], T(lambda))
                g = zeros(T, nr)
                if kind === :dense
                    S = fill(T(NaN), nr, nr)                       # overwritten, including the lower triangle
                    reduced_normal_dense!(S, g, ws)
                    @test all(iszero, tril(S, -1))
                    S[diagind(S)] .+= T(lambda) .* sc[1:nr]
                    @test relerr(triu(S), triu(S_ref)) <= qr_rtol(T)
                else
                    qpattern = build_schur_pattern(st, T; extra_blocks=free_block_pairs(jst))
                    @test qpattern.block_pattern == pattern.block_pattern
                    S = copy(qpattern.S_sparse)
                    S.nzval .= T(NaN)
                    reduced_normal_sparse!(S, g, ws, qpattern)
                    S.nzval[qpattern.diag_inds] .+= T(lambda) .* sc[1:nr]
                    @test S.colptr == S_ref.colptr && S.rowval == S_ref.rowval
                    @test relerr(S.nzval, S_ref.nzval) <= qr_rtol(T)
                end
                @test relerr(g, g_ref) <= qr_rtol(T)

                # Back-substitution from the same retained values
                x_r = randn(rng, T, nr)
                x_e = zeros(T, n - nr)
                backsubstitute!(x_e, x_r, ws)
                x_e_ref = zeros(T, n - nr)
                backsubstitute!(x_e_ref, x_r, st, Hs.C_data, sws.E_inv_data, sws.z_data)
                @test relerr(x_e, x_e_ref) <= qr_rtol(T)
            end
        end
    end
end

@testset "QR elimination equals dense least squares (E4)" begin
    for T in (Float64, Float32)
        rng = Random.Xoshiro(8)
        for p in qr_problems(T), lambda in (0.0, 0.3, 4.0)
            n = size(p.Jf, 2)
            sc = lambda == 0 ? ones(T, n) : qr_scaling(p, rng)
            x, ok, _ = qr_solve(p, sc, lambda)
            @test ok
            x_ref = dense_ls(p, sc, lambda)
            A = [Float64.(p.Jf); Diagonal(sqrt.(lambda .* Float64.(sc)))]
            # Tolerance proportional to cond(J), not its square
            @test relerr(x, x_ref) <= 50 * cond(A) * eps(T)
        end
    end
end

# Near-dependence between a retained and an eliminated column: retained block 3 is coupled only to eliminated
# block 5, and its first column is that block's first column plus 1e-8 noise. cond(J) is then about 1e8: S has
# condition about 1e16, and the Schur route loses everything that the QR route keeps.
function illconditioned_problem(::Type{T}=Float64; delta=1e-8, seed=4) where T
    rng = Random.Xoshiro(seed)
    p = QRTestProblem("ill-conditioned", rng, T, [6, 6, 3], [3, 3, 3, 3, 3],
                      [[1, 2], [1, 2], [1, 2], [2], [1, 3]], [8, 8, 8, 6, 9]; free_rows=[10], free_retained=[[1, 2]])
    J = p.J
    jst = J.structure
    u = coupling_entry(jst, 5, 3)
    B = coupling_block(J, u)
    E = eliminated_block(J, 5)
    B[:, 1] .= E[:, 1] .+ T(delta) .* randn(rng, T, size(E, 1))
    nr = jst.schur.nr
    rows = sum(jst.e_rows[1:4])+1:sum(jst.e_rows[1:5])
    p.Jf[rows, jst.schur.boffsets_r[3]] .= B[:, 1]
    # Consistent right-hand side: zero residual at the solution, so the forward error scales with cond(J)
    x_true = randn(rng, T, size(p.Jf, 2))
    rf = p.Jf * x_true
    row = 0
    for t in 1:jst.ntables
        m = t <= jst.schur.neblocks ? jst.e_rows[t] : jst.free_rows[t - jst.schur.neblocks]
        residual_block(J, t) .= rf[row+1:row+m]
        row += m
    end
    p.rf .= rf
    return p, x_true
end

@testset "Ill-conditioned problem: QR keeps the accuracy Schur loses (E4)" begin
    p, x_true = illconditioned_problem()
    n = size(p.Jf, 2)
    kJ = cond(p.Jf)
    @test 1e7 <= kJ <= 1e9
    x_qr, ok, _ = qr_solve(p, ones(n), 0.0)
    @test ok
    err_qr = relerr(x_qr, x_true)
    # The Schur route: the reduction itself succeeds (every E_e is well conditioned), but S has lost its
    # positive definiteness to rounding, so Cholesky fails; solve it with Bunch-Kaufman instead to get a number
    Hs, b = schur_matrix(p, :dense)
    nr = Hs.structure.nr
    S, g, sws, ok_s, _ = schur_reduced(Hs, b, ones(n), 0.0)
    @test ok_s
    @test !issuccess(cholesky(Symmetric(S, :U); check=false))
    x_s = zeros(n)
    x_s[1:nr] .= bunchkaufman(Symmetric(S, :U)) \ g
    backsubstitute!(view(x_s, nr+1:n), x_s[1:nr], Hs.structure, Hs.C_data, sws.E_inv_data, sws.z_data)
    err_schur = relerr(x_s, x_true)
    @info "Ill-conditioned least squares" cond_J=kJ err_qr err_schur
    @test err_qr <= 10 * kJ * eps()
    @test err_schur >= 1e3 * err_qr
end

@testset "Orthogonal invariance: raw and factored tables give the same full-problem quantities (E4)" begin
    for T in (Float64, Float32)
        rng = Random.Xoshiro(9)
        tol = T === Float64 ? 1e-11 : 1e-4
        for p in qr_problems(T), lambda in (0.0, 0.7)
            J = p.J
            nr = J.structure.schur.nr
            n = size(p.Jf, 2)
            sc = qr_scaling(p, rng)
            ws = QRReductionWorkspace(J)
            @test reduce_qr!(ws, J, sc[nr+1:n], lambda)
            v = randn(rng, T, n)
            g_raw, g_fac = jtr!(zeros(T, n), J), jtr!(zeros(T, n), ws)
            d_raw, d_fac = jtj_diag!(zeros(T, n), J), jtj_diag!(zeros(T, n), ws)
            q_raw, q_fac = fast_bAb(J, v), fast_bAb(ws, v)
            # The raw tables against the dense problem
            @test relerr(g_raw, p.Jf' * p.rf) <= tol
            @test relerr(d_raw, vec(sum(abs2, p.Jf; dims=1))) <= tol
            @test abs(q_raw - sum(abs2, p.Jf * v)) <= tol * q_raw
            # The factored tables hold the damped Jacobian [J; √(λ s_e)] in the eliminated columns
            damp = [zeros(T, nr); T(lambda) .* sc[nr+1:n]]
            @test relerr(g_fac, g_raw) <= tol
            @test relerr(d_fac, d_raw .+ damp) <= tol
            @test abs(q_fac - (q_raw + dot(damp, v .^ 2))) <= tol * q_fac
            if lambda == 0
                @test relerr(d_fac, d_raw) <= tol
            end
            @test fast_bAb(J, Float32.(v)) isa promote_type(T, Float32)
        end
    end
end

@testset "Products with the reduced Jacobian (E4)" begin
    for T in (Float64, Float32)
        rng = Random.Xoshiro(10)
        tol = T === Float64 ? 1e-12 : 1e-5
        for p in qr_problems(T), lambda in (0.0, 0.5)
            J = p.J
            jst = J.structure
            nr = jst.schur.nr
            n = size(p.Jf, 2)
            ws = QRReductionWorkspace(J)
            @test reduce_qr!(ws, J, qr_scaling(p, rng)[nr+1:n], lambda)
            m = reduced_nrows(jst)
            @test m == sum(jst.e_rows) + sum(jst.free_rows; init=0)   # m_e rows per table, whatever λ is
            Jt, rt = zeros(T, m, nr), zeros(T, m)
            reduced_jacobian!(Jt, rt, ws)
            @test reduced_residual!(zeros(T, m), ws) == rt
            v, w = randn(rng, T, nr), randn(rng, T, m)
            y = reduced_mul!(fill(T(NaN), m), ws, v)
            g = reduced_tmul!(fill(T(NaN), nr), ws, w)
            @test relerr(y, Jt * v) <= tol
            @test relerr(g, Jt' * w) <= tol
            @test abs(dot(y, w) - dot(v, g)) <= tol * norm(y) * norm(w)
            # Diagonal blocks of J̃ᵀJ̃
            D = reduced_diagonal_blocks!(fill(T(NaN), jst.doffsets[end] - 1), ws)
            JtJ = Jt' * Jt
            for r in 1:jst.schur.nrblocks
                dr = Int(jst.schur.r_dims[r])
                rr = jst.schur.boffsets_r[r] .+ (0:dr-1)
                @test relerr(reshape(D[jst.doffsets[r]:jst.doffsets[r+1]-1], dr, dr), JtJ[rr, rr]) <= tol
            end
            @test_throws DimensionMismatch reduced_mul!(zeros(T, m + 1), ws, v)
            @test_throws DimensionMismatch reduced_tmul!(zeros(T, nr), ws, zeros(T, m - 1))
            @test_throws DimensionMismatch reduced_jacobian!(zeros(T, m, nr + 1), rt, ws)
        end
    end
end

qr_mul_allocs(y, ws, v) = @allocated reduced_mul!(y, ws, v)
qr_tmul_allocs(g, ws, w) = @allocated reduced_tmul!(g, ws, w)
qr_backsub_allocs(x_e, x_r, ws) = @allocated backsubstitute!(x_e, x_r, ws)
qr_normal_allocs(S, g, ws) = @allocated reduced_normal_dense!(S, g, ws)
qr_snormal_allocs(S, g, ws, pat) = @allocated reduced_normal_sparse!(S, g, ws, pat)
qr_misc_allocs(rt, D, d, gg, ws, J, v) = @allocated begin
    reduced_residual!(rt, ws); reduced_diagonal_blocks!(D, ws); jtj_diag!(d, J); jtr!(gg, ws); fast_bAb(ws, v)
end

@testset "Kernels of the reduced problem: no allocation, type stable (E4)" begin
    for T in (Float64, Float32), p in qr_problems(T)
        rng = Random.Xoshiro(11)
        J = p.J
        jst = J.structure
        nr, n = jst.schur.nr, size(p.Jf, 2)
        m = reduced_nrows(jst)
        sc_e = qr_scaling(p, rng)[nr+1:n]
        ws = QRReductionWorkspace(J)
        @test reduce_qr!(ws, J, sc_e, T(0.1))
        v, w = randn(rng, T, nr), randn(rng, T, m)
        y, g = zeros(T, m), zeros(T, nr)
        x_e = zeros(T, n - nr)
        S, gS = zeros(T, nr, nr), zeros(T, nr)
        pat = build_schur_pattern(jst.schur, T; extra_blocks=free_block_pairs(jst))
        Ssp = copy(pat.S_sparse)
        rt, D, d, gg = zeros(T, m), zeros(T, jst.doffsets[end] - 1), zeros(T, n), zeros(T, n)
        vf = randn(rng, T, n)
        # Serial path: nothing at all. Threaded path: only Polyester's per-launch argument reference, a few
        # kilobytes at most (more on Julia 1.10), independent of the problem size (the existing Schur kernels do the same).
        old_minwork = BlockLinearSystems.SCHUR_THREAD_MIN_WORK[]
        try
            for (minwork, bound) in ((typemax(Int), 0), (0, 4096))
                BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = minwork
                for _ in 1:2
                    @test qr_mul_allocs(y, ws, v) <= bound
                    @test qr_tmul_allocs(g, ws, w) <= bound
                    @test qr_backsub_allocs(x_e, v, ws) <= bound
                    @test qr_normal_allocs(S, gS, ws) <= bound
                    @test qr_snormal_allocs(Ssp, gS, ws, pat) <= bound
                    @test qr_misc_allocs(rt, D, d, gg, ws, J, vf) <= 5 * bound
                end
            end
        finally
            BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = old_minwork
        end
        # Type stability
        @inferred reduced_mul!(y, ws, v)
        @inferred reduced_tmul!(g, ws, w)
        @inferred backsubstitute!(x_e, v, ws)
        @inferred reduced_normal_dense!(S, gS, ws)
        @inferred reduced_normal_sparse!(Ssp, gS, ws, pat)
        @test @inferred(fast_bAb(ws, vf)) isa T
        @test @inferred(fast_bAb(J, vf)) isa T
        @inferred jtr!(gg, J)
        @inferred jtj_diag!(d, ws)
    end
end

@testset "Threading: identical results for any chunking, serial below the work threshold (E4)" begin
    old = BlockLinearSystems.SCHUR_THREAD_MIN_WORK[]
    try
        for T in (Float64, Float32)
            serial = qr_problems(T; nchunks=1)
            for (k, p1) in enumerate(serial)
                p4 = qr_problems(T; nchunks=4)[k]
                @test p4.J.data == p1.J.data
                jst1, jst4 = p1.J.structure, p4.J.structure
                @test length(jst4.col_chunks) == 4
                @test reduce(vcat, collect.(jst4.col_chunks)) == collect(1:jst4.schur.nrblocks)
                nr, n = jst1.schur.nr, size(p1.Jf, 2)
                m = reduced_nrows(jst1)
                rng = Random.Xoshiro(12)
                sc_e = qr_scaling(p1, rng)[nr+1:n]
                v, w = randn(rng, T, nr), randn(rng, T, m)
                results = map(((p1, 1 << 40), (p4, 0))) do (p, minwork)
                    BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = minwork
                    jst = p.J.structure
                    @test BlockLinearSystems.qr_threaded(jst, 4) == (minwork == 0 && Threads.nthreads() > 1)
                    ws = QRReductionWorkspace(p.J)
                    ok = reduce_qr!(ws, p.J, sc_e, 0.2)
                    S, gS = zeros(T, nr, nr), zeros(T, nr)
                    reduced_normal_dense!(S, gS, ws)
                    x_e = zeros(T, n - nr)
                    backsubstitute!(x_e, v, ws)
                    (ok, copy(ws.factored), reduced_mul!(zeros(T, m), ws, v), reduced_tmul!(zeros(T, nr), ws, w), S, gS, x_e,
                     jtr!(zeros(T, n), p.J), jtj_diag!(zeros(T, n), ws), reduced_diagonal_blocks!(zeros(T, jst.doffsets[end] - 1), ws))
                end
                @test results[1][1] && results[2][1]
                @test all(results[1][i] == results[2][i] for i in 2:length(results[1]))
            end
        end
        # Below the threshold the serial path is taken
        BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = old
        p = qr_problems(Float64; nchunks=4)[end]
        @test BlockLinearSystems.table_length(p.J.structure) < old
        @test !BlockLinearSystems.qr_threaded(p.J.structure, 4)
    finally
        BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = old
    end
end

@testset "Under-determined block with damping: Schur and dense least squares agree (E4)" begin
    for T in (Float64, Float32), de in (3, 12)
        rng = Random.Xoshiro(30 + de)
        p = QRTestProblem("under-determined", rng, T, [6, 6], [de, de, de], [[1, 2], [1], [2]], [2de + 4, de + 3, de - 2];
                          free_rows=[8], free_retained=[[1, 2]])
        n = size(p.Jf, 2)
        sc = qr_scaling(p, rng)
        x, ok, ws = qr_solve(p, sc, 0.2)
        @test ok
        A = [Float64.(p.Jf); Diagonal(sqrt.(0.2 .* Float64.(sc)))]
        @test relerr(x, dense_ls(p, sc, 0.2)) <= 50 * cond(A) * eps(T)
        Hs, b = schur_matrix(p, :dense)
        S_ref, g_ref, _, ok_ref, _ = schur_reduced(Hs, b, sc, 0.2)
        S, g = zeros(T, 12, 12), zeros(T, 12)
        reduced_normal_dense!(S, g, ws)
        S[diagind(S)] .+= T(0.2) .* sc[1:12]
        @test ok_ref && relerr(triu(S), triu(S_ref)) <= qr_rtol(T) && relerr(g, g_ref) <= qr_rtol(T)
        @test !reduce_qr!(QRReductionWorkspace(p.J), p.J, sc[13:end], 0)     # and it fails undamped
    end
end
