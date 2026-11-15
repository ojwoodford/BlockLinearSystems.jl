using Random, LinearAlgebra, SparseArrays, Test, BlockLinearSystems

# Every kernel that threads allocates nothing on a warm call, threaded or serial (R2, R3), apart from the bounded
# allocation of a threaded call on Julia 1.10 (`threaded_allocs_ok`, runtests.jl). Run on 1 and on 4 threads:
# the threshold is forced to 0 and the structures have one chunk per thread, so on 4 threads every kernel threads.
# The problems have eliminated blocks of 9 and 12, above D_STATIC, as well as static ones.

# Measured inside functions: @allocated at a testset's top level counts spurious boxing on Julia 1.10
alloc_reduce_qr(ws, J, sc, lambda) = @allocated reduce_qr!(ws, J, sc, lambda)
alloc_jtr(g, src) = @allocated jtr!(g, src)
alloc_jtj_diag(d, src) = @allocated jtj_diag!(d, src)
alloc_jtj_quadform(src, v) = @allocated jtj_quadform(src, v)
alloc_reduced_mul(y, ws, v) = @allocated reduced_mul!(y, ws, v)
alloc_reduced_tmul(g, ws, w) = @allocated reduced_tmul!(g, ws, w)
alloc_reduced_residual(rt, ws) = @allocated reduced_residual!(rt, ws)
alloc_reduced_jacobian(Jt, rt, ws) = @allocated reduced_jacobian!(Jt, rt, ws)
alloc_reduced_normal_dense(S, g, ws) = @allocated reduced_normal_dense!(S, g, ws)
alloc_reduced_normal_sparse(S, g, ws, pat) = @allocated reduced_normal_sparse!(S, g, ws, pat)
alloc_reduced_diagonal_blocks(D, ws) = @allocated reduced_diagonal_blocks!(D, ws)
alloc_backsubstitute_qr(x_e, x_r, ws) = @allocated backsubstitute!(x_e, x_r, ws)
alloc_reset_schur_dense(S, R, sc_r, lambda, ws) = @allocated reset_schur_dense!(S, R, sc_r, lambda, ws)
alloc_reset_schur_sparse(Sv, Rv, map, di, sc_r, lambda, ws) = @allocated reset_schur_sparse!(Sv, Rv, map, di, sc_r, lambda, ws)
alloc_reduce_schur_dense(S, g, b_r, b_e, sc_e, lambda, st, H, ws) =
    @allocated reduce_schur_dense!(S, g, b_r, b_e, sc_e, lambda, st, H.E_data, H.C_data, ws)
alloc_reduce_schur_sparse(S, g, b_r, b_e, sc_e, lambda, st, H, pat, ws) =
    @allocated reduce_schur_sparse!(S, g, b_r, b_e, sc_e, lambda, st, H.E_data, H.C_data,
                                    pat.block_colptr, pat.block_rowval, pat.block_row_offsets, pat.col_heights, ws)
alloc_backsubstitute_schur(x_e, x_r, st, H, ws) = @allocated backsubstitute!(x_e, x_r, st, H.C_data, ws)

# Warm the call, then the least of three measurements
warm_allocs(f, args...) = (f(args...); minimum(f(args...) for _ in 1:3))

with_min_work(f, minwork) = begin
    old = BlockLinearSystems.SCHUR_THREAD_MIN_WORK[]
    try
        BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = minwork
        f()
    finally
        BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = old
    end
end

@testset "Threaded QR-elimination kernels allocate nothing (R2)" begin
    nt = Threads.nthreads()
    with_min_work(0) do
        for T in (Float64, Float32), p in qr_problems(T; nchunks=nt)[1:3]
            rng = Random.Xoshiro(21)
            J = p.J
            jst = J.structure
            nr, n = jst.schur.nr, size(p.Jf, 2)
            m = reduced_nrows(jst)
            @test BlockLinearSystems.qr_threaded(jst, nt) == (nt > 1)
            sc_e = qr_scaling(p, rng)[nr+1:n]
            ws = QRReductionWorkspace(J)
            @test reduce_qr!(ws, J, sc_e, T(0.1))
            v, w, vf = randn(rng, T, nr), randn(rng, T, m), randn(rng, T, n)
            pat = build_schur_pattern(jst.schur, T; extra_blocks=free_block_pairs(jst))
            Ssp = copy(pat.S_sparse)
            @testset "$(p.name), $T, $nt threads" begin
                @test threaded_allocs_ok(warm_allocs(alloc_reduce_qr, ws, J, sc_e, T(0.3)))
                for src in (J, ws)
                    @test threaded_allocs_ok(warm_allocs(alloc_jtr, zeros(T, n), src))
                    @test threaded_allocs_ok(warm_allocs(alloc_jtj_diag, zeros(T, n), src))
                    @test threaded_allocs_ok(warm_allocs(alloc_jtj_quadform, src, vf))
                end
                @test threaded_allocs_ok(warm_allocs(alloc_reduced_mul, zeros(T, m), ws, v))
                @test threaded_allocs_ok(warm_allocs(alloc_reduced_tmul, zeros(T, nr), ws, w))
                @test threaded_allocs_ok(warm_allocs(alloc_reduced_residual, zeros(T, m), ws))
                @test threaded_allocs_ok(warm_allocs(alloc_reduced_jacobian, zeros(T, m, nr), zeros(T, m), ws))
                @test threaded_allocs_ok(warm_allocs(alloc_reduced_normal_dense, zeros(T, nr, nr), zeros(T, nr), ws))
                @test threaded_allocs_ok(warm_allocs(alloc_reduced_normal_sparse, Ssp, zeros(T, nr), ws, pat))
                @test threaded_allocs_ok(warm_allocs(alloc_reduced_diagonal_blocks, zeros(T, jst.doffsets[end] - 1), ws))
                @test threaded_allocs_ok(warm_allocs(alloc_backsubstitute_qr, zeros(T, n - nr), v, ws))
            end
        end
    end
end

@testset "Threaded Schur kernels allocate nothing, blocks above D_STATIC included (R2, R3)" begin
    nt = Threads.nthreads()
    with_min_work(0) do
        T = Float64
        for p in (mixed_dims_problem(), standard_problem())
            n = length(p.b)
            b, sc = p.b, abs.(diag(p.H))
            lambda = 0.3
            for make in (dense_schur_matrix, sparse_schur_matrix)
                H = make(p; nchunks=nt)
                st = H.structure
                nr = st.nr
                @test nt == 1 || BlockLinearSystems.use_threads(nt, length(st.entry_retained))
                b_r, b_e = b[1:nr], b[nr+1:n]
                sc_r, sc_e = sc[1:nr], sc[nr+1:n]
                ws = SchurReductionWorkspace{T}(st, nt)
                g = zeros(T, nr)
                x_e, x_r = zeros(T, n - nr), randn(Random.Xoshiro(5), T, nr)
                kind = H.R isa Matrix ? "dense" : "sparse"
                @testset "e_dims $(sort(unique(st.e_dims))), $kind, $nt threads" begin
                    if H.R isa Matrix
                        S = zeros(T, nr, nr)
                        @test threaded_allocs_ok(warm_allocs(alloc_reset_schur_dense, S, H.R, sc_r, lambda, ws))
                        @test threaded_allocs_ok(warm_allocs(alloc_reduce_schur_dense, S, g, b_r, b_e, sc_e, lambda, st, H, ws))
                    else
                        pat = build_schur_pattern(st, T; extra_blocks=retained_block_pairs(H.R))
                        S = pat.S_sparse
                        rmap = build_r_to_s_map(H.R, pat, st.boffsets_r)
                        @test threaded_allocs_ok(warm_allocs(alloc_reset_schur_sparse, S.nzval, H.R.data, rmap, pat.diag_inds, sc_r, lambda, ws))
                        @test threaded_allocs_ok(warm_allocs(alloc_reduce_schur_sparse, S, g, b_r, b_e, sc_e, lambda, st, H, pat, ws))
                    end
                    @test threaded_allocs_ok(warm_allocs(alloc_backsubstitute_schur, x_e, x_r, st, H, ws))
                end
            end
        end
    end
end

# In a fresh session on 2 threads: without Polyester the kernels run serially (and give the serial result); once it is
# loaded, the very first threaded call of the session works and gives the same result.
const FRESH_SESSION = raw"""
using BlockLinearSystems, Random
const BLS = BlockLinearSystems
rng = Random.Xoshiro(3)
jst = SchurJacobianStructure(fill(6, 8), [3, 3, 9, 12, 3, 3], [[1, 2], [2, 3], [4], [5, 6], [7], [8, 1]], fill(14, 6); nchunks=2)
J = SchurBlockJacobian{Float64, Matrix{Float64}}(jst)
J.data .= randn(rng, length(J.data))
sc = ones(sum(Int, jst.schur.e_dims))
function factored(minwork)
    BLS.SCHUR_THREAD_MIN_WORK[] = minwork
    ws = QRReductionWorkspace(J)
    reduce_qr!(ws, J, sc, 0.1) || error("reduce_qr! failed")
    return copy(ws.factored)
end
Base.get_extension(BLS, :PolyesterExt) === nothing || error("PolyesterExt loaded without Polyester")
serial = factored(typemax(Int))
factored(0) == serial || error("without Polyester: differs from serial")
@eval using Polyester
Base.get_extension(BLS, :PolyesterExt) === nothing && error("PolyesterExt not loaded")
factored(0) == serial || error("first threaded call: differs from serial")
print("ok")
"""

@testset "Threading needs Polyester; the first threaded call of a session works (R2)" begin
    cmd = `$(Base.julia_cmd()) --startup-file=no --threads=2 --project=$(Base.active_project()) -e $FRESH_SESSION`
    @test readchomp(cmd) == "ok"
end

