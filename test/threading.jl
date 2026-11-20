using Random, LinearAlgebra, SparseArrays, Test, BlockLinearSystems
using BlockLinearSystems: SchurReductionWorkspace, build_r_to_s_map, retained_block_pairs, QRReductionWorkspace, reduce_qr!, reduced_nrows, free_block_pairs, build_reduced_jacobian_pattern   # internal: the kernels, withdrawn from the exports by EL6
using Polyester   # BlockLinearSystems' kernels thread only with Polyester loaded
include("fixtures/load.jl")
using Main.SchurFixtures, Main.QRFixtures, Main.AllocFixtures

# Every kernel that threads allocates nothing on a warm call, threaded or serial (R2, R3), apart from the bounded
# allocation of a threaded call on Julia 1.10 (`threaded_allocs_ok`, fixtures/AllocFixtures.jl). Run on 1 and on 4 threads:
# the structures have one chunk per thread and `min_work = 0`, so on 4 threads every kernel threads.
# The problems have eliminated blocks of 9 and 12, above D_STATIC, as well as static ones.

@testset "Threaded QR-elimination kernels allocate nothing (R2)" begin
    nt = Threads.nthreads()
    begin
        for T in (Float64, Float32), p in qr_problems(T; nchunks=nt, min_work=0)[1:3]
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
                @test threaded_allocs_ok(warm_allocs(alloc_reduced_jacobian, build_reduced_jacobian_pattern(jst, T), zeros(T, m), ws))
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
    begin
        T = Float64
        for p in (mixed_dims_problem(), standard_problem())
            n = length(p.b)
            b, sc = p.b, abs.(diag(p.H))
            lambda = 0.3
            for make in (dense_schur_hessian, sparse_schur_hessian)
                H = make(p; nchunks=nt, min_work=0)
                st = H.structure
                nr = st.nr
                @test nt == 1 || BlockLinearSystems.use_threads(nt, length(st.entry_retained), st.min_work)
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
