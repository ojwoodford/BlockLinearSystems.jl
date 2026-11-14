using Random, LinearAlgebra, SparseArrays, StaticArrays, Static, Test, BlockLinearSystems

# The QR elimination kernel. Test problems come from schur_jacobian.jl.

# U, B and q of table e (the top d_e rows), and the rows below U in the eliminated columns
function qr_factors(ws::QRReductionWorkspace, e)
    jst = ws.structure
    M, C, de = jst.table_nrows[e], jst.table_ncols[e], jst.table_top[e]
    Tm = reshape(ws.factored[jst.table_offsets[e]:jst.table_offsets[e+1]-1], M, C)
    return (U = Tm[1:de, C-de:C-1], B = Tm[1:de, 1:C-de-1], q = Tm[1:de, C], below = Tm[de+1:M, C-de:C-1])
end

@testset "QR kernel: UᵀU = E + λ diag(s_e), UᵀB = C, Uᵀq = b_e (E2)" begin
    for T in (Float64, Float32)
        rng = Random.Xoshiro(21)
        tol = T === Float64 ? 1e-11 : 1e-4
        for p in qr_problems(T), lambda in (0.0, 0.4)
            J = p.J
            jst = J.structure
            st = jst.schur
            nr, n = st.nr, size(p.Jf, 2)
            sc_e = qr_scaling(p, rng)[nr+1:n]
            ws = QRReductionWorkspace(J)
            @test reduce_qr!(ws, J, sc_e, lambda)
            @test ws.failed_block[] == 0
            row = 0
            for e in 1:st.neblocks
                de, m = Int(st.e_dims[e]), jst.e_rows[e]
                rows = row+1:row+m
                ecols = nr .+ st.boffsets_e[e] .+ (0:de-1)
                rcols = reduce(vcat, [st.boffsets_r[r] .+ (0:Int(st.r_dims[r])-1) for r in jst.entry_retained[jst.table_ptr[e]:jst.table_ptr[e+1]-1]]; init=Int[])
                Je, Jr, re = Float64.(p.Jf[rows, ecols]), Float64.(p.Jf[rows, rcols]), Float64.(p.rf[rows])
                f = qr_factors(ws, e)
                U = Float64.(f.U)
                @test istriu(f.U) && all(iszero, f.below)
                E = Je' * Je + lambda * Diagonal(Float64.(sc_e[st.boffsets_e[e] .+ (0:de-1)]))
                @test norm(U' * U - E) <= tol * norm(E)
                @test norm(U' * Float64.(f.B) - Je' * Jr) <= tol * norm(Je) * norm(Jr)
                @test norm(U' * Float64.(f.q) - Je' * re) <= tol * norm(Je) * norm(re)
                row += m
            end
            # Free tables are copied unchanged
            o = jst.table_offsets[st.neblocks + 1]
            @test ws.factored[o:end] == J.data[o:end]
        end
    end
end

@testset "Under-determined, rank-deficient and non-finite blocks fail undamped, naming the block (E4)" begin
    for T in (Float64, Float32), de in (1, 2, 6, 12)
        rng = Random.Xoshiro(de)
        r_dims, e_dims, couplings = [6, 6], [de, de, de], [[1, 2], [1], [2]]
        cases = (
            ("under-determined", [2de + 2, de + 2, max(de - 1, 1)], (J, e) -> nothing, de > 1),
            ("duplicated column", [2de + 2, de + 2, de + 3], (J, e) -> (E = eliminated_block(J, e); E[:, end] .= E[:, 1]), de > 1),
            ("zero block", [2de + 2, de + 2, de + 3], (J, e) -> (eliminated_block(J, e) .= 0), true),
        )
        for (name, e_rows, spoil!, applicable) in cases
            applicable || continue
            p = QRTestProblem(name, rng, T, r_dims, e_dims, couplings, e_rows)
            J = p.J
            spoil!(J, 3)
            n = size(p.Jf, 2)
            sc_e = ones(T, n - 12)
            ws = QRReductionWorkspace(J)
            @test !reduce_qr!(ws, J, sc_e, 0)
            @test ws.failed_block[] == 3
            @test reduce_qr!(ws, J, sc_e, 0.1)
            @test ws.failed_block[] == 0
            # Lowest failing block wins when several fail
            spoil!(J, 2)
            name == "under-determined" || (@test !reduce_qr!(ws, J, sc_e, 0); @test ws.failed_block[] == 2)
        end
        # A NaN or Inf anywhere in one table fails that block, damped or not
        for bad in (T(NaN), T(Inf))
            p = QRTestProblem("non-finite", rng, T, r_dims, e_dims, couplings, [2de + 2, de + 2, de + 3])
            coupling_block(p.J, coupling_entry(p.J.structure, 2, 1))[1, 1] = bad
            ws = QRReductionWorkspace(p.J)
            @test !reduce_qr!(ws, p.J, ones(T, 3de), 0.5)
            @test ws.failed_block[] == 2
        end
    end
end

@testset "reduce_qr! checks its arguments (E2)" begin
    p = qr_problems(Float64)[2]
    n_e = size(p.Jf, 2) - p.J.structure.schur.nr
    ws = QRReductionWorkspace(p.J)
    other = SchurBlockJacobian{Float64}(SchurJacobianStructure(p.J.structure.schur, p.J.structure.e_rows))
    @test_throws ArgumentError reduce_qr!(ws, other, ones(n_e), 0.1)
    @test_throws DimensionMismatch reduce_qr!(ws, p.J, ones(n_e + 1), 0.1)
    @test reduce_qr!(ws, p.J, ones(n_e), 1)              # any real λ
end

reduce_allocs(ws, J, sc, lambda) = @allocated reduce_qr!(ws, J, sc, lambda)

@testset "Re-reduction with a sequence of λ equals fresh reductions, without allocating (E2)" begin
    for T in (Float64, Float32), p in qr_problems(T)
        rng = Random.Xoshiro(11)
        J = p.J
        jst = J.structure
        nr, n = jst.schur.nr, size(p.Jf, 2)
        sc_e = qr_scaling(p, rng)[nr+1:n]
        raw = copy(J.data)
        ws = QRReductionWorkspace(J)
        reduce_qr!(ws, J, sc_e, T(0.1))                       # warm-up
        for lambda in T.((2.0, 0.0, 0.5, 1e-3))
            reduce_qr!(ws, J, sc_e, lambda)
            fresh = QRReductionWorkspace(J)
            @test reduce_qr!(fresh, J, sc_e, lambda)
            @test ws.factored == fresh.factored
        end
        @test J.data == raw                                   # the raw tables are kept
        # Serial path: no allocation at all. Threaded path: only Polyester's per-launch argument reference
        # (at most a few kilobytes, independent of the problem size), as for the existing Schur kernels.
        old = BlockLinearSystems.SCHUR_THREAD_MIN_WORK[]
        try
            for (minwork, bound) in ((typemax(Int), 0), (0, 4096))
                BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = minwork
                reduce_allocs(ws, J, sc_e, T(0.3))
                @test reduce_allocs(ws, J, sc_e, T(0.3)) <= bound
            end
        finally
            BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = old
        end
        @test @inferred(reduce_qr!(ws, J, sc_e, T(0.1))) isa Bool
    end
end

@testset "Threaded reduction equals serial; trial workspaces share only scratch (E2)" begin
    old = BlockLinearSystems.SCHUR_THREAD_MIN_WORK[]
    try
        for T in (Float64, Float32), k in 1:5
            p1, p4 = qr_problems(T; nchunks=1)[k], qr_problems(T; nchunks=4)[k]
            nr, n = p1.J.structure.schur.nr, size(p1.Jf, 2)
            sc_e = qr_scaling(p1, Random.Xoshiro(3))[nr+1:n]
            BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = typemax(Int)
            ws1 = QRReductionWorkspace(p1.J)
            @test reduce_qr!(ws1, p1.J, sc_e, 0.2)
            BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = 0
            ws4 = QRReductionWorkspace(p4.J)
            @test reduce_qr!(ws4, p4.J, sc_e, 0.2)
            @test ws4.factored == ws1.factored
        end
    finally
        BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = old
    end
    p = qr_problems(Float64)[2]
    ws = QRReductionWorkspace(p.J)
    tw = trial_workspace(ws)
    @test tw.structure === ws.structure && tw.scratch === ws.scratch
    @test tw.factored !== ws.factored && tw.failed_block !== ws.failed_block
    n_e = size(p.Jf, 2) - p.J.structure.schur.nr
    @test reduce_qr!(ws, p.J, ones(n_e), 0.1)
    before = copy(ws.factored)
    J2 = copy_structure(p.J)
    eliminated_block(J2, 1) .= 0                      # fails at block 1
    @test !reduce_qr!(tw, J2, ones(n_e), 0.0)
    @test tw.failed_block[] == 1 && ws.failed_block[] == 0 && ws.factored == before
end

@testset "Memory estimate matches the allocation (E2)" begin
    for T in (Float64, Float32), p in qr_problems(T)
        jst = p.J.structure
        ws = QRReductionWorkspace(p.J)
        @test schur_jacobian_bytes(jst, T) == sizeof(p.J.data) + sizeof(ws.factored) + sum(sizeof, ws.scratch)
        @test schur_jacobian_bytes(jst, T; workspace=false) == sizeof(p.J.data)
    end
    # The estimate needs only the structure, and grows with couplings times rows
    big = SchurJacobianStructure(fill(6, 10), fill(3, 1000), [collect(1:10) for _ in 1:1000], fill(40, 1000))
    @test schur_jacobian_bytes(big) == 2 * 8 * 1000 * (40 + 3) * (60 + 3 + 1)
end
