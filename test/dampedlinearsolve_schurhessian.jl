module DampedLinearSolveSchurTests

using Test, Random, LinearAlgebra, SparseArrays, StaticArrays, Static
using DampedLinearSolve, BlockLinearSystems, Polyester   # Polyester: BlockLinearSystems' threaded kernels
using DampedLinearSolve: check_linear_system
include("fixtures/load.jl")
using Main.SchurFixtures, Main.AllocFixtures

# Tests of src/schur_linear_systems.jl and src/linear_system_conformance.jl: the Schur-complement
# linear systems of a SchurBlockHessian, through DampedLinearSolve's contract. Moved from SecondOrderOptimization's
# test/blocklinearsystems_ext.jl; the tests that run its iterators stay there.
# Every Schur solve is compared against a dense Cholesky solve of the full damped system.

const SOOExt = BlockLinearSystems   # the linear systems are the package's own (EL5)

function make_schur_ls(kind::Symbol, p::SchurTestProblem, scaling=ones(length(p.b)); nchunks=1)
    n = length(p.b)
    if kind === :dense
        return SOOExt.DenseSchurLinearSystem(dense_schur_hessian(p; nchunks), copy(p.b), zeros(n), copy(scaling))
    else
        return SOOExt.SparseSchurLinearSystem(sparse_schur_hessian(p; nchunks), copy(p.b), zeros(n), copy(scaling))
    end
end

@testset "BlockLinearSystemsExt: Schur solve equals full Cholesky solve" begin
    p = standard_problem()
    n = length(p.b)
    scalings = (
        "I" => ones(n),
        # Marquardt (:levenbergmarquardt_dtd) scaling: |diag(H)|, damping is lambda * diag(H), not squared (B2)
        "D'D" => abs.(diag(p.H)),
        "wide-range" => exp.(range(-4, 8; length=n)),
    )
    for kind in (:dense, :sparse), solver in (:auto, :ldl), (sname, scaling) in scalings, lambda in (0.0, 0.3)
        @testset "$kind, $solver, $sname, lambda=$lambda" begin
            ls = make_schur_ls(kind, p, scaling)
            DampedLinearSolve.dampen!(ls, lambda, scaling)
            @test DampedLinearSolve.solve!(ls, solver) == DampedLinearSolve.Success
            x_ref = reference_solve(p, lambda, scaling)
            @test norm(ls.x - x_ref) <= 1e-10 * norm(x_ref)
        end
    end
end

# Copy the numerical content of `src` (a SchurBlockHessian) into `dst`
function copy_content!(dst::SchurBlockHessian, src::SchurBlockHessian)
    dst.R isa Matrix ? copyto!(dst.R, src.R) : copyto!(dst.R.data, src.R.data)
    copyto!(dst.E_data, src.E_data)
    copyto!(dst.C_data, src.C_data)
    return dst
end

@testset "BlockLinearSystemsExt: copy_structure gives an independent trial system (B1)" begin
    p1 = standard_problem(1)
    p2 = standard_problem(2)
    n = length(p1.b)
    for kind in (:dense, :sparse)
        @testset "$kind" begin
            ls = make_schur_ls(kind, p1)
            trial = DampedLinearSolve.copy_structure(ls)

            # New storage for the Hessian and gradient, shared x
            @test trial.H !== ls.H
            @test trial.H.R !== ls.H.R
            @test trial.H.E_data !== ls.H.E_data
            @test trial.H.C_data !== ls.H.C_data
            @test trial.b !== ls.b
            @test trial.x === ls.x
            @test DampedLinearSolve.getA(trial) === trial.H

            # Linearize the trial at a different point: this must not touch the current system
            DampedLinearSolve.zero!(trial)
            copy_content!(trial.H, make_schur_ls(kind, p2).H)
            trial.b .= p2.b
            # Reject the trial; retry on the current system with different damping, twice
            for lambda in (0.1, 2.0)
                DampedLinearSolve.dampen!(ls, lambda)
                @test DampedLinearSolve.solve!(ls) == DampedLinearSolve.Success
                x_ref = reference_solve(p1, lambda, ones(n))
                @test norm(ls.x - x_ref) <= 1e-10 * norm(x_ref)
            end

            # And the trial system solves its own problem, writing into the shared x
            DampedLinearSolve.dampen!(trial, 0.1)
            @test DampedLinearSolve.solve!(trial) == DampedLinearSolve.Success
            x_ref2 = reference_solve(p2, 0.1, ones(n))
            @test norm(trial.x - x_ref2) <= 1e-10 * norm(x_ref2)
        end
    end
end

@testset "BlockLinearSystemsExt: retained-retained couplings without a shared eliminated variable (B4)" begin
    p = coupled_groups_problem()
    n = length(p.b)
    structure = schur_structure(p)

    # Pattern: the extra pairs get slots, in either order, and nothing else changes
    plain = build_schur_pattern(structure)
    withextra = build_schur_pattern(structure; extra_blocks=[(3, 1), (2, 4)])
    @test !plain.block_pattern[1, 3] && !plain.block_pattern[2, 4]
    @test withextra.block_pattern[1, 3] && withextra.block_pattern[2, 4]
    @test nnz(withextra.block_pattern) == nnz(plain.block_pattern) + 2
    @test_throws ArgumentError build_schur_pattern(structure; extra_blocks=[(1, 5)])

    # The sparse system fails to build a slot for R's coupling without the extra pairs ...
    H = sparse_schur_hessian(p)
    @test Set(retained_block_pairs(H.R)) ⊇ Set([(3, 1), (4, 2)])
    @test_throws ArgumentError build_r_to_s_map(H.R, plain.S_sparse, structure.boffsets_r)
    # ... and builds one with them
    pattern = build_schur_pattern(structure; extra_blocks=retained_block_pairs(H.R))
    b_to_s = build_r_to_s_map(H.R, pattern.S_sparse, structure.boffsets_r)
    # every upper-triangle entry of the diagonal blocks (21 each) and all 36 entries of each off-diagonal block
    @test count(>(0), b_to_s) == 4 * 21 + 36 * length(retained_block_pairs(H.R))

    for kind in (:dense, :sparse), lambda in (0.0, 0.5)
        @testset "$kind, lambda=$lambda" begin
            ls = make_schur_ls(kind, p)
            DampedLinearSolve.dampen!(ls, lambda)
            @test DampedLinearSolve.solve!(ls) == DampedLinearSolve.Success
            x_ref = reference_solve(p, lambda, ones(n))
            @test norm(ls.x - x_ref) <= 1e-10 * norm(x_ref)
        end
    end
end

@testset "A precomputed pattern passed to SparseSchurLinearSystem gives the same system (B8)" begin
    p = coupled_groups_problem()
    H = sparse_schur_hessian(p)
    pre = build_schur_pattern(H.structure; extra_blocks=retained_block_pairs(H.R))
    ls1 = SOOExt.SparseSchurLinearSystem(H, copy(p.b), zeros(length(p.b)))
    ls2 = SOOExt.SparseSchurLinearSystem(H, copy(p.b), zeros(length(p.b)); pattern=pre)
    @test ls1.S == ls2.S && ls1.r_to_s_map == ls2.r_to_s_map
end

@testset "BlockLinearSystemsExt: undamped solves use Cholesky, fall back on failure (B5)" begin
    p = standard_problem()
    n = length(p.b)
    x_ref = reference_solve(p, 0.0, ones(n))
    for kind in (:dense, :sparse)
        @testset "$kind positive definite" begin
            ls = make_schur_ls(kind, p)
            DampedLinearSolve.dampen!(ls, 0.0)
            # Solving twice gives identical results (S is rebuilt each time even though Cholesky works in place)
            for _ in 1:2
                @test DampedLinearSolve.solve!(ls) == DampedLinearSolve.Success
                @test norm(ls.x - x_ref) <= 1e-10 * norm(x_ref)
            end
        end
        @testset "$kind indefinite" begin
            # Make the retained diagonal very negative so that S is indefinite, and Cholesky must fail
            q = standard_problem()
            q.H[1, 1] = -50.0
            for solver in (:auto, :ldl)
                ls = make_schur_ls(kind, q)
                DampedLinearSolve.dampen!(ls, 0.0)
                @test DampedLinearSolve.solve!(ls, solver) == DampedLinearSolve.Indefinite
            end
            # With enough damping it is positive definite again, and the fallback is not needed
            ls = make_schur_ls(kind, q)
            DampedLinearSolve.dampen!(ls, 100.0)
            @test DampedLinearSolve.solve!(ls) == DampedLinearSolve.Success
            x_ref_q = reference_solve(q, 100.0, ones(n))
            @test norm(ls.x - x_ref_q) <= 1e-10 * norm(x_ref_q)
        end
    end
end

@testset "quadratic_model_change uses the undamped model on the full system (B10)" begin
    p = standard_problem()
    n = length(p.b)
    dx = randn(Random.Xoshiro(6), n)
    expected = 0.5 * dot(dx, p.H * dx) + dot(p.b, dx)
    for kind in (:dense, :sparse), lambda in (0.0, 0.7)
        ls = make_schur_ls(kind, p, abs.(diag(p.H)))
        DampedLinearSolve.dampen!(ls, lambda, abs.(diag(p.H)))
        @test DampedLinearSolve.quadratic_model_change(ls, dx) ≈ expected rtol = 1e-12
    end
end

# Measured through a function barrier: at testset scope Julia 1.10 boxes the Float64 result
schur_quadform_allocations(H, v) = @allocated DampedLinearSolve.quadform(H, v)
@testset "quadform of a SchurBlockHessian is BlockLinearSystems' dot, through the fallback (D1)" begin
    p = standard_problem()
    n = length(p.b)
    v = randn(Random.Xoshiro(7), n)
    for kind in (:dense, :sparse)
        H = DampedLinearSolve.getA(make_schur_ls(kind, p))
        @test H isa SchurBlockHessian
        @test which(DampedLinearSolve.quadform, (typeof(H), typeof(v))) == which(DampedLinearSolve.quadform, (Any, Any))
        @test DampedLinearSolve.quadform(H, v) ≈ dot(v, p.H * v) rtol = 1e-12
        @test DampedLinearSolve.quadform(H, v) == dot(v, H, v)
        @test (@inferred DampedLinearSolve.quadform(H, v)) isa Float64
        @test schur_quadform_allocations(H, v) == 0
    end
end

@testset "Schur solve is serial and allocation free for small problems; threading matches (B11)" begin
    @test Base.get_extension(BlockLinearSystems, :PolyesterExt) !== nothing
    p = standard_problem()
    n = length(p.b)
    x_ref = reference_solve(p, 0.4, ones(n))
    for kind in (:dense, :sparse)
        ls = make_schur_ls(kind, p; nchunks=1)
        DampedLinearSolve.dampen!(ls, 0.4)
        DampedLinearSolve.solve!(ls)
        @test norm(ls.x - x_ref) <= 1e-10 * norm(x_ref)
    end

    # The reduction and back-substitution kernels allocate nothing (TP3b): with one chunk per thread and the
    # threading threshold at zero, so that on more than one thread the threaded paths run (rules.md 30), and on
    # a problem with eliminated blocks above D_STATIC (9 and 12) as well as the standard one
    old = BlockLinearSystems.SCHUR_THREAD_MIN_WORK[]
    try
        BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = 0
        @test maximum(mixed_dims_problem().e_dims) > BlockLinearSystems.D_STATIC
        for q in (standard_problem(), mixed_dims_problem()), kind in (:dense, :sparse)
            ls = make_schur_ls(kind, q; nchunks=Threads.nthreads())
            DampedLinearSolve.dampen!(ls, 0.4)
            @test DampedLinearSolve.solve!(ls) == DampedLinearSolve.Success
            xq = reference_solve(q, 0.4, ones(length(q.b)))
            @test norm(ls.x - xq) <= 1e-9 * norm(xq)
            H = ls.H
            ws = ls.workspace
            if kind === :dense
                run_reduce() = reduce_schur_dense!(ls.S, ls.g_S, ls.b_r, ls.b_e, ls.scaling_e, ls.lambda, H.structure, H.E_data, H.C_data, ws)
                @test run_reduce()
                GC.gc()   # @allocated counts every thread: no finalizer from earlier tests runs inside the measurement
                @test threaded_allocs_ok(@allocated run_reduce())
            else
                run_reduce_s() = reduce_schur_sparse!(ls.S, ls.g_S, ls.b_r, ls.b_e, ls.scaling_e, ls.lambda, H.structure, H.E_data, H.C_data,
                    ls.block_colptr, ls.block_rowval, ls.block_row_offsets, ls.col_heights, ws)
                @test run_reduce_s()
                GC.gc()
                @test threaded_allocs_ok(@allocated run_reduce_s())
            end
            run_back() = backsubstitute!(ls.x_e, ls.x_r, H.structure, H.C_data, ws)
            run_back()
            GC.gc()
            @test threaded_allocs_ok(@allocated run_back())
        end
    finally
        BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = old
    end

    # Force the threaded code paths (they only engage with more than one thread) and compare
    old = BlockLinearSystems.SCHUR_THREAD_MIN_WORK[]
    try
        BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = 0
        for kind in (:dense, :sparse), nchunks in (1, 3)
            ls = make_schur_ls(kind, p; nchunks)
            DampedLinearSolve.dampen!(ls, 0.4)
            @test DampedLinearSolve.solve!(ls) == DampedLinearSolve.Success
            @test norm(ls.x - x_ref) <= 1e-10 * norm(x_ref)
        end
    finally
        BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = old
    end
end


@testset "Chunks write disjoint column ranges of one S: no per-chunk copies (M3)" begin
    p = standard_problem()
    n = length(p.b)
    x_ref = reference_solve(p, 0.4, ones(n))
    for kind in (:dense, :sparse), nchunks in (1, 2, 4, 9)
        ls = make_schur_ls(kind, p; nchunks)
        ws = ls.workspace
        @test !hasproperty(ws, :thread_dense_S) && !hasproperty(ws, :thread_nzvals)
        # col_chunks tile the retained blocks without overlap
        @test length(ws.col_chunks) == nchunks
        @test reduce(vcat, collect.(ws.col_chunks)) == collect(1:ls.H.structure.nrblocks)
        DampedLinearSolve.dampen!(ls, 0.4)
        @test DampedLinearSolve.solve!(ls) == DampedLinearSolve.Success
        @test norm(ls.x - x_ref) <= 1e-10 * norm(x_ref)
        @test DampedLinearSolve.solve!(ls) == DampedLinearSolve.Success   # workspace reuse
        @test norm(ls.x - x_ref) <= 1e-10 * norm(x_ref)
    end
    # Forced-threaded run with many chunks gives the same answer
    old = BlockLinearSystems.SCHUR_THREAD_MIN_WORK[]
    try
        BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = 0
        for kind in (:dense, :sparse)
            ls = make_schur_ls(kind, p; nchunks=3)
            DampedLinearSolve.dampen!(ls, 0.4)
            @test DampedLinearSolve.solve!(ls) == DampedLinearSolve.Success
            @test norm(ls.x - x_ref) <= 1e-10 * norm(x_ref)
        end
    finally
        BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = old
    end
end

function to_float32(H::SchurBlockHessian)
    R = if H.R isa Matrix
        Float32.(H.R)
    else
        S = H.R
        SparseBlockLinearSystem{Float32}(Float32.(S.data), S.indices, S.indicestransposed, S.rowblocksizes, S.columnblocksizes, S.m, S.n)
    end
    return SchurBlockHessian(H.structure, R, Float32.(H.E_data), Float32.(H.C_data))
end



@testset "Eliminated blocks of any size (M5)" begin
    @test BlockLinearSystems.D_STATIC < 12
    p = mixed_dims_problem()
    n = length(p.b)
    scalings = ("I" => ones(n), "D'D" => abs.(diag(p.H)))
    for kind in (:dense, :sparse), nchunks in (1, 3), (sname, scaling) in scalings, lambda in (0.0, 0.5)
        @testset "$kind, K=$nchunks, $sname, lambda=$lambda" begin
            ls = make_schur_ls(kind, p, scaling; nchunks)
            DampedLinearSolve.dampen!(ls, lambda, scaling)
            @test DampedLinearSolve.solve!(ls) == DampedLinearSolve.Success
            x_ref = reference_solve(p, lambda, scaling)
            @test norm(ls.x - x_ref) <= 1e-9 * norm(x_ref)
        end
    end

    # One size alone: the Schur systems have no size-dependent code, and mixed_dims_problem has just been through the same
    # wiring; 7 is a static size it does not contain (decided 2026-11-17)
    for de in (7,)
        rng = Random.Xoshiro(de)
        q = SchurTestProblem(rng, [6, 6], [de, de], [[1, 2], [1]])
        for kind in (:dense, :sparse)
            ls = make_schur_ls(kind, q)
            DampedLinearSolve.dampen!(ls, 0.2)
            @test DampedLinearSolve.solve!(ls) == DampedLinearSolve.Success
            x_ref = reference_solve(q, 0.2, ones(length(q.b)))
            @test norm(ls.x - x_ref) <= 1e-9 * norm(x_ref)
        end
    end

    # A single 255-dimensional eliminated block
    rng = Random.Xoshiro(255)
    big = SchurTestProblem(rng, [3, 3], [255], [[1, 2]])
    for kind in (:dense, :sparse)
        ls = make_schur_ls(kind, big)
        DampedLinearSolve.dampen!(ls, 0.1)
        @test DampedLinearSolve.solve!(ls) == DampedLinearSolve.Success
        x_ref = reference_solve(big, 0.1, ones(length(big.b)))
        @test norm(ls.x - x_ref) <= 1e-8 * norm(x_ref)
    end
end

@testset "Large eliminated blocks: singular blocks are reported, kernels do not allocate (M5)" begin
    for de in (6, 12)   # generic static tier and dynamic tier
        rng = Random.Xoshiro(de)
        q = SchurTestProblem(rng, [6, 6], [de, de], [[1, 2], [1, 2]])
        H = dense_schur_hessian(q)
        # Make eliminated block 2 rank deficient: an outer product of one vector
        st = H.structure
        eo = st.eoffsets[2]
        v = randn(rng, de)
        H.E_data[eo:eo+de^2-1] .= vec(v * v')
        ls = SOOExt.DenseSchurLinearSystem(H, copy(q.b), zeros(length(q.b)))
        DampedLinearSolve.dampen!(ls, 0.0)
        @test DampedLinearSolve.solve!(ls) == DampedLinearSolve.NotPositiveDefinite
        @test ls.workspace.failed_block[] == 2
        # Damping repairs it
        DampedLinearSolve.dampen!(ls, 1.0)
        @test DampedLinearSolve.solve!(ls) == DampedLinearSolve.Success
    end

    q = SchurTestProblem(Random.Xoshiro(3), [6, 6], [12, 12, 5, 5], [[1, 2], [1], [2], [1, 2]])
    for kind in (:dense, :sparse)
        ls = make_schur_ls(kind, q; nchunks=1)
        DampedLinearSolve.dampen!(ls, 0.3)
        DampedLinearSolve.solve!(ls)
        H, ws = ls.H, ls.workspace
        run1 = if kind === :dense
            () -> reduce_schur_dense!(ls.S, ls.g_S, ls.b_r, ls.b_e, ls.scaling_e, ls.lambda, H.structure, H.E_data, H.C_data, ws)
        else
            () -> reduce_schur_sparse!(ls.S, ls.g_S, ls.b_r, ls.b_e, ls.scaling_e, ls.lambda, H.structure, H.E_data, H.C_data,
                ls.block_colptr, ls.block_rowval, ls.block_row_offsets, ls.col_heights, ws)
        end
        @test run1()
        # The reduction, including the dynamic-dimension blocks, does not allocate
        @test (@allocated run1()) == 0
    end
end

@testset "Persistent CHOLMOD matrix (B5)" begin
    p = standard_problem()
    n = length(p.b)
    ls = make_schur_ls(:sparse, p)
    @test ls.S_chol isa SparseArrays.CHOLMOD.Sparse
    S_chol_before = ls.S_chol
    for lambda in (0.2, 0.9)
        DampedLinearSolve.dampen!(ls, lambda)
        @test DampedLinearSolve.solve!(ls) == DampedLinearSolve.Success
        x_ref = reference_solve(p, lambda, ones(n))
        @test norm(ls.x - x_ref) <= 1e-10 * norm(x_ref)
    end
    @test ls.S_chol === S_chol_before          # refreshed in place, not rebuilt
    tr = DampedLinearSolve.copy_structure(ls)
    @test tr.S_chol !== ls.S_chol
end

@testset "BlockLinearSystemsExt: well-conditioned problems, forward error of x_r and x_e (T1)" begin
    for p in (standard_problem(), coupled_groups_problem(), mixed_dims_problem())
        @test cond(p.H) <= 1e4
        n = length(p.b)
        nr_ = sum(p.r_dims)
        for kind in (:dense, :sparse), solver in (:auto, :cholesky, :ldl), lambda in (0.0, 0.3)
            ls = make_schur_ls(kind, p)
            DampedLinearSolve.dampen!(ls, lambda)
            @test DampedLinearSolve.solve!(ls, solver) == DampedLinearSolve.Success
            x_ref = reference_solve(p, lambda, ones(n))
            @test norm(ls.x_r - x_ref[1:nr_]) <= 1e-10 * norm(x_ref[1:nr_])
            @test norm(ls.x_e - x_ref[nr_+1:end]) <= 1e-10 * norm(x_ref[nr_+1:end])
        end
    end
end

@testset "BlockLinearSystemsExt: gauge-deficient, ill-conditioned problem: relative residual (T1)" begin
    p = gauge_problem()
    n = length(p.b)
    @test cond(p.H) > 1e7
    for kind in (:dense, :sparse), solver in (:auto, :ldl), lambda in (1e-6, 1e-3, 0.1)
        ls = make_schur_ls(kind, p)
        DampedLinearSolve.dampen!(ls, lambda)
        @test DampedLinearSolve.solve!(ls, solver) == DampedLinearSolve.Success
        Hd = p.H + lambda * Diagonal(ones(n))
        rel_res = norm(Hd * ls.x - p.b) / (opnorm(Hd) * norm(ls.x) + norm(p.b))
        @test rel_res <= 1e-13
    end
end

@testset "BlockLinearSystemsExt: no damping drift across rejected trials (T2)" begin
    p = standard_problem()
    n = length(p.b)
    for kind in (:dense, :sparse)
        sc = abs.(diag(p.H))
        ls = make_schur_ls(kind, p, sc)      # Marquardt scaling
        snapshot(H) = (copy(H.E_data), copy(H.C_data), H.R isa Matrix ? copy(H.R) : copy(H.R.data), copy(ls.b), copy(ls.scaling))
        before = snapshot(ls.H)
        lambda0 = 1e-3
        DampedLinearSolve.dampen!(ls, lambda0, sc)
        @test DampedLinearSolve.solve!(ls) == DampedLinearSolve.Success
        x_first = copy(ls.x)
        # 100 increasing damping values, as in a run of rejected trial steps
        for lambda in lambda0 .* 1.5 .^ (1:100)
            DampedLinearSolve.dampen!(ls, lambda, sc)
            @test DampedLinearSolve.solve!(ls) == DampedLinearSolve.Success
        end
        DampedLinearSolve.dampen!(ls, lambda0, sc)
        @test DampedLinearSolve.solve!(ls) == DampedLinearSolve.Success
        @test ls.x == x_first                                   # bit for bit
        @test all(snapshot(ls.H) .== before)                     # R, C, E, b and the scaling are untouched
        @test ls.lambda == lambda0
    end
end

@testset "Sparse Schur system takes its pattern from H (M26)" begin
    p = coupled_groups_problem()
    n = length(p.b)
    H0 = sparse_schur_hessian(p)
    pre = build_schur_pattern(H0.structure; extra_blocks=retained_block_pairs(H0.R))
    Hp = SchurBlockHessian(H0.structure, H0.R, H0.E_data, H0.C_data; pattern=pre)
    x_ref = reference_solve(p, 0.3, ones(n))
    mk(H) = DampedLinearSolve.makehessiansystem(H, :auto)

    # With and without a pattern on H: the same solution
    for H in (H0, Hp)
        ls = mk(H)
        ls.b .= p.b
        DampedLinearSolve.dampen!(ls, 0.3)
        @test DampedLinearSolve.solve!(ls) == DampedLinearSolve.Success
        @test norm(ls.x - x_ref) <= 1e-10 * norm(x_ref)
    end

    # The pattern was used, not rebuilt: its block structure is shared, S is a copy
    ls_a = mk(Hp)
    @test ls_a.block_rowval === pre.block_rowval
    @test ls_a.S !== pre.S_sparse && ls_a.S == pre.S_sparse
    @test mk(H0).block_rowval !== pre.block_rowval
    # ...and building the system does less work than rebuilding the pattern
    mk(Hp); mk(H0)
    @test (@allocated mk(Hp)) < (@allocated mk(H0))

    # Two systems from one H don't share S: solving the first leaves the second's untouched
    ls_b = mk(Hp)
    S_b = copy(ls_b.S)
    S_pre = copy(pre.S_sparse)
    ls_a.b .= p.b
    DampedLinearSolve.dampen!(ls_a, 0.3)
    @test DampedLinearSolve.solve!(ls_a) == DampedLinearSolve.Success
    @test ls_b.S == S_b
    @test pre.S_sparse == S_pre
    @test ls_a.S != S_b   # the first system did write its own S

    # An analysed factor carried by the pattern is reused
    fac = SparseArrays.CHOLMOD.analyze(SparseArrays.CHOLMOD.Sparse(Symmetric(copy(pre.S_sparse), :U)))
    Hf = SchurBlockHessian(H0.structure, H0.R, H0.E_data, H0.C_data; pattern=merge(pre, (S_factor = fac,)))
    ls_f = mk(Hf)
    @test ls_f.S_factor === fac
    ls_f.b .= p.b
    DampedLinearSolve.dampen!(ls_f, 0.3)
    @test DampedLinearSolve.solve!(ls_f) == DampedLinearSolve.Success
    @test norm(ls_f.x - x_ref) <= 1e-10 * norm(x_ref)

    # The dense system ignores a pattern
    Hd = dense_schur_hessian(p)
    Hdp = SchurBlockHessian(Hd.structure, Hd.R, Hd.E_data, Hd.C_data; pattern=build_schur_pattern(Hd.structure))
    @test DampedLinearSolve.makehessiansystem(Hdp, :auto) isa SOOExt.DenseSchurLinearSystem
end

# Add d to the diagonal of the full matrix a SchurBlockHessian represents
function add_schur_diag!(H::SchurBlockHessian, d)
    st = H.structure
    for r in 1:st.nrblocks, i in 1:Int(st.r_dims[r])
        k = st.boffsets_r[r] + i - 1
        if H.R isa Matrix
            H.R[k, k] += d[k]
        else
            block(H.R, r, r)[i, i] += d[k]
        end
    end
    for e in 1:st.neblocks
        de = Int(st.e_dims[e])
        for i in 1:de
            H.E_data[st.eoffsets[e] + (i - 1) * (de + 1)] += d[st.nr + st.boffsets_e[e] + i - 1]
        end
    end
    return H
end


@testset "Schur systems meet the linear-system interface (A3)" begin
    for (name, p) in (("standard", standard_problem()), ("retained-retained couplings", coupled_groups_problem()),
                      ("mixed eliminated sizes", mixed_dims_problem()))
        n = length(p.b)
        @testset "$name, dense R" begin
            @test check_linear_system(s -> DampedLinearSolve.makehessiansystem(dense_schur_hessian(p), s); n, linsolvers=(:auto, :cholesky, :ldl)) === nothing
        end
        @testset "$name, sparse R" begin
            @test check_linear_system(s -> DampedLinearSolve.makehessiansystem(sparse_schur_hessian(p), s); n, linsolvers=(:auto, :cholesky, :ldl)) === nothing
        end
    end
    # The helpers read back exactly the full matrix the problem was built from
    p = standard_problem()
    for H in (dense_schur_hessian(p), sparse_schur_hessian(p))
        @test DampedLinearSolve.conformance_matrix(H) ≈ p.H
    end
end




@testset "No Schur solve reports Success with a non-finite step (AS4)" begin
    p = standard_problem()
    setR!(H, i, j, v) = H.R isa Matrix ? (H.R[i, j] = H.R[j, i] = v) : (B = block(H.R, 1, 1); B[i, j] = B[j, i] = v)
    variants = (("NaN on R's diagonal", H -> setR!(H, 1, 1, NaN)), ("NaN off R's diagonal", H -> setR!(H, 1, 2, NaN)),
                ("NaN on E's diagonal", H -> (H.E_data[1] = NaN)), ("NaN in C", H -> (H.C_data[1] = NaN)),
                ("Inf on R's diagonal", H -> setR!(H, 1, 1, Inf)))
    for kind in (:dense, :sparse), s in (:auto, :cholesky, :ldl), (name, f!) in variants, lambda in (0.0, 1e-3)
        @testset "$kind, $s, $name, λ = $lambda" begin
            ls = make_schur_ls(kind, p)
            @test DampedLinearSolve.isfinitesystem(ls)
            f!(ls.H)
            @test !DampedLinearSolve.isfinitesystem(ls)
            DampedLinearSolve.dampen!(ls, lambda)
            st = DampedLinearSolve.solve!(ls, static(s))
            @test st != DampedLinearSolve.Success || all(isfinite, ls.x)
        end
    end
end

# From SecondOrderOptimization's tests (TF2): what only this package can see of its Schur systems

@testset "The storage of R picks the dense or the sparse Schur system, whatever the reduced-system solver" begin
    p = standard_problem()
    Hd, Hs = dense_schur_hessian(p), sparse_schur_hessian(p)
    for s in (:auto, :cholesky, :ldl, static(:auto), static(:cholesky), static(:ldl))
        @test DampedLinearSolve.makehessiansystem(Hd, s) isa SOOExt.DenseSchurLinearSystem
        @test DampedLinearSolve.makehessiansystem(Hs, s) isa SOOExt.SparseSchurLinearSystem
    end
    # The sparse system's factor type is concrete
    ls = SOOExt.SparseSchurLinearSystem(sparse_schur_hessian(p), copy(p.b), zeros(length(p.b)))
    @test isconcretetype(typeof(ls.S_factor))
    @test isconcretetype(fieldtype(typeof(ls), :S_factor))
end

# AS3: the dense Schur system's Bunch-Kaufman solve, for :ldl and as :auto's fallback, against the solve of the reduced
# system as it was written before: Cholesky (for :auto), then LinearAlgebra's bunchkaufman
function as3_reference(ls, solver)
    SOOExt.build_reduced!(ls) || return (DampedLinearSolve.NotPositiveDefinite, nothing)
    S, g = copy(ls.S), copy(ls.g_S)
    if solver === :auto
        F = cholesky(Symmetric(copy(S), :U); check = false)
        issuccess(F) && return (DampedLinearSolve.Success, F \ g)
    end
    bk = bunchkaufman(Symmetric(S, :U); check = false)
    issuccess(bk) || return (DampedLinearSolve.NotPositiveDefinite, nothing)
    xr = bk \ g
    return (DampedLinearSolve.is_indefinite(bk) ? DampedLinearSolve.Indefinite : DampedLinearSolve.Success, xr)
end
as3_solve_allocs(ls, s) = @allocated DampedLinearSolve.solve!(ls, s)

@testset "The dense Schur system's Bunch-Kaufman solve equals the reduced reference and allocates nothing (AS3)" begin
    p = standard_problem()
    q = standard_problem()
    q.H[1, 1] = -50.0                        # S indefinite, so Cholesky fails
    zeroS(ls) = (ls.H.R .= 0; ls.H.C_data .= 0; ls)   # S = 0: Bunch-Kaufman finds an exactly zero pivot
    for (name, mk, expected) in (("positive definite", () -> make_schur_ls(:dense, p), Dict(:auto => :Success, :ldl => :Success)),
                                 ("indefinite", () -> make_schur_ls(:dense, q), Dict(:auto => :Indefinite, :ldl => :Indefinite)),
                                 ("singular", () -> zeroS(make_schur_ls(:dense, p)), Dict(:auto => :NotPositiveDefinite, :ldl => :NotPositiveDefinite)))
        for s in (:auto, :ldl)
            @testset "$name, $s" begin
                ls = mk()
                DampedLinearSolve.dampen!(ls, 0.0)
                st_ref, xr_ref = as3_reference(ls, s)
                @test Symbol(st_ref) === expected[s]
                @test DampedLinearSolve.solve!(ls, static(s)) == st_ref
                isnothing(xr_ref) || @test norm(ls.x_r - xr_ref) <= 1e-12 * norm(xr_ref)
                as3_solve_allocs(ls, static(s))
                @test as3_solve_allocs(ls, static(s)) == 0
            end
        end
    end
end

end # module DampedLinearSolveSchurTests
