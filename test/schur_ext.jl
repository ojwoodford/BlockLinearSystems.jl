using SecondOrderOptimization

# Tests for the SecondOrderOptimization extension (ext/SecondOrderOptimizationExt.jl).
# Every Schur solve is compared against a dense Cholesky solve of the full damped system.

const SOOExt = Base.get_extension(BlockLinearSystems, :SecondOrderOptimizationExt)

"""
    SchurTestProblem

A random, strictly diagonally dominant (hence positive definite) bipartite system with retained
variables first and eliminated variables second, in full dense form plus everything needed to build
`SchurBlockMatrix` objects from it.
"""
struct SchurTestProblem
    H::Matrix{Float64}
    b::Vector{Float64}
    c_dims::Vector{Int}
    p_dims::Vector{Int}
    observations::Vector{Vector{Int}}
    cc_pairs::Vector{Tuple{Int, Int}}   # retained-retained block couplings (r > c), beyond the diagonal
end

function SchurTestProblem(rng, c_dims, p_dims, observations; cc_pairs=Tuple{Int, Int}[])
    nc, np = sum(c_dims), sum(p_dims)
    n = nc + np
    coff = cumsum([1; c_dims[1:end-1]])
    poff = nc .+ cumsum([1; p_dims[1:end-1]])
    H = zeros(n, n)
    function fillblock!(r0, nr, c0, ncol)
        H[r0:r0+nr-1, c0:c0+ncol-1] .= 2 .* rand(rng, nr, ncol) .- 1
    end
    for (c, d) in enumerate(c_dims)
        fillblock!(coff[c], d, coff[c], d)
    end
    for (l, d) in enumerate(p_dims)
        fillblock!(poff[l], d, poff[l], d)
    end
    for (l, cams) in enumerate(observations), c in cams
        fillblock!(coff[c], c_dims[c], poff[l], p_dims[l])
    end
    for (r, c) in cc_pairs
        fillblock!(coff[r], c_dims[r], coff[c], c_dims[c])
    end
    # Symmetrize from the upper triangle (block diagonals are symmetrized element-wise)
    H = Matrix(Symmetric(H, :U))
    # Make strictly diagonally dominant, but keep the conditioning moderate
    for i in 1:n
        H[i, i] = sum(abs, view(H, i, :)) - abs(H[i, i]) + 0.5 + rand(rng)
    end
    return SchurTestProblem(H, randn(rng, n), c_dims, p_dims, observations, cc_pairs)
end

nc(p::SchurTestProblem) = sum(p.c_dims)

function schur_structure(p::SchurTestProblem; nchunks=1)
    return SchurStructure(p.c_dims, p.p_dims, p.observations; nchunks)
end

function fill_cp!(p::SchurTestProblem, structure::SchurStructure, C_data, E_data)
    nc_ = nc(p)
    for l in eachindex(p.p_dims)
        dp = p.p_dims[l]
        po = structure.ppoffsets[l]
        poff = nc_ + structure.boffsets_p[l]
        C_data[po:po+dp^2-1] .= vec(p.H[poff:poff+dp-1, poff:poff+dp-1])
        for e in structure.cpptr[l]:structure.cpptr[l+1]-1
            c = structure.cpblock[e]
            dc = p.c_dims[c]
            co = structure.boffsets_c[c]
            eo = structure.cpoffsets[e]
            # Stored as dp x dc column-major: E_data[eo + (r-1)*dp + i-1] = H[c_r, p_i]
            E_data[eo:eo+dc*dp-1] .= vec(p.H[co:co+dc-1, poff:poff+dp-1]')
        end
    end
    return nothing
end

function dense_schur_matrix(p::SchurTestProblem; nchunks=1)
    structure = schur_structure(p; nchunks)
    n_c = nc(p)
    B = p.H[1:n_c, 1:n_c]
    C_data = zeros(sum(abs2, p.p_dims))
    E_data = zeros(structure.cpoffsets[end] + Int(structure.c_dims[structure.cpblock[end]]) * Int(structure.p_dims[end]) - 1)
    fill_cp!(p, structure, C_data, E_data)
    return SchurBlockMatrix(structure, B, C_data, E_data)
end

function sparse_schur_matrix(p::SchurTestProblem; nchunks=1)
    structure = schur_structure(p; nchunks)
    covis = build_schur_covisibility(structure)
    coords = Tuple{Int, Int}[]
    cb = covis.block_colptr
    for c in 1:structure.ncblocks, k in cb[c]:cb[c+1]-1
        r = covis.block_rowval[k]       # r <= c in the upper-triangle covisibility
        push!(coords, (c, r))           # store lower-triangle block (row = c, col = r)
    end
    append!(coords, p.cc_pairs)
    unique!(coords)
    B = SparseBlockLinearSystem{Float64}(coords, p.c_dims)
    coff = structure.boffsets_c
    for (r, c) in coords
        block(B, r, c) .= p.H[coff[r]:coff[r]+p.c_dims[r]-1, coff[c]:coff[c]+p.c_dims[c]-1]
    end
    C_data = zeros(sum(abs2, p.p_dims))
    E_data = zeros(structure.cpoffsets[end] + Int(structure.c_dims[structure.cpblock[end]]) * Int(structure.p_dims[end]) - 1)
    fill_cp!(p, structure, C_data, E_data)
    return SchurBlockMatrix(structure, B, C_data, E_data)
end

function make_schur_ls(kind::Symbol, p::SchurTestProblem, scaling=ones(length(p.b)); nchunks=1)
    n = length(p.b)
    if kind === :dense
        return SOOExt.DenseSchurLinearSystem(dense_schur_matrix(p; nchunks), copy(p.b), zeros(n), copy(scaling))
    else
        return SOOExt.SparseSchurLinearSystem(sparse_schur_matrix(p; nchunks), copy(p.b), zeros(n), copy(scaling))
    end
end

"Reference solution: dense Cholesky of the full system with `lambda * diag(damping)` added to the diagonal."
function reference_solve(p::SchurTestProblem, lambda, damping)
    Hd = p.H + lambda * Diagonal(damping)
    return cholesky(Symmetric(Hd)) \ p.b
end

function standard_problem(seed=1; cc_pairs=Tuple{Int, Int}[])
    rng = Random.Xoshiro(seed)
    c_dims = [6, 6, 6, 6]
    p_dims = [3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
    # Cameras 1-2 and 3-4 observe landmarks together; 2-3 are linked through a few landmarks
    observations = [
        [1, 2], [1, 2], [1, 2, 3], [2, 3], [3, 4], [3, 4], [1, 4],
        [1, 2], [2, 3], [3, 4], [1, 2, 3, 4], [4],
        [1], [2], [3], [4], [1, 2], [3, 4], [2, 3], [1, 4],
    ]
    return SchurTestProblem(rng, c_dims, p_dims, observations; cc_pairs)
end

@testset "SecondOrderOptimization extension: Schur solve equals full Cholesky solve" begin
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
            SecondOrderOptimization.dampen!(ls, lambda)
            @test SecondOrderOptimization.solve!(ls, solver) == SecondOrderOptimization.Success
            x_ref = reference_solve(p, lambda, scaling)
            @test norm(ls.x - x_ref) <= 1e-10 * norm(x_ref)
        end
    end
end

# Copy the numerical content of `src` (a SchurBlockMatrix) into `dst`
function copy_content!(dst::SchurBlockMatrix, src::SchurBlockMatrix)
    dst.B isa Matrix ? copyto!(dst.B, src.B) : copyto!(dst.B.data, src.B.data)
    copyto!(dst.C_data, src.C_data)
    copyto!(dst.E_data, src.E_data)
    return dst
end

@testset "SecondOrderOptimization extension: copy_structure gives an independent trial system (B1)" begin
    p1 = standard_problem(1)
    p2 = standard_problem(2)
    n = length(p1.b)
    for kind in (:dense, :sparse)
        @testset "$kind" begin
            ls = make_schur_ls(kind, p1)
            trial = SecondOrderOptimization.copy_structure(ls)

            # New storage for the Hessian and gradient, shared x
            @test trial.H !== ls.H
            @test trial.H.B !== ls.H.B
            @test trial.H.C_data !== ls.H.C_data
            @test trial.H.E_data !== ls.H.E_data
            @test trial.b !== ls.b
            @test trial.x === ls.x
            @test SecondOrderOptimization.getA(trial) === trial.H

            # Linearize the trial at a different point: this must not touch the current system
            SecondOrderOptimization.zero!(trial)
            copy_content!(trial.H, make_schur_ls(kind, p2).H)
            trial.b .= p2.b
            # Reject the trial; retry on the current system with different damping, twice
            for lambda in (0.1, 2.0)
                SecondOrderOptimization.dampen!(ls, lambda)
                @test SecondOrderOptimization.solve!(ls) == SecondOrderOptimization.Success
                x_ref = reference_solve(p1, lambda, ones(n))
                @test norm(ls.x - x_ref) <= 1e-10 * norm(x_ref)
            end

            # And the trial system solves its own problem, writing into the shared x
            SecondOrderOptimization.dampen!(trial, 0.1)
            @test SecondOrderOptimization.solve!(trial) == SecondOrderOptimization.Success
            x_ref2 = reference_solve(p2, 0.1, ones(n))
            @test norm(trial.x - x_ref2) <= 1e-10 * norm(x_ref2)
        end
    end
end

# Two groups of cameras ({1, 2} and {3, 4}) that share no landmark with each other, coupled only by
# direct retained-retained blocks (odometry / loop-closure style), which have no eliminated variable.
function coupled_groups_problem(seed=3)
    rng = Random.Xoshiro(seed)
    c_dims = [6, 6, 6, 6]
    p_dims = [3, 3, 3, 3, 2, 2, 1, 1]
    observations = [[1, 2], [1, 2], [3, 4], [3, 4], [1], [4], [2, 1], [3, 4]]
    return SchurTestProblem(rng, c_dims, p_dims, observations; cc_pairs=[(3, 1), (4, 2), (2, 1)])
end

@testset "SecondOrderOptimization extension: retained-retained couplings without a shared eliminated variable (B4)" begin
    p = coupled_groups_problem()
    n = length(p.b)
    structure = schur_structure(p)

    # Pattern: the extra pairs get slots, in either order, and nothing else changes
    plain = build_schur_covisibility(structure)
    withextra = build_schur_covisibility(structure; extra_blocks=[(3, 1), (2, 4)])
    @test !plain.block_covis[1, 3] && !plain.block_covis[2, 4]
    @test withextra.block_covis[1, 3] && withextra.block_covis[2, 4]
    @test nnz(withextra.block_covis) == nnz(plain.block_covis) + 2
    @test_throws ArgumentError build_schur_covisibility(structure; extra_blocks=[(1, 5)])

    # The sparse system fails to build a slot for B's coupling without the extra pairs ...
    H = sparse_schur_matrix(p)
    @test Set(retained_block_pairs(H.B)) ⊇ Set([(3, 1), (4, 2)])
    @test_throws AssertionError build_b_to_s_map(H.B, plain.S_sparse, structure.boffsets_c)
    # ... and builds one with them
    covis = build_schur_covisibility(structure; extra_blocks=retained_block_pairs(H.B))
    b_to_s = build_b_to_s_map(H.B, covis.S_sparse, structure.boffsets_c)
    # every upper-triangle entry of the diagonal blocks (21 each) and all 36 entries of each off-diagonal block
    @test count(>(0), b_to_s) == 4 * 21 + 36 * length(retained_block_pairs(H.B))

    for kind in (:dense, :sparse), lambda in (0.0, 0.5)
        @testset "$kind, lambda=$lambda" begin
            ls = make_schur_ls(kind, p)
            SecondOrderOptimization.dampen!(ls, lambda)
            @test SecondOrderOptimization.solve!(ls) == SecondOrderOptimization.Success
            x_ref = reference_solve(p, lambda, ones(n))
            @test norm(ls.x - x_ref) <= 1e-10 * norm(x_ref)
        end
    end
end

@testset "build_schur_covisibility scales to many observations (B8) and can be reused" begin
    # Many cameras, each landmark seen by a window of consecutive cameras
    ncam, nlm = 3000, 6000
    obs = [collect(c:min(c + 3, ncam)) for c in (min.(1 .+ (0:nlm-1) .* ncam .÷ nlm, ncam))]
    structure = SchurStructure(fill(6, ncam), fill(3, nlm), obs)
    build_schur_covisibility(structure)   # compile
    t = @elapsed covis = build_schur_covisibility(structure)
    @test t < 5
    @test nnz(covis.block_covis) == length(covis.block_rowval)
    # Upper triangle with the diagonal; the sliding window gives bandwidth 3
    @test all(covis.block_rowval[k] <= c for c in 1:ncam for k in covis.block_colptr[c]:covis.block_colptr[c+1]-1)
    @test nnz(covis.block_covis) <= ncam * 4

    # Passing a precomputed covis to the constructor gives the same system
    p = coupled_groups_problem()
    H = sparse_schur_matrix(p)
    pre = build_schur_covisibility(H.structure; extra_blocks=retained_block_pairs(H.B))
    ls1 = SOOExt.SparseSchurLinearSystem(H, copy(p.b), zeros(length(p.b)))
    ls2 = SOOExt.SparseSchurLinearSystem(H, copy(p.b), zeros(length(p.b)); covis=pre)
    @test ls1.S == ls2.S && ls1.b_to_s_map == ls2.b_to_s_map
end

@testset "SecondOrderOptimization extension: undamped solves use Cholesky, fall back on failure (B5)" begin
    p = standard_problem()
    n = length(p.b)
    x_ref = reference_solve(p, 0.0, ones(n))
    for kind in (:dense, :sparse)
        @testset "$kind positive definite" begin
            ls = make_schur_ls(kind, p)
            SecondOrderOptimization.dampen!(ls, 0.0)
            # Solving twice gives identical results (S is rebuilt each time even though Cholesky works in place)
            for _ in 1:2
                @test SecondOrderOptimization.solve!(ls) == SecondOrderOptimization.Success
                @test norm(ls.x - x_ref) <= 1e-10 * norm(x_ref)
            end
        end
        @testset "$kind indefinite" begin
            # Make the retained diagonal very negative so that S is indefinite, and Cholesky must fail
            q = standard_problem()
            q.H[1, 1] = -50.0
            for solver in (:auto, :ldl)
                ls = make_schur_ls(kind, q)
                SecondOrderOptimization.dampen!(ls, 0.0)
                @test SecondOrderOptimization.solve!(ls, solver) == SecondOrderOptimization.Indefinite
            end
            # With enough damping it is positive definite again, and the fallback is not needed
            ls = make_schur_ls(kind, q)
            SecondOrderOptimization.dampen!(ls, 100.0)
            @test SecondOrderOptimization.solve!(ls) == SecondOrderOptimization.Success
            x_ref_q = reference_solve(q, 100.0, ones(n))
            @test norm(ls.x - x_ref_q) <= 1e-10 * norm(x_ref_q)
        end
    end
end

@testset "fast_bAb equals the full quadratic form (B6)" begin
    for p in (standard_problem(), coupled_groups_problem())
        n = length(p.b)
        dx = randn(Random.Xoshiro(5), n)
        expected = dot(dx, p.H * dx)
        for kind in (:dense, :sparse)
            H = kind === :dense ? dense_schur_matrix(p) : sparse_schur_matrix(p)
            @test BlockLinearSystems.fast_bAb(H, dx) ≈ expected rtol = 1e-12
            @test SecondOrderOptimization.fast_bAb(H, dx) ≈ expected rtol = 1e-12
            BlockLinearSystems.fast_bAb(H, dx)
            kind === :sparse && @test (@allocated BlockLinearSystems.fast_bAb(H, dx)) == 0
        end
    end
end

@testset "quadratic_model_reduction uses the undamped model on the full system (B10)" begin
    p = standard_problem()
    n = length(p.b)
    dx = randn(Random.Xoshiro(6), n)
    expected = 0.5 * dot(dx, p.H * dx) + dot(p.b, dx)
    for kind in (:dense, :sparse), lambda in (0.0, 0.7)
        ls = make_schur_ls(kind, p, abs.(diag(p.H)))
        SecondOrderOptimization.dampen!(ls, lambda)
        @test SecondOrderOptimization.quadratic_model_reduction(ls, dx) ≈ expected rtol = 1e-12
    end
end

@testset "Schur solve is serial and allocation free for small problems; threading matches (B11)" begin
    p = standard_problem()
    n = length(p.b)
    x_ref = reference_solve(p, 0.4, ones(n))
    for kind in (:dense, :sparse)
        ls = make_schur_ls(kind, p; nchunks=1)
        SecondOrderOptimization.dampen!(ls, 0.4)
        SecondOrderOptimization.solve!(ls)
        @test norm(ls.x - x_ref) <= 1e-10 * norm(x_ref)
        H = ls.H
        ws = ls.workspace
        # The reduction and back-substitution kernels allocate nothing when serial
        if kind === :dense
            run_reduce() = reduce_schur_dense!(ls.S, ls.g_S, ls.b_c, ls.b_p, ls.scaling_p, ls.lambda, H.structure, H.C_data, H.E_data, ws)
            @test run_reduce()
            @test (@allocated run_reduce()) == 0
        else
            run_reduce_s() = reduce_schur_sparse!(ls.S, ls.g_S, ls.b_c, ls.b_p, ls.scaling_p, ls.lambda, H.structure, H.C_data, H.E_data,
                ls.block_colptr, ls.block_rowval, ls.block_row_offsets, ls.col_heights, ws)
            @test run_reduce_s()
            @test (@allocated run_reduce_s()) == 0
        end
        run_back() = backsubstitute!(ls.x_p, ls.x_c, H.structure, H.E_data, ws.C_inv_data, ws.z_data)
        run_back()
        @test (@allocated run_back()) == 0
    end

    # Force the threaded code paths (they only engage with more than one thread) and compare
    old = BlockLinearSystems.SCHUR_THREAD_MIN_WORK[]
    try
        BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = 0
        for kind in (:dense, :sparse), nchunks in (1, 3)
            ls = make_schur_ls(kind, p; nchunks)
            SecondOrderOptimization.dampen!(ls, 0.4)
            @test SecondOrderOptimization.solve!(ls) == SecondOrderOptimization.Success
            @test norm(ls.x - x_ref) <= 1e-10 * norm(x_ref)
        end
    finally
        BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = old
    end
end

@testset "SecondOrderOptimization extension: solver symbols (B12)" begin
    p = standard_problem()
    Hd, Hs = dense_schur_matrix(p), sparse_schur_matrix(p)
    mk(H, s) = SecondOrderOptimization.makelinearsystem(nothing, H, s)
    @test mk(Hd, :schur) isa SOOExt.DenseSchurLinearSystem
    @test mk(Hd, :dense_schur) isa SOOExt.DenseSchurLinearSystem
    @test mk(Hs, :schur) isa SOOExt.SparseSchurLinearSystem
    @test mk(Hs, :sparse_schur) isa SOOExt.SparseSchurLinearSystem
    @test mk(Hd, static(:schur)) isa SOOExt.DenseSchurLinearSystem
    @test mk(Hd, SecondOrderOptimization.auto_solver) isa SOOExt.DenseSchurLinearSystem
    # A request that contradicts the storage of B is an error, not silently ignored or a MethodError
    @test_throws ArgumentError mk(Hd, :sparse_schur)
    @test_throws ArgumentError mk(Hs, :dense_schur)

    # solve! accepts Symbols, static symbols and the OptimizationLinearSolver enum without ambiguity
    n = length(p.b)
    x_ref = reference_solve(p, 0.3, ones(n))
    for kind in (:dense, :sparse), s in (:auto, :cholesky, :ldl, static(:auto), static(:ldl),
                                        SecondOrderOptimization.auto_solver, SecondOrderOptimization.cholesky_solver, SecondOrderOptimization.ldl_solver)
        ls = make_schur_ls(kind, p)
        SecondOrderOptimization.dampen!(ls, 0.3)
        @test SecondOrderOptimization.solve!(ls, s) == SecondOrderOptimization.Success
        @test norm(ls.x - x_ref) <= 1e-10 * norm(x_ref)
    end
end

@testset "SchurStructure validates its inputs (B13)" begin
    ok = SchurStructure([6, 6], [3, 3], [[1, 2], [2]])
    @test ok.npblocks == 2
    @test_throws DimensionMismatch SchurStructure([6, 6], [3, 3], [[1, 2]])
    @test_throws ArgumentError SchurStructure([6, 6], [3, 256], [[1, 2], [2]])    # eliminated block over 255
    @test SchurStructure([6, 6], [4, 255], [[1, 2], [2]]).npblocks == 2            # any size up to 255 is supported
    @test_throws ArgumentError SchurStructure([6, 6], [3, 0], [[1, 2], [2]])      # empty eliminated block
    @test_throws ArgumentError SchurStructure([6, 0], [3, 3], [[1, 2], [2]])      # empty retained block
    @test_throws ArgumentError SchurStructure([6, 300], [3, 3], [[1, 2], [2]])    # retained block over 255
    @test_throws ArgumentError SchurStructure([6, 6], [3, 3], [[1, 2], [3]])      # retained block out of range
    @test_throws ArgumentError SchurStructure([6, 6], [3, 3], [[1, 0], [2]])
    @test_throws ArgumentError SchurStructure([6, 6], [3, 3], [[1, 2, 1], [2]])   # duplicate observation
    @test_throws ArgumentError SchurStructure([6, 6], [3, 3], [[1, 2], [2]]; nchunks=0)
    # The same retained block may of course observe different landmarks
    @test SchurStructure([6, 6], [3, 3], [[1], [1]]).npblocks == 2
end

@testset "Multi-chunk Schur reduction needs buffers only for chunks 2..K (M3)" begin
    p = standard_problem()
    n = length(p.b)
    x_ref = reference_solve(p, 0.4, ones(n))
    for kind in (:dense, :sparse), nchunks in (1, 2, 4)
        ls = make_schur_ls(kind, p; nchunks)
        extra = kind === :dense ? ls.workspace.thread_dense_S : ls.workspace.thread_nzvals
        @test length(extra) == nchunks - 1
        SecondOrderOptimization.dampen!(ls, 0.4)
        @test SecondOrderOptimization.solve!(ls) == SecondOrderOptimization.Success
        @test norm(ls.x - x_ref) <= 1e-10 * norm(x_ref)
        # A second solve (buffers are reused) gives the same answer
        @test SecondOrderOptimization.solve!(ls) == SecondOrderOptimization.Success
        @test norm(ls.x - x_ref) <= 1e-10 * norm(x_ref)
    end
end

function to_float32(H::SchurBlockMatrix)
    B = if H.B isa Matrix
        Float32.(H.B)
    else
        S = H.B
        SparseBlockLinearSystem{Float32}(Float32.(S.data), S.indices, S.indicestransposed, S.rowblocksizes, S.columnblocksizes, S.m, S.n)
    end
    return SchurBlockMatrix(H.structure, B, Float32.(H.C_data), Float32.(H.E_data))
end

@testset "Schur solve in Float32 (M4)" begin
    p = standard_problem()
    n = length(p.b)
    x_ref = reference_solve(p, 0.3, ones(n))
    for kind in (:dense, :sparse)
        H = to_float32(kind === :dense ? dense_schur_matrix(p) : sparse_schur_matrix(p))
        ls = kind === :dense ? SOOExt.DenseSchurLinearSystem(H, Float32.(p.b), zeros(Float32, n)) :
                               SOOExt.SparseSchurLinearSystem(H, Float32.(p.b), zeros(Float32, n))
        @test eltype(ls.S) == Float32
        @test eltype(ls.x) == Float32
        SecondOrderOptimization.dampen!(ls, 0.3)
        @test SecondOrderOptimization.solve!(ls) == SecondOrderOptimization.Success
        @test norm(Float64.(ls.x) - x_ref) <= 1e-3 * norm(x_ref)
    end
    # The sparse system's factor type is concrete
    ls = SOOExt.SparseSchurLinearSystem(sparse_schur_matrix(p), copy(p.b), zeros(n))
    @test isconcretetype(typeof(ls.S_factor))
    @test isconcretetype(fieldtype(typeof(ls), :S_factor))

    # Singularity tolerance scales with the precision: a block that is singular to Float32 precision is rejected
    H2 = SMatrix{2, 2, Float32, 4}(1, 1, 1, 1 + 1f-6)
    @test !inv_sym_block(H2, 0f0, SVector{2, Float32}(1, 1)).success
    H2d = SMatrix{2, 2, Float64, 4}(1, 1, 1, 1 + 1e-6)
    @test inv_sym_block(H2d, 0.0, SVector{2, Float64}(1, 1)).success
end

@testset "build_b_to_s_map: block-level and scalar-level searches agree (M6)" begin
    for p in (standard_problem(), coupled_groups_problem())
        H = sparse_schur_matrix(p)
        covis = build_schur_covisibility(H.structure; extra_blocks=retained_block_pairs(H.B))
        boffsets = H.structure.boffsets_c
        m_scalar = build_b_to_s_map(H.B, covis.S_sparse, boffsets)
        m_block = build_b_to_s_map(H.B, covis, boffsets)
        @test m_block == m_scalar
        # Mapped entries are unique and land on upper-triangle slots
        mapped = filter(>(0), m_block)
        @test allunique(mapped)
        S = covis.S_sparse
        rows = rowvals(S)
        cols = [j for j in 1:size(S, 2) for _ in nzrange(S, j)]
        @test all(rows[k] <= cols[k] for k in mapped)
    end
    # Missing block gives an assertion rather than a wrong slot
    p = coupled_groups_problem()
    H = sparse_schur_matrix(p)
    plain = build_schur_covisibility(H.structure)
    @test_throws AssertionError build_b_to_s_map(H.B, plain, H.structure.boffsets_c)
end

# Eliminated blocks of every size class: tiny closed forms (1-3), the generic static tier (4..D_STATIC)
# and the dynamic tier above D_STATIC
function mixed_dims_problem(seed=11)
    rng = Random.Xoshiro(seed)
    c_dims = [6, 3, 9, 6]
    p_dims = [12, 12, 9, 9, 8, 8, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1]
    cams = 1:4
    observations = [sort(unique([rand(rng, cams) for _ in 1:3])) for _ in p_dims]
    return SchurTestProblem(rng, c_dims, p_dims, observations)
end

@testset "Eliminated blocks of any size (M5)" begin
    @test BlockLinearSystems.D_STATIC < 12
    p = mixed_dims_problem()
    n = length(p.b)
    scalings = ("I" => ones(n), "D'D" => abs.(diag(p.H)))
    for kind in (:dense, :sparse), nchunks in (1, 3), (sname, scaling) in scalings, lambda in (0.0, 0.5)
        @testset "$kind, K=$nchunks, $sname, lambda=$lambda" begin
            ls = make_schur_ls(kind, p, scaling; nchunks)
            SecondOrderOptimization.dampen!(ls, lambda)
            @test SecondOrderOptimization.solve!(ls) == SecondOrderOptimization.Success
            x_ref = reference_solve(p, lambda, scaling)
            @test norm(ls.x - x_ref) <= 1e-9 * norm(x_ref)
        end
    end

    # Each size alone, including the largest allowed (255) and sizes at the tier boundaries
    for dp in (1, 3, 4, 5, 7, 8, 9, 10, 20)
        rng = Random.Xoshiro(dp)
        q = SchurTestProblem(rng, [6, 6], [dp, dp], [[1, 2], [1]])
        for kind in (:dense, :sparse)
            ls = make_schur_ls(kind, q)
            SecondOrderOptimization.dampen!(ls, 0.2)
            @test SecondOrderOptimization.solve!(ls) == SecondOrderOptimization.Success
            x_ref = reference_solve(q, 0.2, ones(length(q.b)))
            @test norm(ls.x - x_ref) <= 1e-9 * norm(x_ref)
        end
    end

    # A single 255-dimensional eliminated block
    rng = Random.Xoshiro(255)
    big = SchurTestProblem(rng, [3, 3], [255], [[1, 2]])
    for kind in (:dense, :sparse)
        ls = make_schur_ls(kind, big)
        SecondOrderOptimization.dampen!(ls, 0.1)
        @test SecondOrderOptimization.solve!(ls) == SecondOrderOptimization.Success
        x_ref = reference_solve(big, 0.1, ones(length(big.b)))
        @test norm(ls.x - x_ref) <= 1e-8 * norm(x_ref)
    end
end

@testset "Large eliminated blocks: singular blocks are reported, kernels do not allocate (M5)" begin
    for dp in (6, 12)   # generic static tier and dynamic tier
        rng = Random.Xoshiro(dp)
        q = SchurTestProblem(rng, [6, 6], [dp, dp], [[1, 2], [1, 2]])
        H = dense_schur_matrix(q)
        # Make eliminated block 2 rank deficient: an outer product of one vector
        st = H.structure
        po = st.ppoffsets[2]
        v = randn(rng, dp)
        H.C_data[po:po+dp^2-1] .= vec(v * v')
        ls = SOOExt.DenseSchurLinearSystem(H, copy(q.b), zeros(length(q.b)))
        SecondOrderOptimization.dampen!(ls, 0.0)
        @test SecondOrderOptimization.solve!(ls) == SecondOrderOptimization.NotPositiveDefinite
        @test ls.workspace.failed_landmark[] == 2
        # Damping repairs it
        SecondOrderOptimization.dampen!(ls, 1.0)
        @test SecondOrderOptimization.solve!(ls) == SecondOrderOptimization.Success
    end

    q = SchurTestProblem(Random.Xoshiro(3), [6, 6], [12, 12, 5, 5], [[1, 2], [1], [2], [1, 2]])
    for kind in (:dense, :sparse)
        ls = make_schur_ls(kind, q; nchunks=1)
        SecondOrderOptimization.dampen!(ls, 0.3)
        SecondOrderOptimization.solve!(ls)
        H, ws = ls.H, ls.workspace
        run1 = if kind === :dense
            () -> reduce_schur_dense!(ls.S, ls.g_S, ls.b_c, ls.b_p, ls.scaling_p, ls.lambda, H.structure, H.C_data, H.E_data, ws)
        else
            () -> reduce_schur_sparse!(ls.S, ls.g_S, ls.b_c, ls.b_p, ls.scaling_p, ls.lambda, H.structure, H.C_data, H.E_data,
                ls.block_colptr, ls.block_rowval, ls.block_row_offsets, ls.col_heights, ws)
        end
        @test run1()
        # The reduction, including the dynamic-dimension blocks, does not allocate
        @test (@allocated run1()) == 0
    end
end

@testset "Persistent CHOLMOD matrix and Bunch-Kaufman indefinite check (B5)" begin
    p = standard_problem()
    n = length(p.b)
    ls = make_schur_ls(:sparse, p)
    @test ls.S_chol isa SparseArrays.CHOLMOD.Sparse
    S_chol_before = ls.S_chol
    for lambda in (0.2, 0.9)
        SecondOrderOptimization.dampen!(ls, lambda)
        @test SecondOrderOptimization.solve!(ls) == SecondOrderOptimization.Success
        x_ref = reference_solve(p, lambda, ones(n))
        @test norm(ls.x - x_ref) <= 1e-10 * norm(x_ref)
    end
    @test ls.S_chol === S_chol_before          # refreshed in place, not rebuilt
    tr = SecondOrderOptimization.copy_structure(ls)
    @test tr.S_chol !== ls.S_chol
    # bk_indefinite agrees with SecondOrderOptimization.is_indefinite
    for A in (let M = randn(Random.Xoshiro(1), 8, 8); Symmetric(M * M' + I) end, Symmetric(randn(Random.Xoshiro(2), 8, 8)))
        bk = bunchkaufman(A)
        @test SOOExt.bk_indefinite(bk) == SecondOrderOptimization.is_indefinite(bk)
    end
end
