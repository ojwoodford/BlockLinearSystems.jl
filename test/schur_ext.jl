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
