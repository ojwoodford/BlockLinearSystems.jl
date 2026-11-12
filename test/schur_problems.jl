using Random, LinearAlgebra, SparseArrays, StaticArrays, Static, Test, BlockLinearSystems

# Schur-complement data structures and kernels, tested with the public API only (no optimizer needed).
# The linear systems that plug these into SecondOrderOptimization are tested in that package.

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

# Two groups of cameras ({1, 2} and {3, 4}) that share no landmark with each other, coupled only by
# direct retained-retained blocks (odometry / loop-closure style), which have no eliminated variable.
function coupled_groups_problem(seed=3)
    rng = Random.Xoshiro(seed)
    c_dims = [6, 6, 6, 6]
    p_dims = [3, 3, 3, 3, 2, 2, 1, 1]
    observations = [[1, 2], [1, 2], [3, 4], [3, 4], [1], [4], [2, 1], [3, 4]]
    return SchurTestProblem(rng, c_dims, p_dims, observations; cc_pairs=[(3, 1), (4, 2), (2, 1)])
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


using Polyester

"""
    schur_solve(H, b, scaling, lambda; nchunks=1) -> (x, success)

Solve `(H + lambda * diag(scaling)) x = b` with the public Schur kernels only: reduce, dense or sparse
Cholesky of the reduced system, back-substitution.
"""
function schur_solve(H::SchurBlockMatrix{T}, b, scaling, lambda; nchunks=1) where T
    st = H.structure
    nc = st.nc
    n = length(b)
    b_c, b_p = view(b, 1:nc), view(b, nc+1:n)
    sc_c, sc_p = view(scaling, 1:nc), view(scaling, nc+1:n)
    ws = SchurReductionWorkspace{T}(st, nchunks)
    g = zeros(T, nc)
    lambda = T(lambda)
    if H.B isa Matrix
        S = zeros(T, nc, nc)
        reset_schur_dense!(S, H.B, sc_c, lambda, ws)
        reduce_schur_dense!(S, g, b_c, b_p, sc_p, lambda, st, H.C_data, H.E_data, ws) || return (zeros(T, n), false)
    else
        covis = build_schur_covisibility(st, T; extra_blocks=retained_block_pairs(H.B))
        S = covis.S_sparse
        reset_schur_sparse!(S.nzval, H.B.data, build_b_to_s_map(H.B, covis, st.boffsets_c), covis.diag_inds, sc_c, lambda, ws)
        reduce_schur_sparse!(S, g, b_c, b_p, sc_p, lambda, st, H.C_data, H.E_data,
                             covis.block_colptr, covis.block_rowval, covis.block_row_offsets, covis.col_heights, ws) || return (zeros(T, n), false)
    end
    x = zeros(T, n)
    x_c = cholesky(Symmetric(S, :U)) \ g
    x[1:nc] .= x_c
    backsubstitute!(view(x, nc+1:n), x_c, st, H.E_data, ws.C_inv_data, ws.z_data)
    return (x, true)
end

@testset "Schur solve equals full Cholesky solve (T1)" begin
    for p in (standard_problem(), coupled_groups_problem(), mixed_dims_problem())
        n = length(p.b)
        scalings = ("I" => ones(n), "D'D" => abs.(diag(p.H)), "wide-range" => exp.(range(-4, 8; length=n)))
        for make in (dense_schur_matrix, sparse_schur_matrix), nchunks in (1, 3), (sname, scaling) in scalings, lambda in (0.0, 0.3)
            x, ok = schur_solve(make(p; nchunks), p.b, scaling, lambda; nchunks)
            x_ref = reference_solve(p, lambda, scaling)
            @test ok
            @test norm(x - x_ref) <= 1e-9 * norm(x_ref)
        end
    end
    # Each eliminated block size on its own, including the extremes and the static/dynamic boundary
    for dp in (1, 3, 4, 5, 7, 8, 9, 10, 20, 255)
        q = SchurTestProblem(Random.Xoshiro(dp), [6, 6], [dp, dp], [[1, 2], [1]])
        for make in (dense_schur_matrix, sparse_schur_matrix)
            x, ok = schur_solve(make(q), q.b, ones(length(q.b)), 0.2)
            x_ref = reference_solve(q, 0.2, ones(length(q.b)))
            @test ok && norm(x - x_ref) <= 1e-8 * norm(x_ref)
        end
    end
end

@testset "copy_structure(::SchurBlockMatrix) shares nothing mutable (B1)" begin
    for make in (dense_schur_matrix, sparse_schur_matrix)
        H = make(standard_problem())
        T = copy_structure(H)
        @test T.structure === H.structure
        @test T.C_data !== H.C_data && T.E_data !== H.E_data && T.B !== H.B
        @test all(iszero, T.C_data) && all(iszero, T.E_data)
        before = copy(H.C_data)
        T.C_data .= 1
        @test H.C_data == before
        T.B isa Matrix ? (T.B .= 7) : (T.B.data .= 7)
        @test !(H.B isa Matrix ? all(==(7), H.B) : all(==(7), H.B.data))
    end
    ws = SchurReductionWorkspace{Float64}(schur_structure(standard_problem()), 2)
    tw = trial_workspace(ws)
    @test tw.C_inv_data !== ws.C_inv_data && tw.z_data !== ws.z_data && tw.failed_landmark !== ws.failed_landmark
end

@testset "Singular eliminated blocks are reported, kernels do not allocate (B11, M5)" begin
    for dp in (2, 6, 12)   # closed form, generic static tier and dynamic tier
        rng = Random.Xoshiro(dp)
        q = SchurTestProblem(rng, [6, 6], [dp, dp], [[1, 2], [1, 2]])
        H = dense_schur_matrix(q)
        po = H.structure.ppoffsets[2]
        v = randn(rng, dp)
        H.C_data[po:po+dp^2-1] .= vec(v * v')
        x, ok = schur_solve(H, q.b, ones(length(q.b)), 0.0)
        @test !ok
        ws = SchurReductionWorkspace{Float64}(H.structure, 1)
        S = copy(H.B)
        reset_schur_dense!(S, H.B, ones(12), 0.0, ws)
        @test !reduce_schur_dense!(S, zeros(12), q.b[1:12], q.b[13:end], ones(2dp), 0.0, H.structure, H.C_data, H.E_data, ws)
        @test ws.failed_landmark[] == 2
        _, ok2 = schur_solve(H, q.b, ones(length(q.b)), 1.0)
        @test ok2
    end
    q = SchurTestProblem(Random.Xoshiro(3), [6, 6], [12, 12, 5, 5], [[1, 2], [1], [2], [1, 2]])
    for make in (dense_schur_matrix, sparse_schur_matrix)
        H = make(q)
        n = length(q.b)
        st = H.structure
        ws = SchurReductionWorkspace{Float64}(st, 1)
        g = zeros(12)
        if H.B isa Matrix
            S = zeros(12, 12)
            run1 = () -> reduce_schur_dense!(S, g, view(q.b, 1:12), view(q.b, 13:n), ones(n - 12), 0.3, st, H.C_data, H.E_data, ws)
        else
            covis = build_schur_covisibility(st; extra_blocks=retained_block_pairs(H.B))
            S = covis.S_sparse
            run1 = () -> reduce_schur_sparse!(S, g, view(q.b, 1:12), view(q.b, 13:n), ones(n - 12), 0.3, st, H.C_data, H.E_data,
                covis.block_colptr, covis.block_rowval, covis.block_row_offsets, covis.col_heights, ws)
        end
        @test run1()
        bp, onesp = collect(view(q.b, 13:n)), ones(n - 12)
        @test (@allocated run1()) < 2048   # views and the closure only; the kernels themselves allocate nothing
    end
end

@testset "Column chunks tile the retained blocks without overlap (M3)" begin
    p = standard_problem()
    st = schur_structure(p)
    for K in (1, 2, 4, 9)
        ws = SchurReductionWorkspace{Float64}(st, K)
        @test length(ws.col_chunks) == K
        @test reduce(vcat, collect.(ws.col_chunks)) == collect(1:st.ncblocks)
        # the retained-block-major index covers every coupling exactly once, in ascending landmark order
        @test sort(ws.cam_entry) == collect(1:length(st.cpblock))
        for c in 1:st.ncblocks
            es = ws.cam_entry[ws.cam_ptr[c]:ws.cam_ptr[c+1]-1]
            @test all(st.cpblock[es] .== c) && issorted(ws.entry_landmark[es])
        end
    end
    # Forcing the threaded paths gives the same result
    old = BlockLinearSystems.SCHUR_THREAD_MIN_WORK[]
    try
        BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = 0
        x_ref = reference_solve(p, 0.4, ones(length(p.b)))
        for make in (dense_schur_matrix, sparse_schur_matrix)
            x, ok = schur_solve(make(p; nchunks=3), p.b, ones(length(p.b)), 0.4; nchunks=3)
            @test ok && norm(x - x_ref) <= 1e-10 * norm(x_ref)
        end
    finally
        BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = old
    end
end

@testset "Schur reduction in Float32 (M4)" begin
    p = standard_problem()
    n = length(p.b)
    x_ref = reference_solve(p, 0.3, ones(n))
    for make in (dense_schur_matrix, sparse_schur_matrix)
        H = make(p)
        B32 = H.B isa Matrix ? Float32.(H.B) :
              SparseBlockLinearSystem{Float32}(Float32.(H.B.data), H.B.indices, H.B.indicestransposed, H.B.rowblocksizes, H.B.columnblocksizes, H.B.m, H.B.n)
        H32 = SchurBlockMatrix(H.structure, B32, Float32.(H.C_data), Float32.(H.E_data))
        x, ok = schur_solve(H32, Float32.(p.b), ones(Float32, n), 0.3)
        @test ok && eltype(x) == Float32
        @test norm(Float64.(x) - x_ref) <= 1e-3 * norm(x_ref)
    end
    H2 = SMatrix{2, 2, Float32, 4}(1, 1, 1, 1 + 1f-6)
    @test !inv_sym_block(H2, 0f0, SVector{2, Float32}(1, 1)).success
    @test inv_sym_block(SMatrix{2, 2, Float64, 4}(1, 1, 1, 1 + 1e-6), 0.0, SVector{2, Float64}(1, 1)).success
end

@testset "fast_bAb equals the full quadratic form (B6)" begin
    for p in (standard_problem(), coupled_groups_problem())
        n = length(p.b)
        dx = randn(Random.Xoshiro(5), n)
        expected = dot(dx, p.H * dx)
        for kind in (:dense, :sparse)
            H = kind === :dense ? dense_schur_matrix(p) : sparse_schur_matrix(p)
            @test BlockLinearSystems.fast_bAb(H, dx) ≈ expected rtol = 1e-12
            BlockLinearSystems.fast_bAb(H, dx)
            kind === :sparse && @test (@allocated BlockLinearSystems.fast_bAb(H, dx)) == 0
        end
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

@testset "retained-retained couplings without a shared eliminated variable (B4)" begin
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

end
