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
    r_dims::Vector{Int}
    e_dims::Vector{Int}
    couplings::Vector{Vector{Int}}
    rr_pairs::Vector{Tuple{Int, Int}}   # retained-retained block couplings (r > r), beyond the diagonal
end

function SchurTestProblem(rng, r_dims, e_dims, couplings; rr_pairs=Tuple{Int, Int}[])
    nr, ne = sum(r_dims), sum(e_dims)
    n = nr + ne
    roff = cumsum([1; r_dims[1:end-1]])
    eoff = nr .+ cumsum([1; e_dims[1:end-1]])
    H = zeros(n, n)
    function fillblock!(r0, nrow, c0, ncol)
        H[r0:r0+nrow-1, c0:c0+ncol-1] .= 2 .* rand(rng, nrow, ncol) .- 1
    end
    for (r, d) in enumerate(r_dims)
        fillblock!(roff[r], d, roff[r], d)
    end
    for (e, d) in enumerate(e_dims)
        fillblock!(eoff[e], d, eoff[e], d)
    end
    for (e, coupled) in enumerate(couplings), r in coupled
        fillblock!(roff[r], r_dims[r], eoff[e], e_dims[e])
    end
    for (r1, r2) in rr_pairs
        fillblock!(roff[r1], r_dims[r1], roff[r2], r_dims[r2])
    end
    # Symmetrize from the upper triangle (block diagonals are symmetrized element-wise)
    H = Matrix(Symmetric(H, :U))
    # Make strictly diagonally dominant, but keep the conditioning moderate
    for i in 1:n
        H[i, i] = sum(abs, view(H, i, :)) - abs(H[i, i]) + 0.5 + rand(rng)
    end
    return SchurTestProblem(H, randn(rng, n), r_dims, e_dims, couplings, rr_pairs)
end

nr(p::SchurTestProblem) = sum(p.r_dims)

function schur_structure(p::SchurTestProblem; nchunks=1)
    return SchurStructure(p.r_dims, p.e_dims, p.couplings; nchunks)
end

function fill_ce!(p::SchurTestProblem, structure::SchurStructure, E_data, C_data)
    nr_ = nr(p)
    for e in eachindex(p.e_dims)
        de = p.e_dims[e]
        eo = structure.eoffsets[e]
        eoff = nr_ + structure.boffsets_e[e]
        E_data[eo:eo+de^2-1] .= vec(p.H[eoff:eoff+de-1, eoff:eoff+de-1])
        for c in structure.eliminated_ptr[e]:structure.eliminated_ptr[e+1]-1
            r = structure.entry_retained[c]
            dr = p.r_dims[r]
            ro = structure.boffsets_r[r]
            co = structure.coffsets[c]
            # Stored as de x dr column-major: C_data[co + (r-1)*de + i-1] = H[c_r, p_i]
            C_data[co:co+dr*de-1] .= vec(p.H[ro:ro+dr-1, eoff:eoff+de-1]')
        end
    end
    return nothing
end

function dense_schur_matrix(p::SchurTestProblem; nchunks=1)
    structure = schur_structure(p; nchunks)
    n_r = nr(p)
    R = p.H[1:n_r, 1:n_r]
    E_data = zeros(sum(abs2, p.e_dims))
    C_data = zeros(structure.coffsets[end] + Int(structure.r_dims[structure.entry_retained[end]]) * Int(structure.e_dims[end]) - 1)
    fill_ce!(p, structure, E_data, C_data)
    return SchurBlockMatrix(structure, R, E_data, C_data)
end

function sparse_schur_matrix(p::SchurTestProblem; nchunks=1)
    structure = schur_structure(p; nchunks)
    pattern = build_schur_pattern(structure)
    coords = Tuple{Int, Int}[]
    cb = pattern.block_colptr
    for r2 in 1:structure.nrblocks, k in cb[r2]:cb[r2+1]-1
        r1 = pattern.block_rowval[k]       # r1 <= r2 in the upper-triangle pattern
        push!(coords, (r2, r1))            # store lower-triangle block (row = r2, col = r1)
    end
    append!(coords, p.rr_pairs)
    unique!(coords)
    R = SparseBlockLinearSystem{Float64}(coords, p.r_dims)
    roff = structure.boffsets_r
    for (r1, r2) in coords
        block(R, r1, r2) .= p.H[roff[r1]:roff[r1]+p.r_dims[r1]-1, roff[r2]:roff[r2]+p.r_dims[r2]-1]
    end
    E_data = zeros(sum(abs2, p.e_dims))
    C_data = zeros(structure.coffsets[end] + Int(structure.r_dims[structure.entry_retained[end]]) * Int(structure.e_dims[end]) - 1)
    fill_ce!(p, structure, E_data, C_data)
    return SchurBlockMatrix(structure, R, E_data, C_data)
end

"Reference solution: dense Cholesky of the full system with `lambda * diag(damping)` added to the diagonal."
function reference_solve(p::SchurTestProblem, lambda, damping)
    Hd = p.H + lambda * Diagonal(damping)
    return cholesky(Symmetric(Hd)) \ p.b
end

function standard_problem(seed=1; rr_pairs=Tuple{Int, Int}[])
    rng = Random.Xoshiro(seed)
    r_dims = [6, 6, 6, 6]
    e_dims = [3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
    # Retained blocks 1-2 and 3-4 are coupled to eliminated blocks together; 2-3 are linked through a few eliminated blocks
    couplings = [
        [1, 2], [1, 2], [1, 2, 3], [2, 3], [3, 4], [3, 4], [1, 4],
        [1, 2], [2, 3], [3, 4], [1, 2, 3, 4], [4],
        [1], [2], [3], [4], [1, 2], [3, 4], [2, 3], [1, 4],
    ]
    return SchurTestProblem(rng, r_dims, e_dims, couplings; rr_pairs)
end

# Two groups of retained blocks ({1, 2} and {3, 4}) that share no eliminated block with each other, coupled only by
# direct retained-retained blocks (odometry / loop-closure style), which have no eliminated variable.
function coupled_groups_problem(seed=3)
    rng = Random.Xoshiro(seed)
    r_dims = [6, 6, 6, 6]
    e_dims = [3, 3, 3, 3, 2, 2, 1, 1]
    couplings = [[1, 2], [1, 2], [3, 4], [3, 4], [1], [4], [2, 1], [3, 4]]
    return SchurTestProblem(rng, r_dims, e_dims, couplings; rr_pairs=[(3, 1), (4, 2), (2, 1)])
end

# Eliminated blocks of every size class: tiny closed forms (1-3), the generic static tier (4..D_STATIC)
# and the dynamic tier above D_STATIC
function mixed_dims_problem(seed=11)
    rng = Random.Xoshiro(seed)
    r_dims = [6, 3, 9, 6]
    e_dims = [12, 12, 9, 9, 8, 8, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1]
    coupled = 1:4
    couplings = [sort(unique([rand(rng, coupled) for _ in 1:3])) for _ in e_dims]
    return SchurTestProblem(rng, r_dims, e_dims, couplings)
end


using Polyester

"""
    schur_solve(H, b, scaling, lambda; nchunks=1) -> (x, success)

Solve `(H + lambda * diag(scaling)) x = b` with the public Schur kernels only: reduce, dense or sparse
Cholesky of the reduced system, back-substitution.
"""
function schur_solve(H::SchurBlockMatrix{T}, b, scaling, lambda; nchunks=1) where T
    st = H.structure
    nr = st.nr
    n = length(b)
    b_r, b_e = view(b, 1:nr), view(b, nr+1:n)
    sc_r, sc_e = view(scaling, 1:nr), view(scaling, nr+1:n)
    ws = SchurReductionWorkspace{T}(st, nchunks)
    g = zeros(T, nr)
    lambda = T(lambda)
    if H.R isa Matrix
        S = zeros(T, nr, nr)
        reset_schur_dense!(S, H.R, sc_r, lambda, ws)
        reduce_schur_dense!(S, g, b_r, b_e, sc_e, lambda, st, H.E_data, H.C_data, ws) || return (zeros(T, n), false)
    else
        pattern = build_schur_pattern(st, T; extra_blocks=retained_block_pairs(H.R))
        S = pattern.S_sparse
        reset_schur_sparse!(S.nzval, H.R.data, build_r_to_s_map(H.R, pattern, st.boffsets_r), pattern.diag_inds, sc_r, lambda, ws)
        reduce_schur_sparse!(S, g, b_r, b_e, sc_e, lambda, st, H.E_data, H.C_data,
                             pattern.block_colptr, pattern.block_rowval, pattern.block_row_offsets, pattern.col_heights, ws) || return (zeros(T, n), false)
    end
    x = zeros(T, n)
    x_r = T.(cholesky(Symmetric(S, :U)) \ g)   # CHOLMOD may return Float64 for Float32 input
    x[1:nr] .= x_r
    backsubstitute!(view(x, nr+1:n), x_r, st, H.C_data, ws.E_inv_data, ws.z_data)
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
    for de in (1, 3, 4, 5, 7, 8, 9, 10, 20, 255)
        q = SchurTestProblem(Random.Xoshiro(de), [6, 6], [de, de], [[1, 2], [1]])
        for make in (dense_schur_matrix, sparse_schur_matrix)
            x, ok = schur_solve(make(q), q.b, ones(length(q.b)), 0.2)
            x_ref = reference_solve(q, 0.2, ones(length(q.b)))
            @test ok && norm(x - x_ref) <= 1e-8 * norm(x_ref)
        end
    end
end

# A gauge-deficient problem: every coupling constrains only the *difference* x_retained[1:3] - x_eliminated, so
# translating all retained and eliminated blocks together changes nothing and H is singular along that direction.
# Dimensions 4:6 of the retained blocks only have a tiny prior, which makes H ill conditioned (and rank deficient without it).
function gauge_problem(seed=21; prior=1e-8)
    rng = Random.Xoshiro(seed)
    r_dims = [6, 6, 6, 6]
    e_dims = [3, 3, 3, 3, 3, 3]
    couplings = [[1, 2], [1, 2, 3], [2, 3, 4], [3, 4], [1, 4], [1, 2, 3, 4]]
    nr, ne = sum(r_dims), sum(e_dims)
    H = zeros(nr + ne, nr + ne)
    for (e, coupled) in enumerate(couplings), r in coupled
        w = 0.5 + rand(rng)
        ri = 6 * (r - 1) .+ (1:3)           # translation dimensions of the retained block
        ei = nr + 3 * (e - 1) .+ (1:3)       # dimensions of the eliminated block
        H[ri, ri] .+= w * I(3); H[ei, ei] .+= w * I(3)
        H[ri, ei] .-= w * I(3); H[ei, ri] .-= w * I(3)
    end
    for r in 1:4
        ri = 6 * (r - 1) .+ (4:6)
        H[ri, ri] .+= prior * I(3)
    end
    return SchurTestProblem(H, randn(rng, nr + ne), r_dims, e_dims, couplings, Tuple{Int, Int}[])
end

@testset "Schur solve on a gauge-deficient, ill-conditioned problem: relative residual (T1)" begin
    p = gauge_problem()
    n = length(p.b)
    @test cond(p.H) > 1e7                        # ill conditioned: a forward-error check would be meaningless
    ev = eigvals(Symmetric(p.H))
    @test ev[1] > -1e-12 && ev[1] < 1e-12        # exactly singular along the gauge direction
    for make in (dense_schur_matrix, sparse_schur_matrix), lambda in (1e-6, 1e-3, 0.1)
        scaling = ones(n)
        x, ok = schur_solve(make(p), p.b, scaling, lambda)
        @test ok
        Hd = p.H + lambda * Diagonal(scaling)
        # backward-stable solve: the residual relative to the data does not grow with the condition number
        rel_res = norm(Hd * x - p.b) / (opnorm(Hd) * norm(x) + norm(p.b))
        @test rel_res <= 1e-13
    end
end

@testset "Well-conditioned test problems (T1)" begin
    for p in (standard_problem(), coupled_groups_problem(), mixed_dims_problem())
        @test cond(p.H) <= 1e4
        n = length(p.b)
        nr_ = nr(p)
        for make in (dense_schur_matrix, sparse_schur_matrix), lambda in (0.0, 0.3)
            x, ok = schur_solve(make(p), p.b, ones(n), lambda)
            x_ref = reference_solve(p, lambda, ones(n))
            @test ok
            # both the retained and the eliminated parts match
            @test norm(x[1:nr_] - x_ref[1:nr_]) <= 1e-10 * norm(x_ref[1:nr_])
            @test norm(x[nr_+1:end] - x_ref[nr_+1:end]) <= 1e-10 * norm(x_ref[nr_+1:end])
        end
    end
end

@testset "copy_structure(::SchurBlockMatrix) shares nothing mutable (B1)" begin
    for make in (dense_schur_matrix, sparse_schur_matrix)
        H = make(standard_problem())
        T = copy_structure(H)
        @test T.structure === H.structure
        @test T.E_data !== H.E_data && T.C_data !== H.C_data && T.R !== H.R
        @test all(iszero, T.E_data) && all(iszero, T.C_data)
        before = copy(H.E_data)
        T.E_data .= 1
        @test H.E_data == before
        T.R isa Matrix ? (T.R .= 7) : (T.R.data .= 7)
        @test !(H.R isa Matrix ? all(==(7), H.R) : all(==(7), H.R.data))
    end
    ws = SchurReductionWorkspace{Float64}(schur_structure(standard_problem()), 2)
    tw = trial_workspace(ws)
    @test tw.E_inv_data !== ws.E_inv_data && tw.z_data !== ws.z_data && tw.failed_block !== ws.failed_block
end

@testset "Singular eliminated blocks are reported, kernels do not allocate (B11, M5)" begin
    for de in (2, 6, 12)   # closed form, generic static tier and dynamic tier
        rng = Random.Xoshiro(de)
        q = SchurTestProblem(rng, [6, 6], [de, de], [[1, 2], [1, 2]])
        H = dense_schur_matrix(q)
        eo = H.structure.eoffsets[2]
        v = randn(rng, de)
        H.E_data[eo:eo+de^2-1] .= vec(v * v')
        x, ok = schur_solve(H, q.b, ones(length(q.b)), 0.0)
        @test !ok
        ws = SchurReductionWorkspace{Float64}(H.structure, 1)
        S = copy(H.R)
        reset_schur_dense!(S, H.R, ones(12), 0.0, ws)
        @test !reduce_schur_dense!(S, zeros(12), q.b[1:12], q.b[13:end], ones(2de), 0.0, H.structure, H.E_data, H.C_data, ws)
        @test ws.failed_block[] == 2
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
        if H.R isa Matrix
            S = zeros(12, 12)
            run1 = () -> reduce_schur_dense!(S, g, view(q.b, 1:12), view(q.b, 13:n), ones(n - 12), 0.3, st, H.E_data, H.C_data, ws)
        else
            pattern = build_schur_pattern(st; extra_blocks=retained_block_pairs(H.R))
            S = pattern.S_sparse
            run1 = () -> reduce_schur_sparse!(S, g, view(q.b, 1:12), view(q.b, 13:n), ones(n - 12), 0.3, st, H.E_data, H.C_data,
                pattern.block_colptr, pattern.block_rowval, pattern.block_row_offsets, pattern.col_heights, ws)
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
        @test reduce(vcat, collect.(ws.col_chunks)) == collect(1:st.nrblocks)
        # the retained-block-major index covers every coupling exactly once, in ascending eliminated block order
        @test sort(ws.retained_entry) == collect(1:length(st.entry_retained))
        for r in 1:st.nrblocks
            es = ws.retained_entry[ws.retained_ptr[r]:ws.retained_ptr[r+1]-1]
            @test all(st.entry_retained[es] .== r) && issorted(ws.entry_eliminated[es])
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
        B32 = H.R isa Matrix ? Float32.(H.R) :
              SparseBlockLinearSystem{Float32}(Float32.(H.R.data), H.R.indices, H.R.indicestransposed, H.R.rowblocksizes, H.R.columnblocksizes, H.R.m, H.R.n)
        H32 = SchurBlockMatrix(H.structure, B32, Float32.(H.E_data), Float32.(H.C_data))
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
            # Type stable (a captured `Type` was once widened to DataType on Julia 1.10, which made the
            # result `Any` and boxed it on every call) and allocation free, for both kinds of R
            @test @inferred(BlockLinearSystems.fast_bAb(H, dx)) isa Float64
            @test @inferred(BlockLinearSystems.fast_bAb(H, Float32.(dx))) isa Float64
            @test (@allocated BlockLinearSystems.fast_bAb(H, dx)) == 0
        end
    end
end

@testset "SchurStructure validates its inputs (B13)" begin
    ok = SchurStructure([6, 6], [3, 3], [[1, 2], [2]])
    @test ok.neblocks == 2
    @test_throws DimensionMismatch SchurStructure([6, 6], [3, 3], [[1, 2]])
    @test_throws ArgumentError SchurStructure([6, 6], [3, 256], [[1, 2], [2]])    # eliminated block over 255
    @test SchurStructure([6, 6], [4, 255], [[1, 2], [2]]).neblocks == 2            # any size up to 255 is supported
    @test_throws ArgumentError SchurStructure([6, 6], [3, 0], [[1, 2], [2]])      # empty eliminated block
    @test_throws ArgumentError SchurStructure([6, 0], [3, 3], [[1, 2], [2]])      # empty retained block
    @test_throws ArgumentError SchurStructure([6, 300], [3, 3], [[1, 2], [2]])    # retained block over 255
    @test_throws ArgumentError SchurStructure([6, 6], [3, 3], [[1, 2], [3]])      # retained block out of range
    @test_throws ArgumentError SchurStructure([6, 6], [3, 3], [[1, 0], [2]])
    @test_throws ArgumentError SchurStructure([6, 6], [3, 3], [[1, 2, 1], [2]])   # duplicate coupling
    @test_throws ArgumentError SchurStructure([6, 6], [3, 3], [[1, 2], [2]]; nchunks=0)
    # The same retained block may of course be coupled to different eliminated blocks
    @test SchurStructure([6, 6], [3, 3], [[1], [1]]).neblocks == 2
end

@testset "build_r_to_s_map: block-level and scalar-level searches agree (M6)" begin
    for p in (standard_problem(), coupled_groups_problem())
        H = sparse_schur_matrix(p)
        pattern = build_schur_pattern(H.structure; extra_blocks=retained_block_pairs(H.R))
        boffsets = H.structure.boffsets_r
        m_scalar = build_r_to_s_map(H.R, pattern.S_sparse, boffsets)
        m_block = build_r_to_s_map(H.R, pattern, boffsets)
        @test m_block == m_scalar
        # Mapped entries are unique and land on upper-triangle slots
        mapped = filter(>(0), m_block)
        @test allunique(mapped)
        S = pattern.S_sparse
        rows = rowvals(S)
        cols = [j for j in 1:size(S, 2) for _ in nzrange(S, j)]
        @test all(rows[k] <= cols[k] for k in mapped)
    end
    # Missing block gives an assertion rather than a wrong slot
    p = coupled_groups_problem()
    H = sparse_schur_matrix(p)
    plain = build_schur_pattern(H.structure)
    @test_throws AssertionError build_r_to_s_map(H.R, plain, H.structure.boffsets_r)
end

@testset "retained-retained couplings without a shared eliminated variable (B4)" begin
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
    H = sparse_schur_matrix(p)
    @test Set(retained_block_pairs(H.R)) ⊇ Set([(3, 1), (4, 2)])
    @test_throws AssertionError build_r_to_s_map(H.R, plain.S_sparse, structure.boffsets_r)
    # ... and builds one with them
    pattern = build_schur_pattern(structure; extra_blocks=retained_block_pairs(H.R))
    r_to_s = build_r_to_s_map(H.R, pattern.S_sparse, structure.boffsets_r)
    # every upper-triangle entry of the diagonal blocks (21 each) and all 36 entries of each off-diagonal block
    @test count(>(0), r_to_s) == 4 * 21 + 36 * length(retained_block_pairs(H.R))

end

@testset "build_schur_pattern scales to many couplings (B8) and can be reused" begin
    # Many retained blocks, each eliminated block coupled to a window of consecutive retained blocks
    nret, nelim = 3000, 6000
    coupled = [collect(r:min(r + 3, nret)) for r in (min.(1 .+ (0:nelim-1) .* nret .÷ nelim, nret))]
    structure = SchurStructure(fill(6, nret), fill(3, nelim), coupled)
    build_schur_pattern(structure)   # compile
    t = @elapsed pattern = build_schur_pattern(structure)
    @test t < 5
    @test nnz(pattern.block_pattern) == length(pattern.block_rowval)
    # Upper triangle with the diagonal; the sliding window gives bandwidth 3
    @test all(pattern.block_rowval[k] <= r for r in 1:nret for k in pattern.block_colptr[r]:pattern.block_colptr[r+1]-1)
    @test nnz(pattern.block_pattern) <= nret * 4

end

@testset "fast_bAb: any element type, and eliminated blocks of every size class (M13)" begin
    for p in (standard_problem(), mixed_dims_problem())
        n = length(p.b)
        dx = randn(Random.Xoshiro(8), n)
        expected = dot(dx, p.H * dx)
        for make in (dense_schur_matrix, sparse_schur_matrix)
            H = make(p)
            @test fast_bAb(H, dx) ≈ expected rtol = 1e-12
            @test fast_bAb(H, view(dx, 1:n)) ≈ expected rtol = 1e-12          # AbstractVector
            @test fast_bAb(H, Float32.(dx)) ≈ dot(Float32.(dx), p.H * Float32.(dx)) rtol = 1e-4
            @test fast_bAb(H, Float32.(dx)) isa Float64                      # promoted element type
            @test fast_bAb(H, ComplexF64.(dx)) ≈ expected rtol = 1e-12       # (real part of a real-quadratic form)
        end
    end
end

# Measured inside functions: @allocated at a testset's top level counts spurious boxing on Julia 1.10
allocs_fast_bAb(H, dx) = @allocated BlockLinearSystems.fast_bAb(H, dx)
allocs_zero!(H) = @allocated zero!(H)

@testset "SchurBlockMatrix carries an optional pattern (A1)" begin
    for make in (dense_schur_matrix, sparse_schur_matrix)
        H0 = make(standard_problem())
        st = H0.structure
        @test H0.pattern === nothing
        extra = H0.R isa Matrix ? () : retained_block_pairs(H0.R)
        pat = build_schur_pattern(st; extra_blocks=extra)
        H = SchurBlockMatrix(st, H0.R, H0.E_data, H0.C_data; pattern=pat)
        @test H.pattern === pat
        @test H.R === H0.R
        # positional (field-order) forms still work and carry no pattern
        @test SchurBlockMatrix(H0.R, H0.E_data, H0.C_data, st).pattern === nothing
        @test SchurBlockMatrix{Float64, typeof(H0.R)}(H0.R, H0.E_data, H0.C_data, st).pattern === nothing
        # extra entries on the named tuple are accepted
        @test SchurBlockMatrix(st, H0.R, H0.E_data, H0.C_data; pattern=merge(pat, (S_factor = 1,))).pattern.S_factor == 1

        # copy shares the pattern, nothing mutable
        Tc = copy_structure(H)
        @test Tc.pattern === H.pattern
        @test Tc.E_data !== H.E_data && Tc.C_data !== H.C_data && Tc.R !== H.R

        # wrong size
        bad = (S_sparse = spzeros(st.nr + 1, st.nr + 1),)
        @test_throws ArgumentError SchurBlockMatrix(st, H0.R, H0.E_data, H0.C_data; pattern=bad)

        # concrete types and allocation-free hot paths with a pattern attached
        @test isconcretetype(typeof(H))
        @test isconcretetype(fieldtype(typeof(H), :R))
        @test isconcretetype(fieldtype(typeof(H), :E_data)) && isconcretetype(fieldtype(typeof(H), :C_data))
        dx = randn(st.nr + sum(st.e_dims))
        allocs_fast_bAb(H, dx); allocs_zero!(H)   # compile
        @test allocs_fast_bAb(H, dx) == 0
        @test allocs_zero!(H) == 0
        zero!(H)
        @test all(iszero, H.E_data) && all(iszero, H.C_data)
    end
end
