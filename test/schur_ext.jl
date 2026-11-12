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
    for kind in (:dense, :sparse), solver in (:auto, :ldl), lambda in (0.0, 0.3)
        @testset "$kind, $solver, lambda=$lambda" begin
            ls = make_schur_ls(kind, p)
            SecondOrderOptimization.dampen!(ls, lambda)
            @test SecondOrderOptimization.solve!(ls, solver) == SecondOrderOptimization.Success
            x_ref = reference_solve(p, lambda, ones(n))
            @test norm(ls.x - x_ref) <= 1e-10 * norm(x_ref)
        end
    end
end
