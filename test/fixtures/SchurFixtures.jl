# Schur test problems: SchurTestProblem, its builders, the named problems and the reference and kernel solves.
# Loaded once per session into Main by fixtures/load.jl; no testsets.
module SchurFixtures

using Random, LinearAlgebra, SparseArrays, BlockLinearSystems

export SchurTestProblem, nr, schur_structure, fill_ce!, dense_schur_hessian, sparse_schur_hessian, reference_solve, standard_problem, coupled_groups_problem, mixed_dims_problem, schur_solve, gauge_problem

"""
    SchurTestProblem

A random, strictly diagonally dominant (hence positive definite) bipartite system with retained
variables first and eliminated variables second, in full dense form plus everything needed to build
`SchurBlockHessian` objects from it.
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

function dense_schur_hessian(p::SchurTestProblem; nchunks=1)
    structure = schur_structure(p; nchunks)
    n_r = nr(p)
    R = p.H[1:n_r, 1:n_r]
    E_data = zeros(sum(abs2, p.e_dims))
    C_data = zeros(structure.coffsets[end] + Int(structure.r_dims[structure.entry_retained[end]]) * Int(structure.e_dims[end]) - 1)
    fill_ce!(p, structure, E_data, C_data)
    return SchurBlockHessian(structure, R, E_data, C_data)
end

function sparse_schur_hessian(p::SchurTestProblem; nchunks=1)
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
    return SchurBlockHessian(structure, R, E_data, C_data)
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

"""
    schur_solve(H, b, scaling, lambda; nchunks=1) -> (x, success)

Solve `(H + lambda * diag(scaling)) x = b` with the public Schur kernels only: reduce, dense or sparse
Cholesky of the reduced system, back-substitution.
"""
function schur_solve(H::SchurBlockHessian{T}, b, scaling, lambda; nchunks=1) where T
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
    backsubstitute!(view(x, nr+1:n), x_r, st, H.C_data, ws)
    return (x, true)
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

end # module SchurFixtures
