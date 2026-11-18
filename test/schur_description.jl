using Test, BlockLinearSystems, Random, SparseArrays
include(joinpath(@__DIR__, "fixtures", "load.jl"))
using Main.SchurFixtures, Main.QRFixtures, Main.DescriptionFixtures

schur_fixture_problems() = [("standard", standard_problem(1)), ("standard with retained pairs", standard_problem(2; rr_pairs=[(2, 1), (4, 3)])),
                            ("coupled groups", coupled_groups_problem()), ("mixed dimensions", mixed_dims_problem()), ("gauge", gauge_problem())]
qr_fixture_problems() = filter(p -> !isempty(p.e_dims), sparse_reduced_problems(Float64))

@testset "The layout from a description (EL1)" begin
    rng = Xoshiro(7)
    @testset "Hessian path: $name, nchunks $nc" for (name, p) in schur_fixture_problems(), nc in (1, 3)
        d = description(p, rng)
        st = SchurStructure(d.blocksizes, d.rowblocks; eliminated=d.eliminated, nchunks=nc)
        st_old, old = old_hessian(d; nchunks=nc)
        @test same_fields(st, st_old; skip=NEW_FIELDS)
        @test blockorder(st) == old.perm
        @test st.retained_pairs == old.extra
        # The structure holds the retained pairs: its pattern is today's with them as extra blocks
        pat, pat_old = build_schur_pattern(st), build_schur_pattern(st_old; extra_blocks=old.extra)
        @test all(k -> isequal(pat[k], pat_old[k]), keys(pat_old))
        # extra_blocks adds to the structure's own pairs
        @test build_schur_pattern(st; extra_blocks=old.extra).block_pattern == pat_old.block_pattern
        # The accessors
        @test sort(blockorder(st)) == 1:length(d.blocksizes)
        @test nblocks(st) == length(d.blocksizes) == st.nrblocks + st.neblocks
        @test nretained(st) == st.nrblocks == count(!, d.eliminated)
        @test blocksizes(st) == [collect(st.r_dims); collect(st.e_dims)] == d.blocksizes[blockorder(st)]
        @test all(!, d.eliminated[blockorder(st)[1:nretained(st)]]) && all(d.eliminated[blockorder(st)[nretained(st)+1:end]])
    end

    @testset "Jacobian path: $(p.name), nchunks $nc" for p in qr_fixture_problems(), nc in (1, 2)
        d = description(p, rng; empty=(p.name == "free groups"))
        jst = SchurJacobianStructure(d.blocksizes, d.rowblocks, d.rowblocksizes; eliminated=d.eliminated, nchunks=nc)
        jst_old, old, table, first = old_jacobian(d; nchunks=nc)
        @test same_fields(jst, jst_old; skip=NEW_FIELDS)
        @test blockorder(jst) == old.perm
        @test jst.schur.retained_pairs == old.extra
        @test jst.row_table == table && jst.row_first == first && jst.row_nrows == d.rowblocksizes
        pos = invperm(old.perm)
        @test all(k -> jst.row_blocks[jst.row_ptr[k]:jst.row_ptr[k+1]-1] == sort(pos[d.rowblocks[k]]), eachindex(d.rowblocks))
        @test SchurStructure(jst) === jst.schur
        @test blockorder(jst) == blockorder(jst.schur) && nretained(jst) == nretained(jst.schur)
        @test nblocks(jst) == nblocks(jst.schur) && blocksizes(jst) == blocksizes(jst.schur)
        # The Hessian path's structure of the same description is the Jacobian structure's own
        @test same_fields(SchurStructure(d.blocksizes, d.rowblocks; eliminated=d.eliminated, nchunks=nc), jst.schur)
    end

    @testset "Lazy row blocks, and the caller's vectors are not kept" begin
        d = description(standard_problem(3), Xoshiro(1))
        flat = reduce(vcat, d.rowblocks); ptr = cumsum([1; length.(d.rowblocks)])
        lazy = [view(flat, ptr[k]:ptr[k+1]-1) for k in eachindex(d.rowblocks)]
        st = SchurStructure(d.blocksizes, d.rowblocks; eliminated=d.eliminated)
        @test same_fields(SchurStructure(d.blocksizes, lazy; eliminated=d.eliminated), st)
        jst = SchurJacobianStructure(d.blocksizes, lazy, d.rowblocksizes; eliminated=d.eliminated)
        @test same_fields(jst, SchurJacobianStructure(d.blocksizes, d.rowblocks, d.rowblocksizes; eliminated=d.eliminated))
        snap, jsnap = deepcopy(st), deepcopy(jst)
        bs, el, rb, rbs = copy(d.blocksizes), copy(d.eliminated), deepcopy(d.rowblocks), copy(d.rowblocksizes)
        st2 = SchurStructure(bs, rb; eliminated=el)
        jst2 = SchurJacobianStructure(bs, rb, rbs; eliminated=el)
        fill!(bs, 1); el .= .!el; foreach(r -> fill!(r, 1), rb); fill!(rbs, 9); fill!(flat, 1)
        @test same_fields(st2, snap) && same_fields(jst2, jsnap) && same_fields(jst, jsnap)
        bo = blockorder(st); bo[1] = 0
        @test blockorder(st) == blockorder(snap)
    end

    @testset "Rejected up front" begin
        bs, el = [6, 3, 3], [false, true, true]
        S(rbs; kw...) = SchurStructure(get(kw, :bs, bs), rbs; eliminated=get(kw, :el, el))
        J(rbs, rows; kw...) = SchurJacobianStructure(get(kw, :bs, bs), rbs, rows; eliminated=get(kw, :el, el))
        good = [[1, 2], [3, 1]]
        @test_throws DimensionMismatch("eliminated has length 2, but there are 3 column blocks") S(good; el=[false, true])
        @test_throws DimensionMismatch("rowblocksizes has length 1, but there are 2 row blocks") J(good, [2])
        @test_throws ArgumentError("Block 2 has dimension 0; dimensions must be between 1 and 65535") S(good; bs=[6, 0, 3])
        @test_throws ArgumentError("Block 3 has dimension 65536; dimensions must be between 1 and 65535") S(good; bs=[6, 3, 65536])
        @test_throws ArgumentError("Row block 2 touches column block 4, but there are 3 column blocks") S([[1, 2], [4]])
        @test_throws ArgumentError("Row block 2 touches column block 0, but there are 3 column blocks") S([[1, 2], [0, 1]])
        @test_throws ArgumentError("Row block 1 lists column block 1 more than once") S([[1, 2, 1], [3]])
        @test_throws ArgumentError("Row block 2 touches two eliminated column blocks, 3 and 2; a row block may touch at most one") S([[1, 2], [3, 1, 2]])
        @test_throws ArgumentError("An elimination needs at least one retained and one eliminated column block; got 0 retained and 3 eliminated") S(good; el=[true, true, true])
        @test_throws ArgumentError("An elimination needs at least one retained and one eliminated column block; got 3 retained and 0 eliminated") S(good; el=[false, false, false])
        @test_throws ArgumentError("Row block 2 has 0 rows; every row block needs at least 1 row") J(good, [2, 0])
        @test_throws ArgumentError("Eliminated column block 3 is in no row block; eliminating a block needs at least one row") J([[1, 2]], [2])
        # The Hessian path allows an eliminated block in no row block (its blocks are then whatever the caller adds)
        @test nblocks(S([[1, 2]])) == 3
        # A row block touching no column block is allowed: on the Jacobian path its rows hold only the residual
        jst = J([[1, 2], Int[], [3, 1]], [2, 1, 3])
        @test jst.ntables == 3 && jst.free_rows == [1] && isempty(jst.entry_retained[jst.table_ptr[3]:jst.table_ptr[4]-1])
        # The keyword is required
        @test_throws UndefKeywordError SchurStructure(bs, good)
    end
end

# A system laid out, allocated and filled through the front door (EL1-EL3), reduced and solved through DampedLinearSolve,
# against today's route: the positional structure, storage built by hand, filled through the views by offset
module FrontDoorSolveTests

using Test, BlockLinearSystems, Random, SparseArrays, LinearAlgebra
using DampedLinearSolve: makehessiansystem, makejacobiansystem, dampen!, solve!, getb, getx, Success
using BlockLinearSystems: retained_block, eliminated_block, coupling_block, coupling_entry, table_residual_block
include(joinpath(@__DIR__, "fixtures", "load.jl"))
using Main.SchurFixtures, Main.QRFixtures, Main.DescriptionFixtures

# Layout ranges of a structure's blocks in the full vector
layout_range(st, b) = (o = computestartindices(blocksizes(st)); o[b]:o[b] + blocksizes(st)[b] - 1)

# The fixture's matrix in the layout order of `st`, for a description whose caller numbering is q (fixture block -> caller)
function layout_matrix(p::SchurTestProblem, q, st)
    nr = length(p.r_dims)
    fst = schur_structure(p)
    frange(i) = i <= nr ? (fst.boffsets_r[i]:fst.boffsets_r[i] + p.r_dims[i] - 1) :
                          (fst.nr + fst.boffsets_e[i - nr]:fst.nr + fst.boffsets_e[i - nr] + p.e_dims[i - nr] - 1)
    pos = invperm(blockorder(st))
    n = size(p.H, 1)
    perm = zeros(Int, n)   # layout index of each fixture index
    for i in 1:length(q)
        perm[frange(i)] .= layout_range(st, pos[q[i]])
    end
    A = zeros(n, n); A[perm, perm] .= p.H
    b = zeros(n); b[perm] .= p.b
    return A, b
end

function pairs_of(st, extra)
    nr = st.nrblocks
    pairs = [(b, b) for b in 1:nblocks(st)]
    for e in 1:st.neblocks, c in st.eliminated_ptr[e]:st.eliminated_ptr[e+1]-1
        push!(pairs, (st.entry_retained[c], nr + e))
    end
    pat = build_schur_pattern(st; extra_blocks=extra).block_pattern
    for j in 1:nr, k in nzrange(pat, j)
        rowvals(pat)[k] < j && push!(pairs, (rowvals(pat)[k], j))
    end
    return pairs
end

function solve_hessian(H, b, lambda, scaling)
    ls = makehessiansystem(H, :auto)
    getb(ls) .= b
    dampen!(ls, lambda, scaling)
    @test solve!(ls, :auto) == Success
    return copy(getx(ls))
end

@testset "A whole solve through the front door equals today's route (EL3)" begin
    rng = Xoshiro(31)
    @testset "Hessian path: $name, $kind" for (name, p) in (("coupled groups", coupled_groups_problem()), ("mixed dimensions", mixed_dims_problem())), kind in (:dense, :sparse)
        # A description in a shuffled caller numbering q: one row block per eliminated block with all its couplings,
        # and one per retained pair
        q, bs, el = caller_numbering(rng, p.r_dims, p.e_dims)
        nr = length(p.r_dims)
        rbs = [[q[nr + e]; q[c]] for (e, c) in enumerate(p.couplings)]
        append!(rbs, [[q[r1], q[r2]] for (r1, r2) in p.rr_pairs])
        d = Description(bs, el, rbs, ones(Int, length(rbs)))
        st = SchurStructure(d.blocksizes, d.rowblocks; eliminated=d.eliminated)
        st_old, old = old_hessian(d)
        A, b = layout_matrix(p, q, st)
        RT = kind === :dense ? Matrix{Float64} : SparseBlockLinearSystem{Float64}
        # The front door
        H = SchurBlockHessian{Float64, RT}(st)
        for (i, j) in pairs_of(st, ())
            addblock!(H, i, j, A, layout_range(st, i), layout_range(st, j))
        end
        # Today's route: storage by hand (as NLLSModel builds it), filled through the views
        E = zeros(sum(abs2, st_old.e_dims)); C = zeros(sum(Int(st_old.e_dims[e]) * Int(st_old.r_dims[st_old.entry_retained[c]]) for e in 1:st_old.neblocks for c in st_old.eliminated_ptr[e]:st_old.eliminated_ptr[e+1]-1))
        H0 = if kind === :dense
            SchurBlockHessian(st_old, zeros(st_old.nr, st_old.nr), E, C)
        else
            pat = build_schur_pattern(st_old; extra_blocks=old.extra)
            SchurBlockHessian(st_old, SparseBlockLinearSystem{Float64}(TransposedPattern(pat.block_pattern), st_old.r_dims, st_old.r_dims), E, C; pattern=pat)
        end
        for (i, j) in pairs_of(st_old, old.extra)
            a, c = minmax(i, j)
            ra, rc = layout_range(st, a), layout_range(st, c)
            if a == c
                a <= st_old.nrblocks ? (retained_block(H0, a, a) .+= A[ra, ra]) : (eliminated_block(H0, a - st_old.nrblocks) .+= A[ra, ra])
            elseif c <= st_old.nrblocks
                kind === :dense ? (retained_block(H0, a, c) .+= A[ra, rc]) : (retained_block(H0, c, a) .+= A[rc, ra])
            else
                coupling_block(H0, coupling_entry(st_old, c - st_old.nrblocks, a)) .+= A[ra, rc]'
            end
        end
        @test (kind === :dense ? H.R == H0.R : H.R.data == H0.R.data) && H.E_data == H0.E_data && H.C_data == H0.C_data
        scaling = abs.(diag(A))
        for lambda in (0.0, 0.3)
            x, x0 = solve_hessian(H, b, lambda, scaling), solve_hessian(H0, b, lambda, scaling)
            @test x == x0
            @test norm(A * x + lambda * scaling .* x - b) <= 1e-8 * norm(b)
        end
    end

    @testset "Jacobian path: $(p.name), $RT" for p in filter(p -> !isempty(p.e_dims), qr_problems(Float64)), RT in (Matrix{Float64}, SparseMatrixCSC{Float64, Int})
        d = description(p, rng; empty=true)
        jst = SchurJacobianStructure(d.blocksizes, d.rowblocks, d.rowblocksizes; eliminated=d.eliminated)
        jst_old, old, table, first = old_jacobian(d)
        J, J0 = SchurBlockJacobian{Float64, RT}(jst), SchurBlockJacobian{Float64, RT}(jst_old)
        pos = invperm(blockorder(jst)); bs = blocksizes(jst); nr = nretained(jst)
        for (k, rb) in enumerate(d.rowblocks)
            m = d.rowblocksizes[k]
            for c in rb
                j = pos[c]
                X = randn(rng, m, bs[j])
                block(J, k, blockentry(jst, k, j), m, bs[j]) .= X
                (j <= nr ? coupling_block(J0, coupling_entry(jst_old, table[k], j), first[k], m) : eliminated_block(J0, table[k], first[k], m)) .= X
            end
            r = randn(rng, m)
            residual_block(J, k) .= r
            table_residual_block(J0, table[k], first[k], m) .= r
        end
        @test J.data == J0.data
        n = sum(bs)
        g = BlockLinearSystems.jtr!(zeros(n), J)
        scaling = 0.5 .+ rand(rng, n)
        for lambda in (0.0, 0.3)
            xs = map((J, J0)) do JJ
                ls = makejacobiansystem(JJ, n, :auto)
                getb(ls) .= g
                dampen!(ls, lambda, scaling)
                @test solve!(ls, :auto) == Success
                copy(getx(ls))
            end
            @test xs[1] == xs[2]
        end
    end
end

end # module FrontDoorSolveTests
