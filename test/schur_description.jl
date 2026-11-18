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
