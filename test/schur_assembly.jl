using Random, LinearAlgebra, SparseArrays, StaticArrays, Static, HybridArrays, Test, BlockLinearSystems
using BlockLinearSystems: validblock   # internal names
include("fixtures/load.jl")
using Main.SchurFixtures

# Sized views into SchurBlockHessian storage (E5). Uses SchurTestProblem and the matrix builders of fixtures/SchurFixtures.jl.

# Fill a zeroed copy of H through the views only, from the dense Hessian of p
function fill_through_views!(Hv::SchurBlockHessian, p::SchurTestProblem)
    st = Hv.structure
    n_r = st.nr
    rrange(r) = st.boffsets_r[r] .+ (0:Int(st.r_dims[r])-1)
    erange(e) = n_r .+ st.boffsets_e[e] .+ (0:Int(st.e_dims[e])-1)
    for e in 1:st.neblocks
        eliminated_block(Hv, e) .= p.H[erange(e), erange(e)]
        for c in st.eliminated_ptr[e]:st.eliminated_ptr[e+1]-1
            coupling_block(Hv, c) .= p.H[erange(e), rrange(st.entry_retained[c])]
        end
    end
    if Hv.R isa Matrix
        for j in 1:st.nrblocks, i in 1:st.nrblocks
            retained_block(Hv, i, j) .= p.H[rrange(i), rrange(j)]
        end
    else
        for j in 1:st.nrblocks, i in j:st.nrblocks
            validblock(Hv.R, i, j) && (retained_block(Hv, i, j) .= p.H[rrange(i), rrange(j)])
        end
    end
    return Hv
end

sab_static(H, e, c) = (eliminated_block(H, e, static(3), static(3)) .+= 1; coupling_block(H, c, static(3), static(6)) .+= 1;
                       retained_block(H, 1, 1, static(6), static(6)) .+= 1; nothing)
sab_dynamic(H, e, c) = (eliminated_block(H, e) .+= 1; coupling_block(H, c) .+= 1; retained_block(H, 2, 1) .+= 1; nothing)
allocs_sab_static(H, e, c) = @allocated sab_static(H, e, c)
allocs_sab_dynamic(H, e, c) = @allocated sab_dynamic(H, e, c)

@testset "Assembly views into SchurBlockHessian storage (E5)" begin
    for p in (standard_problem(), coupled_groups_problem(), mixed_dims_problem()), make in (dense_schur_hessian, sparse_schur_hessian)
        H = make(p)
        st = H.structure
        Hv = fill_through_views!(copy_structure(H), p)
        @test Hv.E_data == H.E_data
        @test Hv.C_data == H.C_data
        @test Hv.R isa Matrix ? Hv.R == H.R : Hv.R.data == H.R.data
        # Views alias the flat storage at the documented offsets
        for e in 1:st.neblocks
            de = Int(st.e_dims[e])
            @test size(eliminated_block(H, e)) == (de, de)
            @test vec(eliminated_block(H, e)) == H.E_data[st.eoffsets[e] .+ (0:de^2-1)]
        end
        for c in eachindex(st.entry_retained)
            e = searchsortedlast(st.eliminated_ptr, c)
            @test coupling_entry(st, e, st.entry_retained[c]) == c
            dr = Int(st.r_dims[st.entry_retained[c]])
            @test size(coupling_block(H, c)) == (Int(st.e_dims[e]), dr)
            @test vec(coupling_block(H, c)) == H.C_data[st.coffsets[c] .+ (0:Int(st.e_dims[e])*dr-1)]
        end
        # Errors
        @test_throws BoundsError eliminated_block(H, st.neblocks + 1)
        @test_throws BoundsError coupling_block(H, length(st.entry_retained) + 1)
        @test_throws DimensionMismatch eliminated_block(H, 1, 2, 2)
        @test_throws DimensionMismatch coupling_block(H, 1, Int(st.e_dims[1]), 1)
        @test_throws ArgumentError coupling_entry(st, 1, findfirst(r -> !(r in p.couplings[1]), 1:st.nrblocks))
        @test_throws BoundsError retained_block(H, st.nrblocks + 1, 1)
        @test_throws DimensionMismatch retained_block(H, 1, 1, 5, 6)
        H.R isa Matrix || @test_throws ArgumentError retained_block(H, 1, 3)   # not stored (upper triangle)
    end
    # Static sizes give statically sized views; all forms are type stable and do not allocate
    p = standard_problem()
    for make in (dense_schur_hessian, sparse_schur_hessian)
        H = make(p)
        st = H.structure
        e, c = 1, 1                                   # eliminated block 1 is 3-dimensional, coupled to retained block 1 (6)
        @test eliminated_block(H, e, static(3), static(3)) isa SizedMatrix{3, 3, Float64}
        @test eliminated_block(H, e, Val(3), Val(3)) isa SizedMatrix{3, 3, Float64}
        @test coupling_block(H, c, static(3), static(6)) isa SizedMatrix{3, 6, Float64}
        @test coupling_block(H, c, static(3), 6) isa HybridArray
        @test coupling_block(H, c, 3, static(6)) isa HybridArray
        @test coupling_block(H, c, static(3), 6) == coupling_block(H, c, 3, static(6)) == coupling_block(H, c)
        @test eliminated_block(H, e, 3, static(3)) == eliminated_block(H, e)
        @test retained_block(H, 1, 1, static(6), static(6)) isa SizedMatrix{6, 6, Float64}
        @inferred eliminated_block(H, e)
        @inferred coupling_block(H, c, static(3), static(6))
        @inferred retained_block(H, 2, 1)
        @inferred retained_block(H, 2, 1, static(6), static(6))
        before = copy(H.E_data)
        eliminated_block(H, e, static(3), static(3)) .= SMatrix{3, 3}(1:9)
        @test H.E_data[1:9] == 1:9 && H.E_data[10:end] == before[10:end]
        sab_static(H, e, c); sab_dynamic(H, e, c)
        @test allocs_sab_static(H, e, c) == 0
        @test allocs_sab_dynamic(H, e, c) == 0
    end
end

# Filling by block: addblock! (EL3)

# Scalar range of layout block b of a fixture problem (retained blocks, then eliminated ones), static or run-time
function fixture_range(st::SchurStructure, b::Int, static_size::Bool)
    first = b <= st.nrblocks ? st.boffsets_r[b] : st.nr + st.boffsets_e[b - st.nrblocks]
    d = b <= st.nrblocks ? Int(st.r_dims[b]) : Int(st.e_dims[b - st.nrblocks])
    return static_size ? blockrange(first, static(d)) : blockrange(first, d)
end

# The pairs a system has storage for: each block with itself, each coupling, and the stored retained pairs
function stored_pairs(H::SchurBlockHessian, rr_pairs)
    st = H.structure
    nr = st.nrblocks
    pairs = [(b, b) for b in 1:nr + st.neblocks]
    for e in 1:st.neblocks, c in st.eliminated_ptr[e]:st.eliminated_ptr[e+1]-1
        push!(pairs, (st.entry_retained[c], nr + e))
    end
    pat = build_schur_pattern(st; extra_blocks=rr_pairs).block_pattern
    for j in 1:nr, k in nzrange(pat, j)
        i = rowvals(pat)[k]
        i < j && push!(pairs, (i, j))
    end
    return pairs
end

# Today's fill, through the views, in the orientation each region stores: the upper triangle of a dense R, the lower
# of a sparse one, E_e, and C_er = H[r, e]'
function fill_by_views!(H::SchurBlockHessian, B, pairs)
    st = H.structure
    nr = st.nrblocks
    rg(b) = fixture_range(st, b, false)
    for (i, j) in pairs
        a, b = minmax(i, j)
        if a == b
            a <= nr ? (retained_block(H, a, a) .+= B[rg(a), rg(a)]) : (eliminated_block(H, a - nr) .+= B[rg(a), rg(a)])
        elseif b <= nr
            H.R isa Matrix ? (retained_block(H, a, b) .+= B[rg(a), rg(b)]) : (retained_block(H, b, a) .+= B[rg(b), rg(a)])
        else
            coupling_block(H, coupling_entry(st, b - nr, a)) .+= B[rg(a), rg(b)]'
        end
    end
    return H
end

function fill_by_addblock!(H::SchurBlockHessian, B, pairs, rng, static_size::Bool)
    st = H.structure
    for (i, j) in pairs
        a, b = rand(rng, Bool) ? (i, j) : (j, i)   # either order
        addblock!(H, a, b, B, fixture_range(st, a, static_size), fixture_range(st, b, static_size))
    end
    return H
end

same_values(A::SchurBlockHessian, B::SchurBlockHessian) =
    (A.R isa Matrix ? A.R == B.R : A.R.data == B.R.data) && A.E_data == B.E_data && A.C_data == B.C_data

alloc_addblock(H, i, j, B, rows, cols) = (addblock!(H, i, j, B, rows, cols); @allocated addblock!(H, i, j, B, rows, cols))

@testset "addblock! (EL3)" begin
    rng = Xoshiro(17)
    problems = (("standard with retained pairs", standard_problem(4; rr_pairs=[(2, 1), (4, 1)])),
                ("coupled groups", coupled_groups_problem()), ("mixed dimensions", mixed_dims_problem()))
    # Static ranges on the first problem only: each block size compiles its own loops
    @testset "$name, $kind, $(static_size ? "static" : "run-time") ranges" for (name, p) in problems, kind in (:dense, :sparse), static_size in (false, true)
        static_size && name != first(first(problems)) && continue
        proto = kind === :dense ? dense_schur_hessian(p) : sparse_schur_hessian(p)
        pairs = stored_pairs(proto, p.rr_pairs)
        ref = fill_by_views!(copy_structure(proto), p.H, pairs)
        H = fill_by_addblock!(copy_structure(proto), p.H, pairs, rng, static_size)
        @test same_values(H, ref)
        # The block of B read is the one in the orientation the system stores, whichever order the call names the
        # pair in: the same system from a B that is not symmetric
        Bn = randn(rng, size(p.H))
        H1, H2 = copy_structure(proto), copy_structure(proto)
        st = proto.structure
        for (i, j) in pairs
            ri, rj = fixture_range(st, i, static_size), fixture_range(st, j, static_size)
            addblock!(H1, i, j, Bn, ri, rj)
            addblock!(H2, j, i, Bn, rj, ri)
        end
        @test same_values(H1, H2)
        # Every region: allocation free and type stable
        nr = st.nrblocks
        e = 1; r = st.entry_retained[st.eliminated_ptr[1]]
        regions = [(1, 1), (nr + 1, nr + 1), (r, nr + e), (nr + e, r)]
        rr = [(i, j) for (i, j) in pairs if i != j && i <= nr && j <= nr]
        isempty(rr) || append!(regions, [rr[1], reverse(rr[1])])
        for (i, j) in regions
            ri, rj = fixture_range(st, i, static_size), fixture_range(st, j, static_size)
            @test alloc_addblock(H, i, j, p.H, ri, rj) == 0
            @test @inferred(addblock!(H, i, j, p.H, ri, rj)) === H
        end
    end

    @testset "Errors" begin
        p = coupled_groups_problem()
        for kind in (:dense, :sparse)
            H = copy_structure(kind === :dense ? dense_schur_hessian(p) : sparse_schur_hessian(p))
            st = H.structure
            nr, n = st.nrblocks, st.nrblocks + st.neblocks
            rg(b) = fixture_range(st, b, false)
            # Two eliminated blocks
            @test_throws ArgumentError("Blocks $(nr + 1) and $(nr + 2) are both eliminated: the system has no storage for the pair") addblock!(H, nr + 1, nr + 2, p.H, rg(nr + 1), rg(nr + 2))
            # A retained and an eliminated block that share no row block: eliminated block 1 is coupled to 1 and 2 only
            @test_throws ArgumentError("Blocks 3 and $(nr + 1) share no row block: the system has no storage for the pair") addblock!(H, 3, nr + 1, p.H, rg(3), rg(nr + 1))
            @test_throws ArgumentError addblock!(H, nr + 1, 3, p.H, rg(nr + 1), rg(3))
            # Out of range, and a wrong size
            @test_throws BoundsError addblock!(H, n + 1, 1, p.H, rg(1), rg(1))
            @test_throws BoundsError addblock!(H, 0, 1, p.H, rg(1), rg(1))
            @test_throws DimensionMismatch("Block (1, $(nr + 1)) has size (6, 3), but (6, 2) was given") addblock!(H, 1, nr + 1, p.H, rg(1), 1:2)
            # Ranges outside B
            @test_throws BoundsError addblock!(H, 1, 1, zeros(3, 3), 1:6, 1:6)
        end
        # A sparse retained part has no storage for retained blocks that share no row block (3 and 2 in this problem)
        Hs = copy_structure(sparse_schur_hessian(p))
        @test !BlockLinearSystems.validblock(Hs.R, 3, 2)
        @test_throws ArgumentError("Blocks 2 and 3 share no row block: the system has no storage for the pair") addblock!(Hs, 2, 3, p.H, fixture_range(Hs.structure, 2, false), fixture_range(Hs.structure, 3, false))
    end
end
