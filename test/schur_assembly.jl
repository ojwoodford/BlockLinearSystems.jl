using Random, LinearAlgebra, SparseArrays, StaticArrays, Static, HybridArrays, Test, BlockLinearSystems

# Sized views into SchurBlockMatrix storage (E5). Uses SchurTestProblem and the matrix builders of schur_problems.jl.

# Fill a zeroed copy of H through the views only, from the dense Hessian of p
function fill_through_views!(Hv::SchurBlockMatrix, p::SchurTestProblem)
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

@testset "Assembly views into SchurBlockMatrix storage (E5)" begin
    for p in (standard_problem(), coupled_groups_problem(), mixed_dims_problem()), make in (dense_schur_matrix, sparse_schur_matrix)
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
    for make in (dense_schur_matrix, sparse_schur_matrix)
        H = make(p)
        st = H.structure
        e, c = 1, 1                                   # eliminated block 1 is 3-dimensional, coupled to retained block 1 (6)
        @test eliminated_block(H, e, static(3), static(3)) isa SizedMatrix{3, 3, Float64}
        @test eliminated_block(H, e, Val(3), Val(3)) isa SizedMatrix{3, 3, Float64}
        @test coupling_block(H, c, static(3), static(6)) isa SizedMatrix{3, 6, Float64}
        @test coupling_block(H, c, static(3), 6) isa HybridArray
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
