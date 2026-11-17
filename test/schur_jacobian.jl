using Random, LinearAlgebra, SparseArrays, StaticArrays, Static, HybridArrays, Test, BlockLinearSystems
include("fixtures/load.jl")
using Main.QRFixtures

# Storage and assembly API of the tall (Jacobian) form, and the random test problems the QR elimination tests use.


@testset "SchurJacobianStructure: layout and validation (E1)" begin
    st = SchurStructure([6, 3, 6], [3, 3, 2], [[3, 1], [2], Int[]])
    jst = SchurJacobianStructure(st, [5, 4, 2]; free_rows=[3, 2], free_retained=[[3, 1], [2]])
    @test jst.schur === st
    @test jst.ntables == 5
    @test jst.table_nrows == [8, 7, 4, 3, 2]                    # m_e + d_e damping rows; free groups have none
    @test jst.table_ncols == [16, 7, 3, 13, 4]
    @test jst.table_top == [3, 3, 2, 0, 0]
    @test jst.table_offsets == cumsum([1; jst.table_nrows .* jst.table_ncols])
    # Coupling entries keep the SchurStructure's order (sorted by retained block), free entries follow, also sorted
    @test jst.entry_retained == [1, 3, 2, 1, 3, 2]
    @test jst.entry_retained[1:length(st.entry_retained)] == st.entry_retained
    @test jst.entry_col == [1, 7, 1, 1, 7, 1]                                    # column offsets inside each table
    @test jst.entry_table == [1, 1, 2, 4, 4, 5]
    @test jst.table_ptr == [1, 3, 4, 4, 6, 7]
    @test jst.reduced_offsets == cumsum([1, 5, 4, 2, 3, 2])
    @test coupling_entry(jst, 1, 3) == 2 && coupling_entry(jst, 4, 3) == 5 && coupling_entry(jst, 5, 2) == 6
    @test_throws ArgumentError coupling_entry(jst, 2, 1)
    @test_throws BoundsError coupling_entry(jst, 6, 1)
    # Retained-block-major index covers every entry once, ascending
    @test sort(jst.retained_entry) == collect(1:6)
    for r in 1:3
        us = jst.retained_entry[jst.retained_ptr[r]:jst.retained_ptr[r+1]-1]
        @test issorted(us) && all(jst.entry_retained[us] .== r)
    end
    @test jst.doffsets == [1, 37, 46, 82]
    @test reduce(vcat, collect.(jst.col_chunks)) == [1, 2, 3]
    @test length(SchurJacobianStructure(st, [5, 4, 2]; nchunks=3).col_chunks) == 3
    # The convenience constructor builds the SchurStructure too
    jst2 = SchurJacobianStructure([6, 3, 6], [3, 3, 2], [[3, 1], [2], Int[]], [5, 4, 2]; nchunks=2)
    @test jst2.schur.entry_retained == st.entry_retained && length(jst2.schur.chunks[1]) == 2 && length(jst2.col_chunks) == 2

    @test_throws DimensionMismatch SchurJacobianStructure(st, [5, 4])
    @test_throws ArgumentError SchurJacobianStructure(st, [5, 0, 2])                                     # no rows
    @test_throws DimensionMismatch SchurJacobianStructure(st, [5, 4, 2]; free_rows=[3], free_retained=[[1], [2]])
    @test_throws ArgumentError SchurJacobianStructure(st, [5, 4, 2]; free_rows=[0], free_retained=[[1]])
    @test_throws ArgumentError SchurJacobianStructure(st, [5, 4, 2]; free_rows=[3], free_retained=[[1, 4]])   # out of range
    @test_throws ArgumentError SchurJacobianStructure(st, [5, 4, 2]; free_rows=[3], free_retained=[[0]])
    @test_throws ArgumentError SchurJacobianStructure(st, [5, 4, 2]; free_rows=[3], free_retained=[[2, 1, 2]]) # duplicate
    @test_throws ArgumentError SchurJacobianStructure(st, [5, 4, 2]; nchunks=0)
    # The same retained block may appear in several free groups
    @test SchurJacobianStructure(st, [5, 4, 2]; free_rows=[3, 1], free_retained=[[1], [1]]).ntables == 5
end

@testset "SchurBlockJacobian storage, zero! and copy_structure (E1)" begin
    p = qr_problems(Float64)[3]
    J = p.J
    jst = J.structure
    @test length(J.data) == jst.table_offsets[end] - 1
    @test_throws DimensionMismatch SchurBlockJacobian{Float64, Matrix{Float64}}(zeros(3), jst)
    @test SchurBlockJacobian{Float64, Matrix{Float64}}(copy(J.data), jst).data == J.data
    @test SchurBlockJacobian{Float32, Matrix{Float32}}(jst).data isa Vector{Float32}
    C = copy_structure(J)
    @test C.structure === J.structure && C.data !== J.data && all(iszero, C.data) && length(C.data) == length(J.data)
    before = copy(J.data)
    C.data .= 1
    @test J.data == before
    @test zero!(C) === C && all(iszero, C.data)
    # The damping rows are zero in the raw storage
    for e in 1:jst.schur.neblocks
        M, de = jst.table_nrows[e], jst.table_top[e]
        T_e = reshape(J.data[jst.table_offsets[e]:jst.table_offsets[e+1]-1], M, :)
        @test all(iszero, T_e[M-de+1:M, :])
    end
    @test schur_jacobian_bytes(jst, Float64; workspace=false) == sizeof(J.data)
    @test schur_jacobian_bytes(jst, Float32; workspace=false) == sizeof(SchurBlockJacobian{Float32, Matrix{Float32}}(jst).data)
end

@testset "SchurBlockJacobian reduced storage type (R1)" begin
    for T in (Float64, Float32)
        p = qr_problems(T)[3]
        jst = p.J.structure
        st = jst.schur
        Jd = SchurBlockJacobian{T, Matrix{T}}(copy(p.J.data), jst)
        Js = SchurBlockJacobian{T, SparseMatrixCSC{T, Int}}(copy(p.J.data), jst)
        @test Jd isa SchurBlockJacobian{T, Matrix{T}} && Js isa SchurBlockJacobian{T, SparseMatrixCSC{T, Int}}
        @test SchurBlockJacobian{T, SparseMatrixCSC{T, Int}}(jst).data == zeros(T, length(p.J.data))
        # Only the two reduced storages that have kernels; no default
        @test_throws ArgumentError SchurBlockJacobian{T, SparseBlockLinearSystem{T}}(jst)
        @test_throws ArgumentError SchurBlockJacobian{T, Matrix{Float16}}(jst)
        @test_throws ArgumentError SchurBlockJacobian{T, SparseMatrixCSC{T, Int32}}(jst)
        @test_throws MethodError SchurBlockJacobian{T}(jst)
        @test_throws MethodError SchurBlockJacobian(copy(p.J.data), jst)
        # copy_structure and zero! keep the parameter
        for J in (Jd, Js)
            C = copy_structure(J)
            @test typeof(C) === typeof(J) && C.structure === J.structure && all(iszero, C.data)
            @test zero!(copy_structure(J)) isa typeof(J)
        end
        # The same tables reduce to the same J̃ᵀJ̃, whatever the parameter, dense and sparse
        sc, lambda = rand(Random.Xoshiro(2), T, sum(p.e_dims)) .+ T(0.5), T(0.3)
        wsd, wss = QRReductionWorkspace(Jd), QRReductionWorkspace(Js)
        @test reduce_qr!(wsd, Jd, sc, lambda) && reduce_qr!(wss, Js, sc, lambda)
        @test wsd.factored == wss.factored
        S, g = zeros(T, st.nr, st.nr), zeros(T, st.nr)
        reduced_normal_dense!(S, g, wsd)
        pat = build_schur_pattern(st, T; extra_blocks=free_block_pairs(jst))
        Ssp, gsp = copy(pat.S_sparse), zeros(T, st.nr)
        reduced_normal_sparse!(Ssp, gsp, wss, pat)
        tol = sqrt(eps(T)) * norm(triu(S))
        @test norm(triu(Matrix(Ssp)) - triu(S)) <= tol && norm(gsp - g) <= sqrt(eps(T)) * norm(g)
    end
end

# Linear index of entry (row, col) of table t in the flat storage
flat_index(jst, t, row, col) = jst.table_offsets[t] + (col - 1) * jst.table_nrows[t] + row - 1

fill_static!(J, u, e, t) = (coupling_block(J, u, 2, static(2), static(6)) .= 1; eliminated_block(J, e, 1, static(2), static(3)) .= 2;
                            residual_block(J, t, 3, static(2)) .= 3; nothing)
fill_dynamic!(J, u, e, t) = (coupling_block(J, u, 2, 2, 6) .= 1; eliminated_block(J, e, 1, 2) .= 2; residual_block(J, t, 3, 2) .= 3; nothing)
allocs_fill_static(J, u, e, t) = @allocated fill_static!(J, u, e, t)
allocs_fill_dynamic(J, u, e, t) = @allocated fill_dynamic!(J, u, e, t)

@testset "Assembly views: sizes, aliasing, bounds, static sizes (E1)" begin
    for T in (Float64, Float32)
        p = qr_problems(T)[3]          # has free groups
        J = p.J
        jst = J.structure
        st = jst.schur
        zero!(J)
        # Every view aliases the right memory: write through it, read the flat storage
        for t in 1:jst.ntables
            m = t <= st.neblocks ? jst.e_rows[t] : jst.free_rows[t - st.neblocks]
            for u in jst.table_ptr[t]:jst.table_ptr[t+1]-1
                dr = Int(st.r_dims[jst.entry_retained[u]])
                B = coupling_block(J, u)
                @test size(B) == (m, dr)
                B .= reshape(1:m*dr, m, dr) .+ 1000u
                @test all(J.data[flat_index(jst, t, i, jst.entry_col[u] + j - 1)] == i + (j - 1) * m + 1000u for i in 1:m, j in 1:dr)
                @test size(coupling_block(J, u, 2, m - 1)) == (m - 1, dr)
                @test coupling_block(J, u, 2, m - 1) == B[2:m, :]
            end
            if t <= st.neblocks
                de = Int(st.e_dims[t])
                E = eliminated_block(J, t)
                @test size(E) == (m, de)
                E .= -reshape(1:m*de, m, de)
                c0 = jst.table_ncols[t] - de
                @test all(J.data[flat_index(jst, t, i, c0 + j - 1)] == -(i + (j - 1) * m) for i in 1:m, j in 1:de)
                @test eliminated_block(J, t, m - 1, 2) == E[m-1:m, :]
            end
            rv = residual_block(J, t)
            @test size(rv) == (m,)
            rv .= 0.5 .* (1:m)
            @test all(J.data[flat_index(jst, t, i, jst.table_ncols[t])] == 0.5i for i in 1:m)
            @test residual_block(J, t, m - 1, 2) == rv[m-1:m]
        end
        # Nothing was written to the damping rows
        for e in 1:st.neblocks
            @test all(iszero(J.data[flat_index(jst, e, i, j)]) for i in jst.e_rows[e]+1:jst.table_nrows[e], j in 1:jst.table_ncols[e])
        end
        # Out of range: past the data rows (into the damping rows), wrong column count, unknown entry or table
        u = coupling_entry(jst, 1, 2)
        m1 = jst.e_rows[1]
        @test_throws BoundsError coupling_block(J, u, m1, 2)
        @test_throws BoundsError coupling_block(J, u, 0, 1)
        @test_throws DimensionMismatch coupling_block(J, u, 1, 2, 5)
        @test_throws BoundsError coupling_block(J, length(jst.entry_retained) + 1)
        @test_throws BoundsError eliminated_block(J, st.neblocks + 1)          # free tables have no eliminated block
        @test_throws BoundsError eliminated_block(J, 1, m1 - 1, static(3))
        @test_throws DimensionMismatch eliminated_block(J, 1, 1, 2, static(2))
        @test_throws BoundsError residual_block(J, jst.ntables + 1)
        @test_throws BoundsError residual_block(J, 1, m1, 2)
        # Static sizes give statically sized views, also via Val; run-time sizes are type stable too
        @test coupling_block(J, u, 1, static(2), static(6)) isa SizedMatrix{2, 6, T}
        @test coupling_block(J, u, 1, Val(2), Val(6)) isa SizedMatrix{2, 6, T}
        @test size(coupling_block(J, u, 1, static(2))) == (2, 6)
        @test coupling_block(J, u, 1, static(2), 6) isa HybridArray
        @test coupling_block(J, u, 1, 2, static(6)) isa HybridArray
        @test eliminated_block(J, 1, 1, static(2), static(3)) isa SizedMatrix{2, 3, T}
        @test residual_block(J, 1, 1, static(2)) isa SizedVector{2, T}
        coupling_block(J, u, 1, static(2), static(6)) .= SMatrix{2, 6, T}(1:12)
        @test coupling_block(J, u)[1:2, :] == reshape(T.(1:12), 2, 6)
        @inferred coupling_block(J, u, 1, static(2), static(6))
        @inferred coupling_block(J, u, 1, 2)
        @inferred coupling_block(J, u)
        @inferred eliminated_block(J, 1, 1, static(2), static(3))
        @inferred eliminated_block(J, 1)
        @inferred residual_block(J, 1, 1, static(2))
        @inferred residual_block(J, 1)
        e, t = 1, 1
        fill_static!(J, u, e, t); fill_dynamic!(J, u, e, t)
        @test allocs_fill_static(J, u, e, t) == 0
        @test allocs_fill_dynamic(J, u, e, t) == 0
    end
end

@testset "Filling different tables from different threads is race-free (E1)" begin
    p = qr_problems(Float64)[1]
    J = p.J
    jst = J.structure
    function fill_table!(J, t)
        jst = J.structure
        m = t <= jst.schur.neblocks ? jst.e_rows[t] : jst.free_rows[t - jst.schur.neblocks]
        for row in 1:2:m - 1                        # two rows per residual, as a caller would write them
            for u in jst.table_ptr[t]:jst.table_ptr[t+1]-1
                coupling_block(J, u, row, static(2), static(6)) .= SMatrix{2, 6}(ntuple(i -> sin(i + 7row + 13u), 12))
            end
            t <= jst.schur.neblocks && (eliminated_block(J, t, row, static(2), static(3)) .= SMatrix{2, 3}(ntuple(i -> cos(i + row + t), 6)))
            residual_block(J, t, row, static(2)) .= SVector(row + t, row - t)
        end
    end
    serial = copy_structure(J)
    for t in 1:jst.ntables
        fill_table!(serial, t)
    end
    threaded = copy_structure(J)
    Threads.@threads for t in 1:jst.ntables
        fill_table!(threaded, t)
    end
    @test threaded.data == serial.data
    @test count(!iszero, serial.data) > 0.8 * sum(jst.e_rows[e] * jst.table_ncols[e] for e in 1:jst.schur.neblocks)
end
