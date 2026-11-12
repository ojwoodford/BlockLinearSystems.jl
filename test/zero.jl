using Test, LinearAlgebra, SparseArrays, StaticArrays, Static, BlockLinearSystems

@testset "zero! functionality across systems" begin
    # --------------------------------------------------------------------------
    # 1. DenseBlockLinearSystem zero!
    # --------------------------------------------------------------------------
    @testset "DenseBlockLinearSystem zero!" begin
        bdm = DenseBlockLinearSystem{Float64}([2, 3])
        fill!(bdm.data, 42.0)
        @test any(!iszero, bdm.data)
        BlockLinearSystems.zero!(bdm)
        @test all(iszero, bdm.data)
    end

    # --------------------------------------------------------------------------
    # 2. SparseBlockLinearSystem zero!
    # --------------------------------------------------------------------------
    @testset "SparseBlockLinearSystem zero!" begin
        pattern = sparse([true false true; false true false; true false true])
        sbls = SparseBlockLinearSystem{Float64}(pattern, [2, 2, 2], [2, 2, 2])
        fill!(sbls.data, 99.0)
        @test any(!iszero, sbls.data)
        BlockLinearSystems.zero!(sbls)
        @test all(iszero, sbls.data)
    end

    # --------------------------------------------------------------------------
    # 3. SchurBlockMatrix zero! (Dense & Sparse B)
    # --------------------------------------------------------------------------
    @testset "SchurBlockMatrix zero!" begin
        c_dims = [3, 3]
        p_dims = [2, 2]
        obs = [[1, 2], [1, 2]]
        struct_schur = SchurStructure(c_dims, p_dims, obs)

        # Dense B
        B_dense = fill(5.0, 6, 6)
        C_data_d = fill(3.0, 8)
        E_data_d = fill(2.0, 24)
        H_dense = SchurBlockMatrix(B_dense, C_data_d, E_data_d, struct_schur)
        BlockLinearSystems.zero!(H_dense)
        @test all(iszero, H_dense.B)
        @test all(iszero, H_dense.C_data)
        @test all(iszero, H_dense.E_data)

        # Sparse B
        pattern = sparse([true false; false true])
        B_sbls = SparseBlockLinearSystem{Float64}(pattern, [3, 3], [3, 3])
        fill!(B_sbls.data, 7.0)
        C_data_s = fill(4.0, 8)
        E_data_s = fill(1.0, 24)
        H_sparse = SchurBlockMatrix(B_sbls, C_data_s, E_data_s, struct_schur)
        BlockLinearSystems.zero!(H_sparse)
        @test all(iszero, H_sparse.B.data)
        @test all(iszero, H_sparse.C_data)
        @test all(iszero, H_sparse.E_data)
    end

    # --------------------------------------------------------------------------
    # 4. DenseSchurLinearSystem and SparseSchurLinearSystem zero!
    #    CRITICAL: zero! must clear b and H, but MUST PRESERVE x!
    # --------------------------------------------------------------------------
    ext = Base.get_extension(BlockLinearSystems, :SecondOrderOptimizationExt)
    if ext !== nothing
        @testset "SchurLinearSystem zero! preserves x" begin
            c_dims = [3, 3]
            p_dims = [2, 2]
            obs = [[1, 2], [1, 2]]
            struct_schur = SchurStructure(c_dims, p_dims, obs)

            # Dense Schur Linear System
            H_d = SchurBlockMatrix(fill(2.0, 6, 6), fill(3.0, 8), fill(1.5, 24), struct_schur)
            b_d = fill(10.0, 10)
            x_d = fill(42.0, 10)
            ls_d = ext.DenseSchurLinearSystem(H_d, b_d, x_d)

            ext.SecondOrderOptimization.zero!(ls_d)
            @test all(iszero, ls_d.b)
            @test all(iszero, ls_d.H.B)
            @test all(iszero, ls_d.H.C_data)
            @test all(iszero, ls_d.H.E_data)
            # Verify x is untouched!
            @test all(==(42.0), ls_d.x)

            # Sparse Schur Linear System
            pattern = sparse([true false; false true])
            B_sparse = SparseBlockLinearSystem{Float64}(pattern, [3, 3], [3, 3])
            fill!(B_sparse.data, 2.5)
            H_s = SchurBlockMatrix(B_sparse, fill(3.5, 8), fill(1.2, 24), struct_schur)
            b_s = fill(20.0, 10)
            x_s = fill(84.0, 10)
            ls_s = ext.SparseSchurLinearSystem(H_s, b_s, x_s)

            ext.SecondOrderOptimization.zero!(ls_s)
            @test all(iszero, ls_s.b)
            @test all(iszero, ls_s.H.B.data)
            @test all(iszero, ls_s.H.C_data)
            @test all(iszero, ls_s.H.E_data)
            # Verify x is untouched!
            @test all(==(84.0), ls_s.x)
        end
    end
end
