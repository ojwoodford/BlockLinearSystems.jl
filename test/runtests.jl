using BlockMatrices, SparseArrays, StaticArrays, Static, HybridArrays, LinearAlgebra, Test

@testset "BlockMatrices.jl" begin
    @testset "BlockSparseMatrix" begin
        # Construct a test matrix
        out = [0. 0  0  0  0  10;
               0  0  0  0  0  11;
               0  0  0  0  0  12;
               1  4  0  0  0  0 ;
               2  5  0  0  0  0 ;
               3  6  0  0  0  0 ;
               0  0  7  8  9  13;
               0  0  0  0  0  0 ]

        # Construct a matching BSM
        b = BlockSparseMatrix{Float64}(sparse([0 0 1; 1 0 0; 0 1 1; 0 0 0]' .> 0), [3, 3, 1, 1], [2, 3, 1])
        block(b, 2, 1) .= reshape(1.:6., (3, 2))
        block(b, 3, 2, 1, 3) .= SMatrix{1, 3, Float64, 3}(7:9)
        block(b, 1, 3, static(3), static(1)) .= [10.; 11; 12]
        block(b, 3, 3) .= 13.

        # HybridArray block views (StaticInt row / Integer col and Integer row / StaticInt col)
        hb1 = block(b, 2, 1, static(3), 2)
        @test hb1 isa HybridArray
        @test hb1 == reshape(1.:6., (3, 2))
        hb2 = block(b, 2, 1, 3, static(2))
        @test hb2 isa HybridArray
        @test hb2 == reshape(1.:6., (3, 2))

        # Interface checks
        @test eltype(b) == Float64
        @test size(b) == (8, 6)
        @test size(b, 1) == 8
        @test size(b, 2) == 6
        @test size(b, 3) == 1
        @test length(b) == 48
        @test nnz(b) == 13
        @test validblock(b, 2, 1)
        @test !validblock(b, 1, 2)

        m = Matrix(b)
        @test typeof(m) == Matrix{Float64}
        @test size(m) == (8, 6)
        @test m == out

        s = sparse(b)
        @test issparse(s)
        @test typeof(s) == SparseMatrixCSC{Float64, Int64}
        @test size(s) == (8, 6)
        @test nnz(s) == 13
        @test Matrix(s) == out

        # Internal checks & cache management
        @test all(b.indices.nzval .!= 0)
        @test b.indices == b.indicestransposed'
        clearindicescache(b)
        @test isempty(b.indices.nzval)
        cacheindices(b)
        @test b.indices == b.indicestransposed'

        # copy_structure and zero!
        b_copy = copy_structure(b)
        @test size(b_copy) == size(b)
        @test all(iszero, b_copy.data)
        block(b_copy, 2, 1) .= 1.0
        zero!(b_copy)
        @test all(iszero, b_copy.data)

        # Direct data constructor
        b_direct = BlockSparseMatrix{Float64}(copy(b.data), b.indices, b.indicestransposed, b.rowblocksizes, b.columnblocksizes, b.m, b.n)
        @test Matrix(b_direct) == out

        # Symmetric lower-triangular BSM tests
        out2 = [0  0  0   0   0   0 ;
                0  0  0   0   0   0 ;
                1  4  7   8   9   0 ;
                2  5  8   10  11  0 ;
                3  6  9   11  12  0 ;
                0  0  13  14  15  16]
        outsym = max.(out2, out2')

        bs = BlockSparseMatrix{Int64}(sparse([0 0 0; 1 1 0; 0 1 1]' .> 0), [2, 3, 1], [2, 3, 1])
        block(bs, 2, 1) .= reshape(1:6, (3, 2))
        block(bs, 3, 2, 1, 3) .= SVector{3, Int64}(13:15)'
        block(bs, 2, 2, 3, 3) .= [7 8 9; 8 10 11; 9 11 12]
        block(bs, 3, 3) .= 16

        @test eltype(bs) == Int64
        @test size(bs) == (6, 6)
        @test length(bs) == 36
        @test nnz(bs) == 19

        m2 = Matrix(bs)
        @test typeof(m2) == Matrix{Int64}
        @test size(m2) == (6, 6)
        @test m2 == out2

        s2 = sparse(bs)
        @test issparse(s2)
        @test typeof(s2) == SparseMatrixCSC{Int64, Int64}
        @test size(s2) == (6, 6)
        @test nnz(s2) == 19
        @test Matrix(s2) == out2

        m3 = symmetrifyfull(bs)
        @test typeof(m3) == Matrix{Int64}
        @test size(m3) == (6, 6)
        @test m3 == outsym

        # symmetrifyfull! with subset of blocks
        sub_mat = zeros(Int64, 4, 4)
        symmetrifyfull!(sub_mat, bs, [2, 3])
        @test sub_mat == outsym[3:6, 3:6]

        s3 = symmetrifysparse(bs)
        @test issparse(s3)
        @test typeof(s3) == SparseMatrixCSC{Int64, Int64}
        @test size(s3) == (6, 6)
        @test nnz(s3) == 28
        @test Matrix(s3) == outsym

        # uniformscaling! on symmetric BSM with full diagonal blocks
        bs_diag = BlockSparseMatrix{Float64}(sparse([1 0; 1 1]' .> 0), [2, 2])
        block(bs_diag, 1, 1) .= [1.0 0.0; 0.0 2.0]
        block(bs_diag, 2, 1) .= [0.5 0.0; 0.0 0.5]
        block(bs_diag, 2, 2) .= [3.0 0.0; 0.0 4.0]
        uniformscaling!(bs_diag, 10.0)
        @test diag(Matrix(bs_diag)) == [11.0, 12.0, 13.0, 14.0]

        # updatelastrowsym!
        bs_upd = updatelastrowsym!(bs_diag, [1, 2], 3)
        @test size(bs_upd) == (5, 5)
        @test bs_upd.rowblocksizes[end] == 3
        @test bs_upd.columnblocksizes[end] == 3
    end

    @testset "BlockDenseMatrix" begin
        bdm = BlockDenseMatrix{Float64}([1, 3, 5])
        @test validblock(bdm, 1, 1) == true
        @test size(bdm) == (4, 4)
        @test size(bdm, 1) == 4
        @test size(bdm, 2) == 4
        @test axes(bdm) == (Base.OneTo(4), Base.OneTo(4))
        @test axes(bdm, 1) == Base.OneTo(4)
        @test length(bdm) == 16
        @test nnz(bdm.data) == 16
        @test eltype(bdm) == Float64

        uniformscaling!(bdm, 1.0)
        @test bdm[1, 1] == 1.0
        @test bdm[4, 4] == 1.0
        bdm[2, 1] = 3.5
        @test bdm[2, 1] == 3.5

        @test block(bdm, 1, 1) isa AbstractMatrix
        @test block(bdm, 1, 1, static(2), static(2)) isa SizedMatrix{2, 2}
        @test block(bdm, 1, 1, static(2), 2) isa HybridArray
        @test block(bdm, 1, 1, 2, static(2)) isa HybridArray

        block(bdm, 2, 1) .= [5.0 6.0; 7.0 8.0]
        sym_mat = symmetrifyfull(bdm)
        @test sym_mat[1:2, 3:4] == [5.0 6.0; 7.0 8.0]'
        @test Matrix(bdm) == sym_mat

        bdm_copy = copy_structure(bdm)
        @test size(bdm_copy) == (4, 4)
        @test all(iszero, bdm_copy.data)

        zero!(bdm)
        @test all(iszero, bdm.data)

        mat_buf = ones(2, 3)
        bdm_rect = BlockDenseMatrix{Float64}(mat_buf, [1, 3], [1, 4])
        @test size(bdm_rect) == (2, 3)
        zero!(mat_buf)
        @test all(iszero, mat_buf)
    end

    @testset "Helper Functions" begin
        # sparse_dense_decision & block_sparse_nnz
        @test sparse_dense_decision(100, 2343)
        @test !sparse_dense_decision(100, 2344)
        @test !sparse_dense_decision(40, 0)
        @test !sparse_dense_decision(10, 0)

        sparsity = sparse([true true; false true])
        @test block_sparse_nnz(sparsity, [2, 3]) == 2 * 2 + 2 * 3 + 3 * 3
        @test block_sparse_nnz(sparse([true false; false true]), [2, 3]) == 2 * 2 + 3 * 3
        @test sparse_dense_decision(5, sparsity, [2, 3]) == sparse_dense_decision(5, 19)

        # computestartindices
        @test computestartindices([2, 3, 1, 4]) == [1, 3, 6, 7]
        @test computestartindices(Int[]) == Int[]

        # find_diag_inds
        S = sparse([1, 2, 2, 3, 3], [1, 1, 2, 2, 3], [10.0, 1.0, 20.0, 2.0, 30.0], 3, 3)
        dinds = find_diag_inds(S)
        @test S.nzval[dinds] == [10.0, 20.0, 30.0]
    end
end
