@testset "Helper Functions" begin
    # block_sparse_nnz (exported)
    sparsity = sparse([true true; false true])
    @test block_sparse_nnz(sparsity, [2, 3]) == 2 * 2 + 2 * 3 + 3 * 3
    @test block_sparse_nnz(sparse([true false; false true]), [2, 3]) == 2 * 2 + 3 * 3

    # computestartindices (unexported helper, accessible via module)
    @test BlockLinearSystems.computestartindices([2, 3, 1, 4]) == [1, 3, 6, 7]
    @test BlockLinearSystems.computestartindices(Int[]) == Int[]

    # find_diag_inds (unexported helper, accessible via module)
    S = sparse([1, 2, 2, 3, 3], [1, 1, 2, 2, 3], [10.0, 1.0, 20.0, 2.0, 30.0], 3, 3)
    dinds = BlockLinearSystems.find_diag_inds(S)
    @test S.nzval[dinds] == [10.0, 20.0, 30.0]
    S_nodiag = sparse([2, 1], [1, 2], [1.0, 2.0], 2, 2)
    @test length(BlockLinearSystems.find_diag_inds(S_nodiag)) == 2

    # inplacecumsum! (unexported helper, accessible via module)
    x = [1, 2, 3, 4]
    @test BlockLinearSystems.inplacecumsum!(x) === x
    @test x == [1, 3, 6, 10]
    y = Float64[]
    @test BlockLinearSystems.inplacecumsum!(y) == Float64[]

    # Unexported status verified
    @test !(:computestartindices in names(BlockLinearSystems))
    @test !(:find_diag_inds in names(BlockLinearSystems))
    @test !(:inplacecumsum! in names(BlockLinearSystems))
end

@testset "Large blocks do not overflow UInt8 (B7)" begin
    # Two 16-dimensional blocks: 16 * 16 = 256 wraps to 0 in UInt8
    @test block_sparse_nnz(sparse([true true; false true]), BlockRLE([16, 16])) == 3 * 256
    @test block_sparse_nnz(sparse([true true; false true]), UInt8[16, 16]) == 3 * 256
    @test block_sparse_nnz(sparse([true false; false true]), BlockRLE([255, 255])) == 2 * 255^2

    sizes = [16, 20, 3]
    S = SparseBlockLinearSystem{Float64}([(1, 1), (2, 1), (2, 2), (3, 3)], sizes)
    for i in 1:3, j in 1:i
        validblock(S, i, j) || continue
        blk = block(S, i, j)
        @test size(blk) == (sizes[i], sizes[j])
        blk .= i * 10 + j
    end
    @test sum(S.data) == sum((i * 10 + j) * sizes[i] * sizes[j] for (i, j) in [(1, 1), (2, 1), (2, 2), (3, 3)])
    M = Matrix(S)
    @test size(M) == (39, 39)
    @test M[1:16, 1:16] == fill(11.0, 16, 16)
    @test M[17:36, 1:16] == fill(21.0, 20, 16)
    F = symmetrifyfull(S)
    @test F[1:16, 17:36] == fill(21.0, 16, 20)
    Fb = zeros(36, 36)
    symmetrifyfull!(Fb, S, [1, 2])
    @test Fb[17:36, 1:16] == fill(21.0, 20, 16)
    @test Fb[1:16, 17:36] == fill(21.0, 16, 20)

    D = DenseBlockLinearSystem{Float64}(sizes)
    @test size(block(D, 1, 2)) == (16, 20)
    @test size(block(D, 2, 2)) == (20, 20)
end
