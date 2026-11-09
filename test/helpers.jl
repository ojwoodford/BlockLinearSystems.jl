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
