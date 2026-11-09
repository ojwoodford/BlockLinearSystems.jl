@testset "DenseBlockLinearSystem" begin
    # Construct using block sizes [2, 2]
    bdm = DenseBlockLinearSystem{Float64}([2, 2])
    @test validblock(bdm, 1, 1) == true
    @test validblock(bdm, 2, 2) == true
    @test validblock(bdm, 3, 1) == false
    @test size(bdm) == (4, 4)
    @test size(bdm, 1) == 4
    @test size(bdm, 2) == 4
    @test axes(bdm) == (Base.OneTo(4), Base.OneTo(4))
    @test axes(bdm, 1) == Base.OneTo(4)
    @test length(bdm) == 16
    @test length(bdm.data) == 16
    @test eltype(bdm) == Float64

    # Accessors
    @test blocksizes(bdm, 1) == [2, 2]
    @test blocksizes(bdm, 2) == [2, 2]
    @test blocksizes(bdm) == ([2, 2], [2, 2])
    @test nblocks(bdm, 1) == 2
    @test nblocks(bdm, 2) == 2
    @test nblocks(bdm) == (2, 2)
    @test blockdata(bdm) === bdm.data
    @test blockrange(bdm, 1, 1) == 1:2
    @test blockrange(bdm, 1, 2) == 3:4

    # eachstoredblock iterator
    all_blocks = [(r, c) for (r, c, B) in eachstoredblock(bdm)]
    @test length(all_blocks) == 4
    @test all_blocks == [(1, 1), (2, 1), (1, 2), (2, 2)]

    uniformscaling!(bdm, 1.0)
    @test bdm[1, 1] == 1.0
    @test bdm[4, 4] == 1.0
    bdm[2, 1] = 3.5
    @test bdm[2, 1] == 3.5

    @test block(bdm, 1, 1) isa AbstractMatrix
    @test block(bdm, 1, 1, static(2), static(2)) isa SizedMatrix{2, 2}
    @test block(bdm, 1, 1, static(2), 2) isa HybridArray
    @test block(bdm, 1, 1, 2, static(2)) isa HybridArray
    @test block(bdm, 1, 1, Val(2), Val(2)) isa SizedMatrix{2, 2}
    @test block(bdm, 1, 1, Val(2), 2) isa HybridArray
    @test block(bdm, 1, 1, 2, Val(2)) isa HybridArray

    # Bounds and dimension mismatch checks
    @test_throws BoundsError block(bdm, 3, 1)
    @test_throws DimensionMismatch block(bdm, 1, 1, static(3), static(2))
    @test_throws DimensionMismatch block(bdm, 1, 1, 2, 3)

    block(bdm, 2, 1) .= [5.0 6.0; 7.0 8.0]

    # symmetrifyfull allocates new matrix, doesn't mutate
    sym_mat = symmetrifyfull(bdm)
    @test sym_mat[1:2, 3:4] == [5.0 6.0; 7.0 8.0]'
    @test sym_mat !== bdm.data
    @test bdm[1, 3] == 0.0 # not mutated

    # symmetrifyfull! mutates in place
    sym_in_place = symmetrifyfull!(bdm)
    @test sym_in_place === bdm.data
    @test bdm.data[1:2, 3:4] == [5.0 6.0; 7.0 8.0]'

    bdm_copy = copy_structure(bdm)
    @test size(bdm_copy) == (4, 4)

    # Rectangular DenseBlockLinearSystem with differing row/column blocksizes
    bdm_rect = DenseBlockLinearSystem{Float64}([2, 3], [3, 4, 1])
    @test size(bdm_rect) == (5, 8)
    @test nblocks(bdm_rect) == (2, 3)
    @test blocksizes(bdm_rect, 1) == [2, 3]
    @test blocksizes(bdm_rect, 2) == [3, 4, 1]
    @test all(iszero, bdm_copy.data)

    zero!(bdm)
    @test all(iszero, bdm.data)

    mat_buf = ones(2, 3)
    bdm_rect = DenseBlockLinearSystem{Float64}(mat_buf, [2], [3])
    @test size(bdm_rect) == (2, 3)
    zero!(mat_buf)
    @test all(iszero, mat_buf)

    # AbstractMatrix support (e.g. SubArray or custom dense matrix)
    parent_buf = zeros(Float64, 6, 6)
    dense_view = view(parent_buf, 1:4, 1:4)
    @test dense_view isa AbstractMatrix
    bdm_view = DenseBlockLinearSystem(dense_view, [2, 2])
    @test bdm_view isa DenseBlockLinearSystem{Float64, typeof(dense_view)}
    block(bdm_view, 1, 1) .= 2.5
    @test parent_buf[1, 1] == 2.5
    @test Matrix(bdm_view) isa Matrix{Float64}
    bdm_view_copy = copy_structure(bdm_view)
    @test bdm_view_copy.data isa AbstractMatrix{Float64}
    @test size(bdm_view_copy.data) == size(dense_view)

    # BlockOffsets and offsets keyword constructors
    bo = BlockOffsets([1, 3, 6])
    co = BlockOffsets([1, 4])
    @test_throws ArgumentError BlockOffsets([1]) |> (b -> DenseBlockLinearSystem{Float64}(b))
    @test_throws ArgumentError DenseBlockLinearSystem{Float64}(BlockOffsets([0, 3]))
    @test_throws ArgumentError DenseBlockLinearSystem{Float64}(bo, BlockOffsets([2, 4]))
    @test_throws ArgumentError DenseBlockLinearSystem(zeros(5, 5), BlockOffsets([1]))
    @test_throws ArgumentError DenseBlockLinearSystem(zeros(5, 5), BlockOffsets([0, 3]))
    @test_throws ArgumentError DenseBlockLinearSystem(zeros(5, 3), bo, BlockOffsets([2, 4]))
    @test_throws DimensionMismatch DenseBlockLinearSystem(zeros(4, 5), bo)

    bdm_bo_sq = DenseBlockLinearSystem{Float64}(bo)
    @test size(bdm_bo_sq) == (5, 5)
    bdm_bo_rect = DenseBlockLinearSystem{Float64}(bo, co)
    @test size(bdm_bo_rect) == (5, 3)

    data55 = zeros(Float64, 5, 5)
    data53 = zeros(Float64, 5, 3)
    bdm_bo_data_sq = DenseBlockLinearSystem{Float64, typeof(data55)}(data55, bo)
    @test size(bdm_bo_data_sq) == (5, 5)
    bdm_bo_data_rect = DenseBlockLinearSystem{Float64, typeof(data53)}(data53, bo, co)
    @test size(bdm_bo_data_rect) == (5, 3)
    @test DenseBlockLinearSystem{Float64}(data55, bo) isa DenseBlockLinearSystem
    @test DenseBlockLinearSystem(data55, bo) isa DenseBlockLinearSystem
    @test DenseBlockLinearSystem{Float64}(data53, bo, co) isa DenseBlockLinearSystem
    @test DenseBlockLinearSystem(data53, bo, co) isa DenseBlockLinearSystem

    # Raw vector offset constructors
    rbs_vec = [2, 3]
    cbs_vec = [3]
    bdm_raw_sq = DenseBlockLinearSystem{Float64, Matrix{Float64}}(data55, [1, 3, 6], [1, 3, 6], rbs_vec, rbs_vec)
    @test size(bdm_raw_sq) == (5, 5)
    bdm_raw_rect = DenseBlockLinearSystem{Float64, Matrix{Float64}}(data53, [1, 3, 6], [1, 4], rbs_vec, cbs_vec)
    @test size(bdm_raw_rect) == (5, 3)
    @test DenseBlockLinearSystem{Float64}(data55, [1, 3, 6], [1, 3, 6], rbs_vec, rbs_vec) isa DenseBlockLinearSystem

    # Keyword argument constructors
    @test size(DenseBlockLinearSystem{Float64}(; offsets=[1, 3, 6])) == (5, 5)
    @test size(DenseBlockLinearSystem{Float64}(; offsets=[1, 3, 6], coloffsets=[1, 4])) == (5, 3)
    @test size(DenseBlockLinearSystem{Float64}(data55; offsets=[1, 3, 6])) == (5, 5)
    @test size(DenseBlockLinearSystem{Float64}(data53; offsets=[1, 3, 6], coloffsets=[1, 4])) == (5, 3)
    @test size(DenseBlockLinearSystem(data55; offsets=[1, 3, 6])) == (5, 5)
    @test size(DenseBlockLinearSystem(data53; offsets=[1, 3, 6], coloffsets=[1, 4])) == (5, 3)
end
