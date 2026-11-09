# Helper to verify zero allocations in inner block accumulation
function test_block_alloc(b)
    val = @SMatrix [1.0 2.0; 3.0 4.0; 5.0 6.0]
    block(b, 2, 1, static(3), static(2)) .+= val
    return nothing
end

@testset "SparseBlockLinearSystem" begin
    # Construct a test matrix
    out = [0. 0  0  0  0  10;
           0  0  0  0  0  11;
           0  0  0  0  0  12;
           1  4  0  0  0  0 ;
           2  5  0  0  0  0 ;
           3  6  0  0  0  0 ;
           0  0  7  8  9  13;
           0  0  0  0  0  0 ]

    # 1. Construct using TransposedPattern
    pattern_t = sparse([0 0 1; 1 0 0; 0 1 1; 0 0 0]' .> 0)
    b = SparseBlockLinearSystem{Float64}(TransposedPattern(pattern_t), [3, 3, 1, 1], [2, 3, 1])
    block(b, 2, 1) .= reshape(1.:6., (3, 2))
    block(b, 3, 2, 1, 3) .= SMatrix{1, 3, Float64, 3}(7:9)
    block(b, 1, 3, static(3), static(1)) .= [10.; 11; 12]
    block(b, 3, 3) .= 13.

    # 2. Construct using standard (natural orientation) pattern
    pattern_natural = sparse(pattern_t')
    b_nat = SparseBlockLinearSystem{Float64}(pattern_natural, [3, 3, 1, 1], [2, 3, 1])
    block(b_nat, 2, 1) .= reshape(1.:6., (3, 2))
    block(b_nat, 3, 2, 1, 3) .= SMatrix{1, 3, Float64, 3}(7:9)
    block(b_nat, 1, 3, static(3), static(1)) .= [10.; 11; 12]
    block(b_nat, 3, 3) .= 13.
    @test Matrix(b_nat) == out

    # 3. Construct using block coordinates [(row, col), ...]
    coords = [(2, 1), (3, 2), (1, 3), (3, 3)]
    b_coords = SparseBlockLinearSystem{Float64}(coords, [3, 3, 1, 1], [2, 3, 1])
    block(b_coords, 2, 1) .= reshape(1.:6., (3, 2))
    block(b_coords, 3, 2, 1, 3) .= SMatrix{1, 3, Float64, 3}(7:9)
    block(b_coords, 1, 3, static(3), static(1)) .= [10.; 11; 12]
    block(b_coords, 3, 3) .= 13.
    @test Matrix(b_coords) == out

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
    @test !validblock(b, 10, 10)

    # Accessors
    @test blocksizes(b, 1) == [3, 3, 1, 1]
    @test blocksizes(b, 2) == [2, 3, 1]
    @test blocksizes(b) == ([3, 3, 1, 1], [2, 3, 1])
    @test nblocks(b, 1) == 4
    @test nblocks(b, 2) == 3
    @test nblocks(b) == (4, 3)
    @test blockdata(b) === b.data
    @test blockrange(b, 1, 1) == 1:3
    @test blockrange(b, 1, 2) == 4:6
    @test blockrange(b, 2, 2) == 3:5

    # eachstoredblock iterator
    stored_coords = [(r, c) for (r, c, B) in eachstoredblock(b)]
    @test (2, 1) in stored_coords
    @test (3, 2) in stored_coords
    @test (1, 3) in stored_coords
    @test (3, 3) in stored_coords
    @test length(stored_coords) == 4

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

    # Internal eager index checks
    @test all(b.indices.nzval .!= 0)
    @test b.indices == b.indicestransposed'

    # copy_structure and zero!
    b_copy = copy_structure(b)
    @test size(b_copy) == size(b)
    @test all(iszero, b_copy.data)
    block(b_copy, 2, 1) .= 1.0
    zero!(b_copy)
    @test all(iszero, b_copy.data)

    # Direct data constructor
    b_direct = SparseBlockLinearSystem{Float64}(copy(b.data), b.indices, b.indicestransposed, b.rowblocksizes, b.columnblocksizes, b.m, b.n)
    @test Matrix(b_direct) == out

    # Bounds & dimension checks
    @test_throws BoundsError block(b, 10, 1)
    @test_throws BoundsError block(b, 1, 10)
    @test_throws ArgumentError block(b, 1, 2) # structurally zero
    @test_throws DimensionMismatch block(b, 2, 1, static(2), static(2)) # expected 3x2
    @test_throws DimensionMismatch block(b, 2, 1, 3, 3) # expected 3x2

    # Allocation test: @inbounds block accumulation should have zero allocations
    test_block_alloc(b) # warmup
    @test (@allocated test_block_alloc(b)) == 0

    # Block dimensions limit (> 255)
    @test_throws ArgumentError SparseBlockLinearSystem{Float64}(sparse([true;;]), [300], [300])

    # Symmetric lower-triangular BSM tests
    out2 = [0  0  0   0   0   0 ;
            0  0  0   0   0   0 ;
            1  4  7   8   9   0 ;
            2  5  8   10  11  0 ;
            3  6  9   11  12  0 ;
            0  0  13  14  15  16]
    outsym = max.(out2, out2')

    bs = SparseBlockLinearSystem{Int64}(TransposedPattern(sparse([0 0 0; 1 1 0; 0 1 1]' .> 0)), [2, 3, 1], [2, 3, 1])
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
    sub_mat1 = zeros(Int64, 2, 2)
    symmetrifyfull!(sub_mat1, bs, [1])
    @test sub_mat1 == outsym[1:2, 1:2]
    # symmetrifyfull! with AbstractMatrix target (e.g. SubArray)
    full_dest = zeros(Int64, 8, 8)
    dest_view = view(full_dest, 1:6, 1:6)
    @test dest_view isa AbstractMatrix
    symmetrifyfull!(dest_view, bs)
    @test dest_view == outsym
    # BitVector mask for blocks
    sub_mat_bv = zeros(Int64, 4, 4)
    symmetrifyfull!(sub_mat_bv, bs, BitVector([false, true, true]))
    @test sub_mat_bv == outsym[3:6, 3:6]
    # Empty blocks subset
    empty_mat = zeros(Int64, 0, 0)
    symmetrifyfull!(empty_mat, bs, Int[])
    @test isempty(empty_mat)
    @test_throws DimensionMismatch symmetrifyfull!(zeros(Int64, 3, 3), bs, [2, 3])
    @test_throws BoundsError symmetrifyfull!(zeros(Int64, 2, 2), bs, [10])

    # Val static sizing
    @test block(bs, 2, 2, Val(3), Val(3)) isa SizedMatrix{3, 3}
    @test block(bs, 2, 1, Val(3), 2) isa HybridArray
    @test block(bs, 2, 1, 3, Val(2)) isa HybridArray

    s3 = symmetrifysparse(bs)
    @test issparse(s3)
    @test typeof(s3) == SparseMatrixCSC{Int64, Int64}
    @test size(s3) == (6, 6)
    @test nnz(s3) == 28
    @test Matrix(s3) == outsym

    # sparse_with_map & zero-allocation copyto! (symmetric)
    csc_sym, map_sym = sparse_with_map(bs; symmetric=true)
    @test csc_sym == s3
    @test length(map_sym) == nnz(s3)
    # Test in-place update after modifying bs
    block(bs, 2, 1) .+= 5
    copyto!(csc_sym, bs, map_sym)
    @test csc_sym == symmetrifysparse(bs)
    # Verify 0 allocations on copyto!
    copyto!(csc_sym, bs, map_sym) # warmup
    @test (@allocated copyto!(csc_sym, bs, map_sym)) == 0

    # sparse_with_map & zero-allocation copyto! (non-symmetric)
    csc_asym, map_asym = sparse_with_map(b; symmetric=false)
    @test csc_asym == sparse(b)
    copyto!(csc_asym, b, map_asym) # warmup
    @test (@allocated copyto!(csc_asym, b, map_asym)) == 0

    # uniformscaling! on symmetric BSM with full diagonal blocks
    bs_diag = SparseBlockLinearSystem{Float64}(TransposedPattern(sparse([1 0; 1 1]' .> 0)), [2, 2])
    block(bs_diag, 1, 1) .= [1.0 0.0; 0.0 2.0]
    block(bs_diag, 2, 1) .= [0.5 0.0; 0.0 0.5]
    block(bs_diag, 2, 2) .= [3.0 0.0; 0.0 4.0]
    uniformscaling!(bs_diag, 10.0)
    @test diag(Matrix(bs_diag)) == [11.0, 12.0, 13.0, 14.0]

    # uniformscaling! error on non-square or missing diagonal
    bs_rect = SparseBlockLinearSystem{Float64}(TransposedPattern(sparse([true;;])), [2], [3])
    @test_throws ArgumentError uniformscaling!(bs_rect, 1.0)
    bs_nodiag = SparseBlockLinearSystem{Float64}(TransposedPattern(sparse([false true; true false])), [2, 2])
    @test_throws ArgumentError uniformscaling!(bs_nodiag, 1.0)
end
