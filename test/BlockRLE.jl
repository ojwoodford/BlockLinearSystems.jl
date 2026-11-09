@testset "BlockRLE" begin
    # Construction from run lengths and block sizes
    rle = BlockRLE([10, 50], [6, 3])
    @test rle isa BlockRLE{UInt8, Int}
    @test length(rle) == 60
    @test size(rle) == (60,)
    @test nruns(rle) == 2
    @test runlength(rle, 1) == 10
    @test runlength(rle, 2) == 50
    @test runblocksize(rle, 1) == 0x06
    @test runblocksize(rle, 2) == 0x03
    @test matrixdimension(rle) == 10 * 6 + 50 * 3
    @test sum(rle) == matrixdimension(rle)

    # Indexing and binary search
    @test rle[1] == 0x06
    @test rle[10] == 0x06
    @test rle[11] == 0x03
    @test rle[60] == 0x03
    @test_throws BoundsError rle[0]
    @test_throws BoundsError rle[61]

    # Ranges
    @test runblockrange(rle, 1) == 1:10
    @test runblockrange(rle, 2) == 11:60
    @test runmatrixrange(rle, 1) == 1:60
    @test runmatrixrange(rle, 2) == 61:210

    # computestartindices
    starts = BlockLinearSystems.computestartindices(rle)
    @test length(starts) == 60
    @test starts[1] == 1
    @test starts[10] == 55
    @test starts[11] == 61
    @test starts[60] == 208
    @test starts == BlockLinearSystems.computestartindices(collect(rle))

    # Constructors from Pairs and Tuples
    rle_pair = BlockRLE([10 => 6, 50 => 3])
    @test rle_pair == rle
    rle_tup = BlockRLE([(10, 6), (50, 3)])
    @test rle_tup == rle

    # Construction from flat vector (automatic compression)
    flat = vcat(fill(6, 10), fill(3, 50))
    rle_from_flat = BlockRLE(flat)
    @test rle_from_flat == rle
    @test nruns(rle_from_flat) == 2
    @test collect(rle) == flat
    @test Vector(rle) == flat
    @test Vector{Int}(rle) == flat
    @test rle == flat
    @test flat == rle

    # Merging adjacent runs
    rle_merge = BlockRLE([5, 5, 20, 30], [4, 4, 2, 2])
    @test nruns(rle_merge) == 2
    @test runlength(rle_merge, 1) == 10
    @test runblocksize(rle_merge, 1) == 4
    @test runlength(rle_merge, 2) == 50
    @test runblocksize(rle_merge, 2) == 2

    # Copy and equality with non-matching run structures
    rle_cp = copy(rle)
    @test rle_cp == rle
    @test rle_cp !== rle
    rle_a = BlockRLE([1, 1], [UInt8(2), UInt8(2)])
    rle_b = BlockRLE([2], [UInt8(2)])
    @test rle_a == rle_b
    rle_c = BlockRLE([1, 1], [UInt8(2), UInt8(3)])
    rle_d = BlockRLE([1, 1], [UInt8(3), UInt8(2)])
    @test rle_c != rle_d
    @test convert(BlockRLE, [2, 3]) == BlockRLE([1, 1], [2, 3])
    @test convert(BlockRLE{UInt8}, [2, 3]) == BlockRLE([1, 1], [2, 3])

    # Empty BlockRLE
    rle_empty = BlockRLE(Int[], UInt8[])
    @test isempty(rle_empty)
    @test length(rle_empty) == 0
    @test matrixdimension(rle_empty) == 0
    @test nruns(rle_empty) == 0
    @test collect(rle_empty) == UInt8[]

    # eachrun iterator
    runs_collected = collect(eachrun(rle))
    @test length(runs_collected) == 2
    @test runs_collected[1].run == 1
    @test runs_collected[1].blockrange == 1:10
    @test runs_collected[1].blocksize == 0x06
    @test runs_collected[1].matrixrange == 1:60
    @test runs_collected[2].run == 2
    @test runs_collected[2].blockrange == 11:60
    @test runs_collected[2].blocksize == 0x03
    @test runs_collected[2].matrixrange == 61:210

    # Invalid arguments
    @test_throws ArgumentError BlockRLE([1], [0])
    @test_throws ArgumentError BlockRLE([1], [256])
    @test_throws ArgumentError BlockRLE([-1], [3])
    @test_throws DimensionMismatch BlockRLE([1, 2], [3])

    # Two-tier hierarchical sorting
    roles = [2, 1, 1, 2, 1]
    dims  = [6, 3, 3, 6, 3]
    perm, rle_sorted = sort_blocks_hierarchical(roles, dims)
    @test dims[perm] == [3, 3, 3, 6, 6]
    @test nruns(rle_sorted) == 2
    @test runlength(rle_sorted, 1) == 3
    @test runblocksize(rle_sorted, 1) == 3
    @test runlength(rle_sorted, 2) == 2
    @test runblocksize(rle_sorted, 2) == 6

    # DenseBlockLinearSystem with BlockRLE
    bdm_rle = DenseBlockLinearSystem{Float64}(rle)
    @test bdm_rle.rowblocksizes isa BlockRLE
    @test size(bdm_rle.data) == (210, 210)
    @test nblocks(bdm_rle, 1) == 60
    @test size(block(bdm_rle, 1, 1)) == (6, 6)
    @test size(block(bdm_rle, 11, 11)) == (3, 3)
    block(bdm_rle, 1, 1) .= 1.0
    @test bdm_rle.data[1, 1] == 1.0
    uniformscaling!(bdm_rle, 2.0)
    @test bdm_rle.data[1, 1] == 3.0
    bdm_copy = copy_structure(bdm_rle)
    @test bdm_copy.rowblocksizes isa BlockRLE
    @test bdm_copy.data[1, 1] == 0.0

    # SparseBlockLinearSystem with BlockRLE
    sparsity = spdiagm(0 => fill(true, 60))
    sbls_rle = SparseBlockLinearSystem{Float64}(sparsity, rle)
    @test sbls_rle.rowblocksizes isa BlockRLE
    @test sbls_rle.m == 210
    @test sbls_rle.n == 210
    @test size(block(sbls_rle, 1, 1)) == (6, 6)
    @test size(block(sbls_rle, 11, 11)) == (3, 3)
    block(sbls_rle, 1, 1) .= 5.0
    uniformscaling!(sbls_rle, 1.0)
    @test block(sbls_rle, 1, 1)[1, 1] == 6.0

    # Conversions and constructors
    @test BlockRLE(rle) === rle
    @test BlockRLE{UInt8}(rle) === rle
    @test BlockRLE{UInt8, Int}(rle) === rle
    rle_i32 = BlockRLE{UInt8, Int32}(rle)
    @test rle_i32 isa BlockRLE{UInt8, Int32}
    @test BlockRLE{UInt8, Int}(rle_i32) isa BlockRLE{UInt8, Int}
    rle_int = BlockRLE{Int}(rle)
    @test rle_int isa BlockRLE{Int, Int}
    @test rle_int.blocksizes == [6, 3]
    @test convert(Vector, rle) == collect(rle)
    @test convert(Vector{UInt8}, rle) == collect(rle)
    @test convert(Vector{Int}, rle) == Vector{Int}(collect(rle))
    @test convert(Vector{Float64}, rle) == Vector{Float64}(collect(rle))
    @test BlockLinearSystems.as_block_rle(rle) === rle
    @test BlockLinearSystems.as_block_rle(rle_i32) isa BlockRLE{UInt8, Int}
    @test BlockLinearSystems.as_block_rle(rle_int) isa BlockRLE{UInt8, Int}
end
