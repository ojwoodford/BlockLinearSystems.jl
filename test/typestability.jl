# Run-time sized `block` accessors must be type stable and non-allocating (B9, B14)
function update_block!(sys, i, j, rows, cols)
    b = block(sys, i, j, rows, cols)
    b[1, 1] += 1
    return nothing
end

function update_block2!(sys, i, j)
    b = block(sys, i, j)
    b[1, 1] += 1
    return nothing
end

@testset "block accessors are type stable" begin
    sizes = [3, 3, 4]
    systems = (
        DenseBlockLinearSystem{Float64}(sizes),
        SparseBlockLinearSystem{Float64}([(1, 1), (2, 1), (2, 2), (3, 3)], sizes),
    )
    for sys in systems
        @testset "$(nameof(typeof(sys)))" begin
            for (r, c) in ((3, 3), (static(3), 3), (3, static(3)), (static(3), static(3)))
                @test isconcretetype(Base.promote_op(block, typeof(sys), Int, Int, typeof(r), typeof(c)))
                @inferred block(sys, 2, 1, r, c)
            end
            @test isconcretetype(Base.promote_op(block, typeof(sys), Int, Int))
            @inferred block(sys, 2, 1)
            @inferred block(sys, 3, 3)

            # Block updates with run-time sizes do not allocate
            update_block!(sys, 2, 1, 3, 3)
            update_block2!(sys, 2, 1)
            @test (@allocated update_block!(sys, 2, 1, 3, 3)) == 0
            @test (@allocated update_block2!(sys, 2, 1)) == 0
            @test (@allocated update_block!(sys, 2, 1, static(3), 3)) == 0

            # Values land in the right place with run-time sizes
            zero!(sys)
            block(sys, 3, 3) .= 7
            @test size(block(sys, 3, 3)) == (4, 4)
            @test all(==(7), block(sys, 3, 3))
            @test sum(blockdata(sys)) == 7 * 16
        end
    end
    @test blockrange(5, 3) === 5:7
    @test blockrange(5, static(3)) == 5:7
    @test blockrange(UInt8(5), UInt8(200)) == 5:204
end
