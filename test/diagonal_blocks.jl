using Test, BlockLinearSystems, LinearAlgebra, Random

# A flat array of SPD blocks of the given dimensions, column-major, one after another
function flat_spd_blocks(rng, dims)
    blocks = Matrix{Float64}[]
    for d in dims
        A = randn(rng, d, d)
        push!(blocks, A' * A + d * I)
    end
    return blocks, reduce(vcat, vec.(blocks); init = Float64[])
end
diag_block_factor_bytes(D, dims) = @allocated factor_diagonal_blocks!(D, dims)
diag_block_solve_bytes(y, D, dims, x) = @allocated solve_diagonal_blocks!(y, D, dims, x)

@testset "diagonal_blocks.jl" begin
    rng = Xoshiro(7)
    dims = [1, 3, 6, 2, 9]
    blocks, D = flat_spd_blocks(rng, dims)
    x = randn(rng, sum(dims))
    F = copy(D)
    @test factor_diagonal_blocks!(F, dims)
    # each block holds its Cholesky factor in the lower triangle
    o = 0
    for (B, d) in zip(blocks, dims)
        L = LowerTriangular(reshape(F[o .+ (1:d * d)], d, d))
        @test L ≈ cholesky(Symmetric(B, :L)).L rtol = 1e-12
        o += d * d
    end
    # the solve against the block-diagonal matrix, into another vector and in place
    ref = cat(blocks...; dims = (1, 2)) \ x
    y = similar(x)
    @test solve_diagonal_blocks!(y, F, dims, x) === y
    @test y ≈ ref rtol = 1e-12
    z = copy(x)
    solve_diagonal_blocks!(z, F, dims, z)
    @test z == y
    # the strict upper triangle is not read: garbage there changes nothing
    G = copy(D)
    o = 0
    for d in dims
        for j in 1:d, i in 1:j-1
            G[o + (j - 1) * d + i] = NaN
        end
        o += d * d
    end
    @test factor_diagonal_blocks!(G, dims)
    @test solve_diagonal_blocks!(similar(x), G, dims, x) == y
    # a block that is not positive definite, or not finite, is reported
    for bad in (B -> (B[1, 1] = -1.0), B -> (B[end, 1] = NaN), B -> (B[1, 1] = Inf), B -> (B .= 0.0))
        bb = deepcopy(blocks)
        bad(bb[3])
        @test !factor_diagonal_blocks!(reduce(vcat, vec.(bb)), dims)
    end
    @test factor_diagonal_blocks!(Float64[], Int[])
    # nothing allocates
    F2 = copy(D)
    diag_block_factor_bytes(copy(D), dims)
    @test diag_block_factor_bytes(F2, dims) == 0
    diag_block_solve_bytes(y, F, dims, x)
    @test diag_block_solve_bytes(y, F, dims, x) == 0
end
