using BlockLinearSystems: inv_sym_block   # internal names

@testset "inv_sym_block in Float32 and Float64 (M4)" begin
    H2 = SMatrix{2, 2, Float32, 4}(1, 1, 1, 1 + 1f-6)
    @test !inv_sym_block(H2, 0f0, SVector{2, Float32}(1, 1)).success
    @test inv_sym_block(SMatrix{2, 2, Float64, 4}(1, 1, 1, 1 + 1e-6), 0.0, SVector{2, Float64}(1, 1)).success
end
