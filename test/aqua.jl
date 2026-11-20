using Test, Aqua, BlockLinearSystems

@testset "Aqua & Code Quality" begin
    Aqua.test_all(BlockLinearSystems)
end
