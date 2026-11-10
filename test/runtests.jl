using BlockLinearSystems, SparseArrays, StaticArrays, Static, HybridArrays, LinearAlgebra, Test, Aqua, JET

@testset "BlockLinearSystems.jl" begin
    include("aqua.jl")
    include("SparseBlockLinearSystem.jl")
    include("DenseBlockLinearSystem.jl")
    include("BlockRLE.jl")
    include("helpers.jl")
    include("schur.jl")
    include("zero.jl")
end
