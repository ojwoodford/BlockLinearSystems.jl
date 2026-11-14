using Random, BlockLinearSystems, SparseArrays, StaticArrays, Static, HybridArrays, LinearAlgebra, Test, Aqua, JET

@testset "BlockLinearSystems.jl" begin
    include("aqua.jl")
    include("SparseBlockLinearSystem.jl")
    include("DenseBlockLinearSystem.jl")
    include("BlockRLE.jl")
    include("typestability.jl")
    include("helpers.jl")
    include("schur.jl")
    include("schur_problems.jl")
    include("schur_jacobian.jl")
    include("qr_reduction.jl")
    include("schur_qr.jl")
    include("schur_assembly.jl")
    include("zero.jl")
end
