using Random, BlockLinearSystems, SparseArrays, StaticArrays, Static, HybridArrays, LinearAlgebra, Test, Aqua, JET

@testset "BlockLinearSystems.jl" begin
    include("aqua.jl")
    include("SparseBlockLinearSystem.jl")
    include("DenseBlockLinearSystem.jl")
    include("BlockRLE.jl")
    include("typestability.jl")
    include("helpers.jl")
    include("schur.jl")
    # The extension tests need SecondOrderOptimization, which is unavailable in the Julia 1.10 CI job
    if Base.find_package("SecondOrderOptimization") !== nothing
        include("schur_ext.jl")
    else
        @info "SecondOrderOptimization not available: skipping extension tests"
    end
    include("zero.jl")
end
