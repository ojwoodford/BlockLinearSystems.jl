using Random, BlockLinearSystems, SparseArrays, StaticArrays, Static, HybridArrays, LinearAlgebra, Test, Aqua, JET, Polyester

# The kernels allocate nothing, serial or threaded, except a threaded call on Julia 1.10 (accepted 2026-11-15:
# rebuilding a nested closure inside the threaded loop allocates there, about 300 bytes per chunk), which is held to a
# bound per thread (3 072 bytes, rules.md 21, raised from 2 048 on 2026-11-16) so that a regression still shows.
# `threaded`: whether the call measured took the threaded path.
threaded_allocs_ok(bytes, threaded=Threads.nthreads() > 1) =
    bytes == 0 || (VERSION < v"1.11" && threaded && bytes <= 3072 * Threads.nthreads())

@testset "BlockLinearSystems.jl" begin
    include("aqua.jl")
    include("exports.jl")
    include("SparseBlockLinearSystem.jl")
    include("DenseBlockLinearSystem.jl")
    include("BlockRLE.jl")
    include("typestability.jl")
    include("helpers.jl")
    include("kernels.jl")
    include("schur.jl")
    include("schur_problems.jl")
    include("linalg.jl")
    include("schur_jacobian.jl")
    include("qr_reduction.jl")
    include("schur_qr.jl")
    include("diagonal_blocks.jl")
    include("schur_assembly.jl")
    include("zero.jl")
    include("threading.jl")
    include("readme.jl")
    # The DampedLinearSolve extension: the Schur and QR-eliminated systems through the linear-system contract
    include("dampedlinearsolve_schurhessian.jl")
    include("dampedlinearsolve_qrelimination.jl")
end
