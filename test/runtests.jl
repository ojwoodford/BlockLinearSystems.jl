using Random, BlockLinearSystems, SparseArrays, StaticArrays, Static, HybridArrays, LinearAlgebra, Test, Aqua, Polyester

# Each test file is included through `timed_include`, which logs its wall time and compile time in seconds
# (`compile_time` is reported by `@timed` from Julia 1.11; NaN before) and keeps them for a table at the end.
const FILE_TIMES = Tuple{String, Float64, Float64}[]
function timed_include(f)
    s = @timed include(f)
    compile = hasproperty(s, :compile_time) ? s.compile_time : NaN
    push!(FILE_TIMES, (f, s.time, compile))
    @info f time = round(s.time; digits = 1) compile = round(compile; digits = 1)
    return nothing
end

function print_file_times()
    println("\nSeconds per test file (Julia ", VERSION, ", ", Threads.nthreads(), " thread(s)): total, compile")
    for (f, t, c) in FILE_TIMES
        println(rpad(f, 40), lpad(round(t; digits = 1), 8), lpad(round(c; digits = 1), 8))
    end
    println(rpad("total", 40), lpad(round(sum(x -> x[2], FILE_TIMES; init = 0.0); digits = 1), 8),
            lpad(round(sum(x -> x[3], FILE_TIMES; init = 0.0); digits = 1), 8))
end

# The table is printed even when a test fails
try
@testset "BlockLinearSystems.jl" verbose = true begin
    timed_include("aqua.jl")
    timed_include("exports.jl")
    timed_include("SparseBlockLinearSystem.jl")
    timed_include("DenseBlockLinearSystem.jl")
    timed_include("BlockRLE.jl")
    timed_include("typestability.jl")
    timed_include("helpers.jl")
    timed_include("kernels.jl")
    timed_include("schur.jl")
    timed_include("schur_problems.jl")
    timed_include("linalg.jl")
    timed_include("schur_jacobian.jl")
    timed_include("qr_reduction.jl")
    timed_include("schur_qr.jl")
    timed_include("diagonal_blocks.jl")
    timed_include("schur_assembly.jl")
    timed_include("zero.jl")
    timed_include("threading.jl")
    timed_include("readme.jl")
    # The DampedLinearSolve extension: the Schur and QR-eliminated systems through the linear-system contract
    timed_include("dampedlinearsolve_schurhessian.jl")
    timed_include("dampedlinearsolve_qrelimination.jl")
end
finally
    print_file_times()
end
