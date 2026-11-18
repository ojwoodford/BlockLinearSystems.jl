using Test   # each test file loads the packages and fixtures it needs

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

# Test groups, run by name: `julia --project=<test env> test/runtests.jl threaded`, or
# `Pkg.test(test_args = ["threaded"])`; all groups when none is named. The full check before a push is every group on
# 1 thread, then `threaded` on 4 threads (rules.md 30).
#   threaded: every test whose behaviour depends on Threads.nthreads() (threaded kernels, threaded allocation bounds)
#   process:  the test that starts a fresh Julia process
#   serial:   everything else, which behaves the same on any number of threads
const TEST_GROUPS = (
    serial = ["aqua.jl", "exports.jl", "SparseBlockLinearSystem.jl", "DenseBlockLinearSystem.jl", "BlockRLE.jl",
              "typestability.jl", "helpers.jl", "kernels.jl", "schur.jl", "schur_problems.jl", "linalg.jl",
              "schur_jacobian.jl", "schur_description.jl", "schur_hessian.jl", "diagonal_blocks.jl", "schur_assembly.jl", "zero.jl", "readme.jl",
              # the DampedLinearSolve extension: the QR-eliminated systems through the linear-system contract
              "dampedlinearsolve_qrelimination.jl"],
    threaded = ["schur_problems_threaded.jl", "schur_jacobian_threaded.jl", "qr_reduction.jl", "schur_qr.jl",
                "threading.jl",
                # the DampedLinearSolve extension: the Schur systems through the linear-system contract
                "dampedlinearsolve_schurhessian.jl"],
    process = ["threading_process.jl", "loading_process.jl"],
)
const RUN_GROUPS = isempty(ARGS) ? collect(keys(TEST_GROUPS)) : Symbol.(ARGS)
for g in RUN_GROUPS
    g in keys(TEST_GROUPS) || throw(ArgumentError("unknown test group $g; the groups are $(join(keys(TEST_GROUPS), ", "))"))
end

# The table is printed even when a test fails
try
@testset "BlockLinearSystems.jl" verbose = true begin
    for g in RUN_GROUPS
        @testset "$g" verbose = true begin
            foreach(timed_include, TEST_GROUPS[g])
        end
    end
end
finally
    print_file_times()
end
