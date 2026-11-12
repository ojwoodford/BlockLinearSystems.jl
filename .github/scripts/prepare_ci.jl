# Prepare Project.toml for CI.
#
# The test target needs SecondOrderOptimization (plus FastNorm and OptimizationVariableInterface), which are
# not in the General registry. Locally Project.toml points `[sources]` at sibling checkouts; on CI we point
# it at GitHub instead. Julia 1.10 ignores `[sources]`, so there the unregistered test dependencies are
# removed and test/runtests.jl skips the extension tests (they run on 1.11 and newer).
using TOML

const UNREGISTERED = Dict(
    "FastNorm" => "https://github.com/ojwoodford/FastNorm.jl",
    "OptimizationVariableInterface" => "https://github.com/ojwoodford/OptimizationVariableInterface.jl",
    "SecondOrderOptimization" => "https://github.com/ojwoodford/SecondOrderOptimization.jl",
)
const REV = "dev"

path = joinpath(@__DIR__, "..", "..", "Project.toml")
proj = TOML.parsefile(path)

if VERSION >= v"1.11"
    proj["sources"] = Dict(name => Dict("url" => url, "rev" => REV) for (name, url) in UNREGISTERED)
else
    delete!(proj, "sources")
    for name in keys(UNREGISTERED)
        delete!(proj["extras"], name)
        haskey(proj["weakdeps"], name) || delete!(proj["compat"], name)   # weakdeps keep their compat entry
    end
    proj["targets"]["test"] = filter(n -> !haskey(UNREGISTERED, n), proj["targets"]["test"])
end

open(path, "w") do io
    TOML.print(io, proj)
end
println("Prepared Project.toml for Julia $VERSION")
