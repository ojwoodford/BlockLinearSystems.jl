module PrecisionTests

using Test, Random, LinearAlgebra, SparseArrays, StaticArrays, Static
using DampedLinearSolve, BlockLinearSystems, Polyester
include("fixtures/load.jl")
using Main.SchurFixtures

# The Schur-complement systems of a SchurBlockHessian{Float32} (single-precision.md, SP-B1): the Hessian, its
# reduction and the reduced factorization in Float32; b, x, the damping and every query in Float64. Solutions are
# compared with a Float64 dense Cholesky solve of the full damped Float64 matrix, to a tolerance in units of
# eps(Float32)·cond.

# A Float32 copy of a Float64 SchurBlockHessian of the same structure
function to32(H::SchurBlockHessian{Float64})
    R = H.R isa Matrix ? Float32.(H.R) :
        BlockLinearSystems.SparseBlockLinearSystem{Float32}(Float32.(H.R.data), H.R.indices, H.R.indicestransposed,
                                                            H.R.rowblocksizes, H.R.columnblocksizes, H.R.m, H.R.n)
    return SchurBlockHessian(H.structure, R, Float32.(H.E_data), Float32.(H.C_data); pattern = H.pattern)
end
pr_hessian(kind, p) = to32(kind === :dense ? dense_schur_hessian(p) : sparse_schur_hessian(p))
pr_tol(A) = 16 * size(A, 1) * eps(Float32) * cond(A)
pr_allocs(ls, s) = (solve!(ls, s); @allocated solve!(ls, s))
const PR_CHOLMOD32 = Float32 <: SparseArrays.CHOLMOD.VTypes
pr_solvers(kind) = kind === :dense || PR_CHOLMOD32 ? (:auto, :cholesky, :ldl) : (:ldl,)

@testset "precision.jl: Schur systems of a SchurBlockHessian{Float32}" begin
    p = standard_problem()
    n = length(p.b)
    for kind in (:dense, :sparse), s in pr_solvers(kind)
        @testset "$kind, :$s" begin
            H = pr_hessian(kind, p)
            ls = makehessiansystem(H, s)
            @test systemeltype(ls) === Float32
            @test eltype(getb(ls)) === Float64 && eltype(getx(ls)) === Float64 && ls.lambda isa Float64 && eltype(ls.scaling) === Float64
            @test eltype(ls.S) === Float32 && eltype(ls.workspace.E_inv_data) === Float32 && eltype(ls.workspace.W_data) === Float32
            @test eltype(ls.workspace.z_data) === Float64 && eltype(ls.g_S) === Float64
            kind === :dense && @test systemeltype(ls.red) === Float32
            getb(ls) .= p.b
            H64 = Float64.(DampedLinearSolve.conformance_matrix(H))   # the stored Float32 matrix
            for (λ, sc) in ((0.0, nothing), (0.3, nothing), (0.3, abs.(diag(p.H))))
                dampen!(ls, λ, sc)
                @test solve!(ls, s) == Success
                ref = reference_solve(p, λ, isnothing(sc) ? ones(n) : sc)
                @test norm(getx(ls) - ref) / norm(ref) <= pr_tol(p.H + λ * (isnothing(sc) ? I : Diagonal(sc)))
            end
            @test max_diag(ls) isa Float64 && max_diag(ls) == maximum(abs, diag(H64))
            v = [cos(0.3 * i) for i in 1:n]
            @test dot(v, getA(ls), v) isa Float64
            @test dot(v, getA(ls), v) ≈ dot(v, H64 * v) rtol = 1e-12   # accumulated in Float64
            d = zeros(n); getdiag!(d, ls)
            @test d == diag(H64)
            @test DampedLinearSolve.quadratic_model_change(ls, v) isa Float64
            # Nothing allocates, but inside CHOLMOD (rules.md 21: 9824 bytes per solve in Float32, 13024 in Float64,
            # measured on Julia 1.13), which the sparse :auto and :cholesky use
            dampen!(ls, 0.3)
            (kind === :dense || s === :ldl) && @test pr_allocs(ls, static(s)) == 0
            t = DampedLinearSolve.copy_structure(ls)
            @test t isa typeof(ls) && systemeltype(t) === Float32 && getx(t) === getx(ls)
        end
    end
    # The checks: element type, and decision 10 for the sparse reduced system
    @test DampedLinearSolve.checkhessian(pr_hessian(:dense, p), n) === nothing
    H16 = let H = dense_schur_hessian(p); SchurBlockHessian(H.structure, Float16.(H.R), Float16.(H.E_data), Float16.(H.C_data)) end
    err = try DampedLinearSolve.checkhessian(H16, n); nothing catch e; e end
    @test err isa ArgumentError && occursin("SchurBlockHessian{Float32}", err.msg)
    err = try makehessiansystem(H16, :auto); nothing catch e; e end
    @test err isa ArgumentError && occursin("Float64 or Float32", err.msg)
    if !PR_CHOLMOD32
        for s in (:auto, :cholesky)
            err = try makehessiansystem(pr_hessian(:sparse, p), s); nothing catch e; e end
            @test err isa ArgumentError && occursin("does not offer Float32", err.msg) && occursin(":ldl", err.msg)
        end
    end
    # The indefiniteness tolerance of the sparse reduced LDLᵀ follows the element type
    @test BlockLinearSystems.indefinite_rtol(Float64) === 1e-6
    @test BlockLinearSystems.indefinite_rtol(Float32) ≈ Float64(eps(Float32))^(3 / 8)
    # The Float64 system is unchanged in kind
    @test systemeltype(makehessiansystem(dense_schur_hessian(p), :auto)) === Float64
end

end # module
