module DampedLinearSolveKrylovExt

# LSMR on the reduced Jacobian J̃ of QR elimination: loaded when both DampedLinearSolve and Krylov are. It adds the
# reduced system DampedLinearSolveExt's QR-elimination system builds for linsolver = :lsmr, as a method of
# DampedLinearSolve's `maketallsystem` for the QRReductionWorkspace that holds J̃, so neither extension names the
# other's types. It stays matrix-free: J̃ is applied through the factored tables, never formed.

using DampedLinearSolve
using BlockLinearSystems
using Krylov
using LinearAlgebra
using Static

import DampedLinearSolve: AbstractLinearSystem, Success, NotPositiveDefinite
import BlockLinearSystems: QRReductionWorkspace, reduced_nrows, reduced_mul!, reduced_tmul!, reduced_residual!

# LSMR's stopping tolerances: Krylov's own defaults, stated here so that a change in Krylov does not change them, and
# equal to those of DampedLinearSolve's LSMR on a Jacobian. Starting values, not measured (rules.md 26; improvements.md
# item 5). itmax is Krylov's default, m̃ + n_r iterations.
const LSMR_ETOL = sqrt(eps(Float64))
const LSMR_AXTOL = sqrt(eps(Float64))
const LSMR_BTOL = sqrt(eps(Float64))
const LSMR_CONLIM = 1 / sqrt(eps(Float64))

# The reduced problem min ‖J̃ x_r − r̃‖² + λ Σ s_r x_r², as LSMR's regularized problem: λₗ = √λ and N = Diagonal(1 ./ s_r).
# The reduced system of the QR-eliminated system: it implements what that system calls (dampen!, solve!, getx), with
# r̃ and J̃ read from the reduction workspace, which the QR-eliminated system has factored before each solve.
mutable struct LsmrReducedSystem{W} <: AbstractLinearSystem
    const ws::QRReductionWorkspace{Float64}
    const kws::W                 # Krylov's workspace
    const rt::Vector{Float64}    # r̃
    const invscale::Vector{Float64}
    const x::Vector{Float64}
    lambda::Float64
    scaling::Union{Nothing, Vector{Float64}}
end
DampedLinearSolve.dampen!(red::LsmrReducedSystem, lambda::Real) = (red.lambda = Float64(lambda); red.scaling = nothing; red)
DampedLinearSolve.dampen!(red::LsmrReducedSystem, lambda::Real, scaling) = (red.lambda = Float64(lambda); red.scaling = scaling; red)
function DampedLinearSolve.maketallsystem(ws::QRReductionWorkspace{Float64}, nr::Integer, ::StaticSymbol{:lsmr})
    mt = reduced_nrows(ws.structure)
    return LsmrReducedSystem(ws, Krylov.LsmrWorkspace(mt, Int(nr), Vector{Float64}), zeros(mt), zeros(Int(nr)), zeros(Int(nr)), 0.0, nothing)
end

# J̃ (and its adjoint) over a QR-elimination workspace whose tables are factored
struct ReducedOperator{W}
    ws::W
    m::Int
    n::Int
end
struct ReducedOperatorT{W}
    ws::W
    m::Int
    n::Int
end
Base.size(A::ReducedOperator) = (A.m, A.n)
Base.size(A::ReducedOperatorT) = (A.n, A.m)
Base.size(A::Union{ReducedOperator, ReducedOperatorT}, d::Integer) = size(A)[d]
Base.eltype(::Union{ReducedOperator, ReducedOperatorT}) = Float64
Base.adjoint(A::ReducedOperator) = ReducedOperatorT(A.ws, A.m, A.n)
LinearAlgebra.mul!(y::AbstractVector, A::ReducedOperator, v::AbstractVector) = reduced_mul!(y, A.ws, v)
LinearAlgebra.mul!(y::AbstractVector, A::ReducedOperatorT, w::AbstractVector) = reduced_tmul!(y, A.ws, w)

function DampedLinearSolve.solve!(red::LsmrReducedSystem, ::StaticSymbol{:lsmr})
    mt, nr = length(red.rt), length(red.x)
    reduced_residual!(red.rt, red.ws)
    all(isfinite, red.rt) || return NotPositiveDefinite
    lambda = max(red.lambda, 0.0)
    # Undamped with fewer reduced rows than retained variables, J̃ is rank deficient
    lambda == 0 && mt < nr && return NotPositiveDefinite
    op = ReducedOperator(red.ws, mt, nr)
    if lambda > 0
        s_r = red.scaling
        @inbounds for i in 1:nr
            red.invscale[i] = isnothing(s_r) ? 1.0 : inv(s_r[i])
        end
        Krylov.lsmr!(red.kws, op, red.rt; λ = sqrt(lambda), N = Diagonal(red.invscale),
                     etol = LSMR_ETOL, axtol = LSMR_AXTOL, btol = LSMR_BTOL, conlim = LSMR_CONLIM)
    else
        Krylov.lsmr!(red.kws, op, red.rt; etol = LSMR_ETOL, axtol = LSMR_AXTOL, btol = LSMR_BTOL, conlim = LSMR_CONLIM)
    end
    x = red.kws.x
    (red.kws.stats.solved && all(isfinite, x)) || return NotPositiveDefinite
    copyto!(red.x, x)
    return Success
end

end # module DampedLinearSolveKrylovExt
