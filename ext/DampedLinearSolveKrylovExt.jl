module BlockLinearSystemsKrylovExt

# LSMR on the reduced Jacobian J̃ of QR elimination (TP3): loaded when both BlockLinearSystems and Krylov are.
# It adds a reduced solver to BlockLinearSystemsExt's QR-elimination system through the core's `make_reduced`
# and `solve_reduced!`, without naming that extension's types.

using SecondOrderOptimization
using DampedLinearSolve
using BlockLinearSystems
using Krylov
using LinearAlgebra
using Static

import DampedLinearSolve: make_reduced, solve_reduced!, builtfor, Success, NotPositiveDefinite
import DampedLinearSolve: LSMR_ETOL, LSMR_AXTOL, LSMR_BTOL, LSMR_CONLIM
import BlockLinearSystems: SchurBlockJacobian, reduced_nrows, reduced_mul!, reduced_tmul!, reduced_residual!

# The reduced problem min ‖J̃ x_r − r̃‖² + λ Σ s_r x_r², as LSMR's regularized problem: λₗ = √λ and
# N = Diagonal(1 ./ s_r) (as in KrylovExt). J̃ is applied through the factored tables, never formed.
struct LsmrReduced{W}
    ws::W
    rt::Vector{Float64}          # r̃
    invscale::Vector{Float64}
end
function make_reduced(J::SchurBlockJacobian{Float64}, ::StaticSymbol{:lsmr})
    jst = J.structure
    mt, nr = reduced_nrows(jst), jst.schur.nr
    return LsmrReduced(Krylov.LsmrWorkspace(mt, nr, Vector{Float64}), zeros(mt), zeros(nr))
end
builtfor(::LsmrReduced) = ":lsmr"

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

function solve_reduced!(ls, red::LsmrReduced, ::StaticSymbol{:lsmr}, lambda, s_r)
    mt, nr = length(red.rt), length(ls.xr)
    reduced_residual!(red.rt, ls.ws)
    all(isfinite, red.rt) || return NotPositiveDefinite
    # Undamped with fewer reduced rows than retained variables, J̃ is rank deficient
    lambda == 0 && mt < nr && return NotPositiveDefinite
    op = ReducedOperator(ls.ws, mt, nr)
    if lambda > 0
        @inbounds for i in 1:nr
            red.invscale[i] = inv(s_r[i])
        end
        Krylov.lsmr!(red.ws, op, red.rt; λ = sqrt(lambda), N = Diagonal(red.invscale),
                     etol = LSMR_ETOL, axtol = LSMR_AXTOL, btol = LSMR_BTOL, conlim = LSMR_CONLIM)
    else
        Krylov.lsmr!(red.ws, op, red.rt; etol = LSMR_ETOL, axtol = LSMR_AXTOL, btol = LSMR_BTOL, conlim = LSMR_CONLIM)
    end
    x = red.ws.x
    (red.ws.stats.solved && all(isfinite, x)) || return NotPositiveDefinite
    copyto!(ls.xr, x)
    return Success
end

end # module BlockLinearSystemsKrylovExt
