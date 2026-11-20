module KrylovExt

# LSMR on the reduced Jacobian J̃ of QR elimination: loaded when Krylov is. It adds the reduced system this package's
# QR-elimination system builds for linsolver = :lsmr, as a method of DampedLinearSolve's `makejacobiansystem` for the
# QRReductionWorkspace that holds J̃, so the package names none of this extension's types. It stays matrix-free: J̃ is applied through the factored tables, never formed, and block-Jacobi
# preconditioned (LP1).

using DampedLinearSolve
using BlockLinearSystems
using Krylov
using LinearAlgebra
using Static

import DampedLinearSolve: AbstractLinearSystem, Success, NotPositiveDefinite
import BlockLinearSystems: QRReductionWorkspace, reduced_nrows, reduced_mul!, reduced_tmul!, reduced_residual!,
    reduced_diagonal_blocks!, factor_diagonal_blocks!, solve_diagonal_blocks!

# LSMR's stopping tolerances: Krylov's own defaults, stated here so that a change in Krylov does not change them, and
# equal to those of DampedLinearSolve's LSMR on a Jacobian. Starting values, not measured (rules.md 26; improvements.md
# item 5). itmax is Krylov's default, m̃ + n_r iterations.
const LSMR_ETOL = sqrt(eps(Float64))
const LSMR_AXTOL = sqrt(eps(Float64))
const LSMR_BTOL = sqrt(eps(Float64))
const LSMR_CONLIM = 1 / sqrt(eps(Float64))

# The reduced problem min ‖J̃ x_r − r̃‖² + λ Σ s_r x_r², solved by LSMR as the least-squares problem
#     min ‖Ā x_r − [r̃; 0]‖,  Ā = [J̃; Diagonal(√(λ s_r))]   (m̃ + n_r rows; with λ = 0 the damping rows are zero),
# with Krylov's regularization λₗ = 0 and its centered preconditioner N = D⁻¹, D the block diagonal of ĀᵀĀ =
# J̃ᵀJ̃ + λ Diagonal(s_r), one block per retained block (block-Jacobi; LP1). The retained variables of a block can
# differ in scale by orders of magnitude (focal length against rotation, say), which an unpreconditioned LSMR
# resolves only to its tolerance in the unscaled norm, or not within itmax. The Marquardt scaling stays in the
# damping rows: it is not replaced by N. itmax is set explicitly to m̃ + n_r.
# The reduced system of the QR-eliminated system: it implements what that system calls (dampen!, solve!, getx), with
# r̃, J̃ and the diagonal blocks read from the reduction workspace, which the QR-eliminated system has factored before
# each solve.
mutable struct LsmrReducedSystem{W} <: AbstractLinearSystem
    const ws::QRReductionWorkspace{Float64}
    const kws::W                 # Krylov's workspace, for m̃ + n_r rows
    const rhs::Vector{Float64}   # [r̃; 0]
    const sqd::Vector{Float64}   # √(λ s_r), the damping rows' diagonal
    const D::Vector{Float64}     # the diagonal blocks of ĀᵀĀ, then their Cholesky factors
    const x::Vector{Float64}
    const mt::Int
    lambda::Float64
    scaling::Union{Nothing, Vector{Float64}}
end
DampedLinearSolve.dampen!(red::LsmrReducedSystem, lambda::Real) = (red.lambda = Float64(lambda); red.scaling = nothing; red)
DampedLinearSolve.dampen!(red::LsmrReducedSystem, lambda::Real, scaling) = (red.lambda = Float64(lambda); red.scaling = scaling; red)
function DampedLinearSolve.makejacobiansystem(ws::QRReductionWorkspace{Float64}, nr::Integer, ::StaticSymbol{:lsmr})
    jst = ws.structure
    mt, n = reduced_nrows(jst), Int(nr)
    return LsmrReducedSystem(ws, Krylov.LsmrWorkspace(mt + n, n, Vector{Float64}), zeros(mt + n), zeros(n),
                             zeros(jst.doffsets[end] - 1), zeros(n), mt, 0.0, nothing)
end

# Ā (and its adjoint) over a system whose reduction workspace is factored and whose damping rows are set
struct AugmentedOperator{S}
    red::S
end
struct AugmentedOperatorT{S}
    red::S
end
Base.size(A::AugmentedOperator) = (A.red.mt + length(A.red.x), length(A.red.x))
Base.size(A::AugmentedOperatorT) = reverse(size(AugmentedOperator(A.red)))
Base.size(A::Union{AugmentedOperator, AugmentedOperatorT}, d::Integer) = size(A)[d]
Base.eltype(::Union{AugmentedOperator, AugmentedOperatorT}) = Float64
Base.adjoint(A::AugmentedOperator) = AugmentedOperatorT(A.red)
function LinearAlgebra.mul!(y::AbstractVector, A::AugmentedOperator, v::AbstractVector)
    red = A.red
    mt = red.mt
    reduced_mul!(view(y, 1:mt), red.ws, v)
    @inbounds for i in eachindex(red.sqd)
        y[mt + i] = red.sqd[i] * v[i]
    end
    return y
end
function LinearAlgebra.mul!(y::AbstractVector, A::AugmentedOperatorT, w::AbstractVector)
    red = A.red
    mt = red.mt
    reduced_tmul!(y, red.ws, view(w, 1:mt))
    @inbounds for i in eachindex(red.sqd)
        y[i] += red.sqd[i] * w[mt + i]
    end
    return y
end

# N = D⁻¹, applied with the factored blocks
struct BlockJacobi{S}
    red::S
end
Base.size(N::BlockJacobi) = (length(N.red.x), length(N.red.x))
Base.size(N::BlockJacobi, d::Integer) = size(N)[d]
Base.eltype(::BlockJacobi) = Float64
LinearAlgebra.mul!(y::AbstractVector, N::BlockJacobi, v::AbstractVector) =
    solve_diagonal_blocks!(y, N.red.D, N.red.ws.structure.schur.r_dims, v)

function DampedLinearSolve.solve!(red::LsmrReducedSystem, ::StaticSymbol{:lsmr})
    mt, nr = red.mt, length(red.x)
    rt = view(red.rhs, 1:mt)
    reduced_residual!(rt, red.ws)
    all(isfinite, rt) || return NotPositiveDefinite
    lambda = max(red.lambda, 0.0)
    # Undamped with fewer reduced rows than retained variables, J̃ is rank deficient
    lambda == 0 && mt < nr && return NotPositiveDefinite
    s_r = red.scaling
    @inbounds for i in 1:nr
        red.sqd[i] = sqrt(lambda * (isnothing(s_r) ? 1.0 : s_r[i]))
        red.rhs[mt + i] = 0.0
    end
    # D = the diagonal blocks of J̃ᵀJ̃, plus λ s_r on their diagonals; a block that is not positive definite (or not
    # finite) is reported as the other reduced solves report a singular reduced matrix
    st = red.ws.structure.schur
    reduced_diagonal_blocks!(red.D, red.ws)
    o, k = 0, 0
    @inbounds for db in st.r_dims
        d = Int(db)
        for i in 1:d
            red.D[o + (i - 1) * d + i] += red.sqd[k + i]^2
        end
        o += d * d
        k += d
    end
    factor_diagonal_blocks!(red.D, st.r_dims) || return NotPositiveDefinite
    Krylov.lsmr!(red.kws, AugmentedOperator(red), red.rhs; N = BlockJacobi(red), itmax = mt + nr,
                 etol = LSMR_ETOL, axtol = LSMR_AXTOL, btol = LSMR_BTOL, conlim = LSMR_CONLIM)
    x = red.kws.x
    (red.kws.stats.solved && all(isfinite, x)) || return NotPositiveDefinite
    copyto!(red.x, x)
    return Success
end

end # module KrylovExt
