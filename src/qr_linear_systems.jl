# QR elimination on the Jacobian path (TP3): the linear system of a JacobianObjective whose prototype is a
# SchurBlockJacobian{T, RType}, T Float64 or Float32 (the tables, their elimination and the reduced solve in T; the
# residual, the gradient, x and every query in Float64; the residual column goes through the elimination and into the
# reduced problem in Float64, the T reflectors applied to it in Float64). Per damping value, `reduce_qr!` eliminates every eliminated block from the
# raw tables (with its damping and scaling), the reduced problem min ‖J̃ x_r − r̃‖² + λ Σ s_r x_r² is solved
# for the retained variables, and `backsubstitute!` gives x_e. RType, set by whoever built the prototype,
# chooses the storage of the reduced system, and `linsolver` the solver applied to it (validated up front,
# `check_jacobian_linsolver`):
#
#   RType               :auto            :qr              :cholesky / :ldl            :lsmr (with Krylov)
#   Matrix{T}           dense QR of J̃    dense QR of J̃    S = J̃ᵀJ̃, dense factor      LSMR on J̃
#   SparseMatrixCSC     SPQR of J̃        SPQR of J̃        S = J̃ᵀJ̃, sparse factor     LSMR on J̃
#
# For T = Float32 (decision 10), SPQR is not offered (the sparse :auto and :qr are an ArgumentError), nor the sparse
# :cholesky where CHOLMOD is Float64 only (SparseArrays 1.10).
#
# The sparse J̃ has a fixed pattern (BlockLinearSystems' `build_reduced_jacobian_pattern`), built once and filled
# per damping value by `reduced_jacobian!`; its damped least-squares problem goes through the sparse Jacobian
# system's SPQR route (`spqr_solve!`: symbolic analysis once, numeric factorization per solve). The gradient, the
# column norms and ‖J v‖² come from the raw tables (`jtr!`, `jtj_diag!`, `jtj_quadform`): Q is orthogonal, so
# nothing else is needed.

"""
    SchurJacobianHessian

The Hessian storage of a QR-elimination system: the raw tables `J` of a `SchurBlockJacobian`, standing for
`A = JᵀJ`, which is never formed. The residual is `J.r`, filled through `residual_block`; `r` is an empty vector,
handed to `resjac!` for its first argument.
"""
mutable struct SchurJacobianHessian{T, RType}
    const J::SchurBlockJacobian{T, RType}
    const r::Vector{Float64}           # empty: the residuals are in J.r
    const colnorm2::Vector{Float64}    # diag(JᵀJ), cached per linearization, accumulated in Float64
    normsstale::Bool
end
SchurJacobianHessian(J::SchurBlockJacobian) = SchurJacobianHessian(J, Float64[], zeros(jacobian_ncols(J)), true)

# The prototype: its checks, column count and workspace key
function DampedLinearSolve.checkjacobian(J::SchurBlockJacobian{T}) where T
    T in REDUCED_ELTYPES || throw(ArgumentError("The Jacobian must be Float64 or Float32; got a SchurBlockJacobian{$T}. A SchurBlockJacobian prototype is a SchurBlockJacobian{T, Matrix{T}} or SchurBlockJacobian{T, SparseMatrixCSC{T, Int}} with T Float64 or Float32, linsolver = :auto, :qr, :cholesky or :ldl, or :lsmr with Krylov.jl loaded (see the solvers each offers for Float32)."))
    return nothing
end
DampedLinearSolve.jacobian_ncols(J::SchurBlockJacobian) = jacobian_ncols(J)
DampedLinearSolve.jacobian_nrows(J::SchurBlockJacobian) = jacobian_nrows(J)
# A workspace is reused for a prototype of the same type and the same (identical) structure
DampedLinearSolve.structurekey(J::SchurBlockJacobian) = (typeof(J), J.structure)

jacobian_ncols(J::SchurBlockJacobian) = J.structure.schur.nr + sum(Int, J.structure.schur.e_dims; init=0)
jacobian_nrows(J::SchurBlockJacobian) = sum(t -> data_rows(J.structure, t), 1:J.structure.ntables; init=0)

Base.size(A::SchurJacobianHessian) = (n = length(A.colnorm2); (n, n))
Base.size(A::SchurJacobianHessian, d::Integer) = size(A)[d]
DampedLinearSolve.markstale!(A::SchurJacobianHessian) = (A.normsstale = true; A)
DampedLinearSolve.quadform(A::SchurJacobianHessian, v::AbstractVector) = jtj_quadform(A.J, v)
function colnorms2!(A::SchurJacobianHessian)
    if A.normsstale
        jtj_diag!(A.colnorm2, A.J)
        A.normsstale = false
    end
    return A.colnorm2
end
function DampedLinearSolve.max_diag(A::SchurJacobianHessian)
    md = 0.0
    for d in colnorms2!(A)
        md = max(md, d)
    end
    return md
end

# The storage the linearization writes (the tables, which hold J and r, and an empty r), and the gradient
DampedLinearSolve.getJ(A::SchurJacobianHessian) = A.J
DampedLinearSolve.getr(A::SchurJacobianHessian) = A.r
DampedLinearSolve.gradient!(g::AbstractVector, A::SchurJacobianHessian) = BlockLinearSystems.jtr!(g, A.J)

# The reduced systems, built once with the system. Each but the sparse normal equations is a DampedLinearSolve system,
# filled per solve from the QR-reduction workspace and solved through the contract:
#   DenseQRReduced: min ‖[J̃; √(λ s_r)] x_r − [r̃; 0]‖ as an undamped dense Jacobian system of m̃ + n_r rows, so that its one
#     Householder QR is of the whole augmented matrix (the damping changes J̃, so nothing is kept between solves);
#   DenseNormalReduced: S = J̃ᵀJ̃ + λ diag(s_r) (upper triangle) as an undamped dense system;
#   SparseQRReduced: J̃ (fixed pattern) and r̃ as a sparse Jacobian system damped by λ s_r (SPQR, analysed once);
#   SparseNormalReduced: S (upper triangle) solved here, as the sparse Schur system is (see `sparse_reduced_solve!`);
#   :lsmr: the system the Krylov extension (KrylovExt) builds for a QRReductionWorkspace (matrix-free LSMR on J̃).
struct DenseQRReduced{S}
    sys::S
end
struct DenseNormalReduced{S}
    sys::S
end
struct SparseQRReduced{S}
    sys::S
    Jt::SparseMatrixCSC{Float64, Int}   # getJ(getA(sys)): J̃, its pattern fixed
    rt::Vector{Float64}                 # getr(getA(sys)): r̃
end
struct SparseNormalReduced{T, TF, TL, TC, P}
    S::SparseMatrixCSC{T, Int}
    g::Vector{Float64}         # J̃ᵀr̃, accumulated in Float64 from the T J̃ and the Float64 r̃ (SP-B6)
    rhs::Vector{T}             # g narrowed: the factorizations solve in T
    sol::Vector{T}
    pattern::P
    diag_inds::Vector{Int}
    S_chol::TC                 # CHOLMOD copy of S, refreshed in place
    S_factor::TF
    S_factor_ldl::TL
end

const ReducedSym = Union{StaticSymbol{:auto}, StaticSymbol{:qr}, StaticSymbol{:cholesky}, StaticSymbol{:ldl}, StaticSymbol{:lsmr}}

reduced_system(J::SchurBlockJacobian, ws, linsolver) = reduced_system(J, ws, static(linsolver_symbol(linsolver)))
# The dense reduced systems are DampedLinearSolve systems built from a T prototype
function reduced_system(J::SchurBlockJacobian{T, Matrix{T}}, ws, ::Union{StaticSymbol{:auto}, StaticSymbol{:qr}}) where T
    jst = J.structure
    nr, mt = jst.schur.nr, reduced_nrows(jst)
    sys = DampedLinearSolve.makejacobiansystem(zeros(T, mt + nr, nr), nr, static(:qr))
    dampen!(sys, 0.0)
    return DenseQRReduced(sys)
end
function reduced_system(J::SchurBlockJacobian{T, Matrix{T}}, ws, ::Union{StaticSymbol{:cholesky}, StaticSymbol{:ldl}}) where T
    sys = makehessiansystem(zeros(T, J.structure.schur.nr, J.structure.schur.nr), static(:ldl))   # holds the Bunch-Kaufman workspace
    return DenseNormalReduced(sys)
end
# QR whatever the storage: `:auto` is QR of J̃ for a sparse RType too (decided 2026-11-15, TP6)
function reduced_system(J::SchurBlockJacobian{Float64, SparseMatrixCSC{Float64, Int}}, ws, ::Union{StaticSymbol{:auto}, StaticSymbol{:qr}})
    Jt = build_reduced_jacobian_pattern(J.structure, Float64)
    sys = DampedLinearSolve.makejacobiansystem(Jt, J.structure.schur.nr, static(:qr))
    return SparseQRReduced(sys, getJ(getA(sys)), getr(getA(sys)))
end
function reduced_system(J::SchurBlockJacobian{T, SparseMatrixCSC{T, Int}}, ws, s::Union{StaticSymbol{:cholesky}, StaticSymbol{:ldl}}) where T
    jst = J.structure
    pattern = build_schur_pattern(jst.schur, T; extra_blocks = free_block_pairs(jst))
    S = copy(pattern.S_sparse)
    # CHOLMOD only where it offers T (for :ldl it is never used)
    usechol = s isa StaticSymbol{:cholesky} && reduced_cholmod(T)
    S_chol = usechol ? SparseArrays.CHOLMOD.Sparse(Symmetric(S, :U)) : nothing
    S_factor = usechol ? SparseArrays.CHOLMOD.analyze(S_chol) : nothing
    S_factor_ldl = ldl_analyze(Symmetric(S, :U))
    nr = jst.schur.nr
    return SparseNormalReduced(S, zeros(nr), zeros(T, nr), zeros(T, nr), pattern, collect(Int, pattern.diag_inds), S_chol, S_factor, S_factor_ldl)
end
# LSMR on J̃, matrix-free: the Krylov extension (KrylovExt) adds the system, for the reduction workspace it reads J̃ from
function reduced_system(J::SchurBlockJacobian, ws, s::StaticSymbol{:lsmr})
    hasmethod(DampedLinearSolve.makejacobiansystem, Tuple{typeof(ws), Int, typeof(s)}) || throw(ArgumentError(linsolver_message(J, :lsmr)))
    return DampedLinearSolve.makejacobiansystem(ws, J.structure.schur.nr, s)
end
reduced_system(J::SchurBlockJacobian, ws, ::StaticSymbol{S}) where S = throw(ArgumentError(linsolver_message(J, S)))

# The solvers accepted for each reduced storage, checked before anything is built or called. For Float32 (decision 10):
# SPQR is Float64 only, so the sparse :auto and :qr are not offered, and the sparse :cholesky only where CHOLMOD offers
# Float32 (SparseArrays 1.13 and later); LSMR on J̃ is matrix-free in T (KrylovExt).
reduced_linsolvers(::Type{Matrix{T}}) where T = (:auto, :qr, :cholesky, :ldl, :lsmr)
reduced_linsolvers(::Type{SparseMatrixCSC{Float64, Int}}) = (:auto, :qr, :cholesky, :ldl, :lsmr)
reduced_linsolvers(::Type{SparseMatrixCSC{T, Int}}) where T = reduced_cholmod(T) ? (:cholesky, :ldl, :lsmr) : (:ldl, :lsmr)
function linsolver_message(J::SchurBlockJacobian{T, RType}, s) where {T, RType}
    ok = join((":$x" for x in reduced_linsolvers(RType)), ", ", " or ")
    T === Float64 || s in reduced_linsolvers(SparseMatrixCSC{Float64, Int}) &&
        return "linsolver = :$s does not offer $T for a SchurBlockJacobian{$T, $RType} Jacobian prototype (SPQR is Float64 only, and CHOLMOD is in SparseArrays $(pkgversion(SparseArrays)) on Julia $VERSION only if it offers $T); the solvers that offer $T for it are $ok (:lsmr with Krylov.jl loaded)."
    return "linsolver = :$s is not supported with a SchurBlockJacobian{$T, $RType} Jacobian prototype; use $ok (:lsmr with Krylov.jl loaded)."
end
function DampedLinearSolve.check_jacobian_linsolver(J::SchurBlockJacobian{T, RType}, s::Symbol) where {T, RType}
    s in reduced_linsolvers(RType) || throw(ArgumentError(linsolver_message(J, s)))
    return nothing
end

"""
    QRSchurLinearSystem

The linear system of a [`JacobianObjective`](@ref) with a `SchurBlockJacobian` prototype: the normal equations
`(JᵀJ + λD) x = Jᵀr` solved by QR elimination of the eliminated blocks and a solve of the reduced problem in the
retained variables. Nothing allocates after construction.
"""
mutable struct QRSchurLinearSystem{T, RType, R} <: AbstractLinearSystem
    const A::SchurJacobianHessian{T, RType}
    const b::Vector{Float64}           # Jᵀr
    const x::Vector{Float64}
    const ws::QRReductionWorkspace{T}
    const reduced::R                   # the reduced system
    const linsolver::Symbol            # the solver the reduced system was built for
    const unit::Vector{Float64}        # ones: the scaling of uniform damping
    const xr::Vector{Float64}          # x_r, n_r
    const xe::Vector{Float64}          # x_e
    const sr::Vector{Float64}          # s_r, the retained part of the scaling, for a reduced system that is damped
    lambda::Float64
    scaling::Union{Nothing, Vector{Float64}}
end
function QRSchurLinearSystem(J::SchurBlockJacobian, linsolver)
    n = jacobian_ncols(J)
    nr = J.structure.schur.nr
    ws = QRReductionWorkspace(J)
    return QRSchurLinearSystem(SchurJacobianHessian(J), zeros(n), zeros(n), ws, reduced_system(J, ws, linsolver),
                               linsolver_symbol(linsolver), ones(n), zeros(nr), zeros(n - nr), zeros(nr), 0.0, nothing)
end
DampedLinearSolve.isfinitesystem(ls::QRSchurLinearSystem) = all(isfinite, ls.A.J.data) && all(isfinite, ls.A.J.r)

function DampedLinearSolve.makejacobiansystem(J::SchurBlockJacobian, n, linsolver)
    DampedLinearSolve.checkjacobian(J)
    DampedLinearSolve.check_jacobian_linsolver(J, linsolver_symbol(linsolver))
    return QRSchurLinearSystem(J, linsolver)
end
DampedLinearSolve.systemeltype(::QRSchurLinearSystem{T}) where T = T

# A new linearization is a new fill of the dense normal-equation reduced system too: its conditioning scale is reset as
# for the dense Schur system (the QR reduced systems are marked stale by every solve)
function DampedLinearSolve.zero!(ls::QRSchurLinearSystem)
    fill!(ls.b, 0.0)
    DampedLinearSolve.markstale!(ls.A)
    ls.reduced isa DenseNormalReduced && DampedLinearSolve.zero!(ls.reduced.sys)
    return ls
end

# The trial system: its own tables (with the same entries), factored tables and gradient; the reduced system and the
# scratch are shared, since only one of the two is solved at a time and every solve refills the reduced system. LSMR's
# reduced system reads J̃ from its own reduction workspace, so the trial system has its own.
function DampedLinearSolve.copy_structure(ls::QRSchurLinearSystem{T, RType}) where {T, RType}
    J = ls.A.J
    At = SchurJacobianHessian(SchurBlockJacobian{T, RType}(copy(J.data), copy(J.r), J.structure))
    ws = trial_workspace(ls.ws)
    reduced = ls.linsolver === :lsmr ? reduced_system(J, ws, static(:lsmr)) : ls.reduced
    return QRSchurLinearSystem(At, zeros(length(ls.b)), ls.x, ws, reduced, ls.linsolver, ls.unit, ls.xr, ls.xe, ls.sr, ls.lambda, ls.scaling)
end
DampedLinearSolve.sharex(ls::QRSchurLinearSystem, from) = ls.x === from.x ? ls :
    QRSchurLinearSystem(ls.A, ls.b, from.x, ls.ws, ls.reduced, ls.linsolver, ls.unit, ls.xr, ls.xe, ls.sr, ls.lambda, ls.scaling)

# The QR routes resolve a damping down to about eps² of the diagonal (as the Jacobian systems do); those that form
# S = J̃ᵀJ̃ only to about eps (as Cholesky of the normal equations)
DampedLinearSolve.dampingfloor(ls::QRSchurLinearSystem{T}) where T = Float64(eps(T))^2
DampedLinearSolve.dampingfloor(ls::QRSchurLinearSystem{T, <:Any, <:Union{DenseNormalReduced, SparseNormalReduced}}) where T = Float64(eps(T))

# The diagonal of JᵀJ, the squared column norms of J (from the tables, once per linearization)
DampedLinearSolve.getdiag!(d::AbstractVector, ls::QRSchurLinearSystem) = copyto!(d, colnorms2!(ls.A))

scaling_of(ls::QRSchurLinearSystem) = isnothing(ls.scaling) ? ls.unit : ls.scaling

function DampedLinearSolve.solve!(ls::QRSchurLinearSystem, s::ReducedSym)
    J = ls.A.J
    jst = J.structure
    nr = jst.schur.nr
    lambda = max(ls.lambda, 0.0)
    sc = scaling_of(ls)
    # A failed block (non-finite, or numerically singular, as an under-determined block is undamped) is reported
    # as the Schur systems report a singular eliminated block
    reduce_qr!(ls.ws, J, view(sc, nr+1:length(sc)), lambda) || return NotPositiveDefinite
    status = reduced_solve!(ls, ls.reduced, s, lambda, view(sc, 1:nr))
    status == NotPositiveDefinite && return status
    # An Indefinite solve (the normal equations' LDLᵀ) has solved too: the whole of x, as DampedLinearSolve's
    # contract requires
    backsubstitute!(ls.xe, ls.xr, ls.ws)
    @inbounds for i in 1:nr
        ls.x[i] = ls.xr[i]
    end
    @inbounds for i in eachindex(ls.xe)
        ls.x[nr + i] = ls.xe[i]
    end
    all(isfinite, ls.x) || return NotPositiveDefinite
    return status
end
DampedLinearSolve.solve!(ls::QRSchurLinearSystem, ::StaticSymbol{S}) where S = reducedmismatch(ls, S)
@noinline reducedmismatch(ls, asked) = throw(ArgumentError("solve! was asked for linsolver = :$asked, but this QRSchurLinearSystem was built for linsolver = :$(ls.linsolver). A linear system must be solved with the linsolver it was built with."))
reduced_solve!(ls, red, s::StaticSymbol{S}, lambda, s_r) where S = reducedmismatch(ls, S)

# Dense: [J̃ r̃; √(λ s_r) 0] written into the undamped Jacobian system, whose QR of the whole matrix then gives R x_r = q
function reduced_solve!(ls::QRSchurLinearSystem, red::DenseQRReduced, ::Union{StaticSymbol{:auto}, StaticSymbol{:qr}}, lambda, s_r)
    nr = length(ls.xr)
    sys = red.sys
    A = getA(sys)
    Jm, r = getJ(A), getr(A)
    mt = size(Jm, 1) - nr
    reduced_jacobian!(view(Jm, 1:mt, 1:nr), view(r, 1:mt), ls.ws)
    @inbounds for j in 1:nr, i in mt+1:mt+nr
        Jm[i, j] = 0.0
    end
    @inbounds for i in mt+1:mt+nr
        r[i] = 0.0
    end
    @inbounds for i in 1:nr
        Jm[mt + i, i] = sqrt(lambda * s_r[i])   # formed in Float64, narrowed to the system's element type
    end
    markstale!(A)
    status = solve!(sys, static(:qr))
    status == NotPositiveDefinite || copyto!(ls.xr, getx(sys))
    return status
end

# Sparse J̃ and r̃, then the damped least-squares problem by SPQR, as for a sparse Jacobian
function reduced_solve!(ls::QRSchurLinearSystem, red::SparseQRReduced, ::Union{StaticSymbol{:auto}, StaticSymbol{:qr}}, lambda, s_r)
    reduced_jacobian!(red.Jt, red.rt, ls.ws)
    markstale!(getA(red.sys))
    copyto!(ls.sr, s_r)
    dampen!(red.sys, lambda, ls.sr)
    status = solve!(red.sys, static(:qr))
    status == NotPositiveDefinite || copyto!(ls.xr, getx(red.sys))
    return status
end

# S = J̃ᵀJ̃ + λ diag(s_r) (upper triangle), then the dense Schur system's rule: Cholesky, falling back to
# Bunch-Kaufman, or Bunch-Kaufman for :ldl
function reduced_solve!(ls::QRSchurLinearSystem, red::DenseNormalReduced, s::Union{StaticSymbol{:cholesky}, StaticSymbol{:ldl}}, lambda, s_r)
    sys = red.sys
    S, g = getA(sys), getb(sys)
    reduced_normal_dense!(S, g, ls.ws)
    @inbounds for i in eachindex(s_r)
        S[i, i] += lambda * s_r[i]
    end
    all(isfinite, S) || return NotPositiveDefinite
    status = solve_reduced_dense!(sys, s)
    status == NotPositiveDefinite || copyto!(ls.xr, getx(sys))   # an Indefinite solve has solved, too
    return status
end

# Any other reduced system (LSMR's): damped by λ s_r and solved through the contract
function reduced_solve!(ls::QRSchurLinearSystem, red::AbstractLinearSystem, s::StaticSymbol{:lsmr}, lambda, s_r)
    copyto!(ls.sr, s_r)
    dampen!(red, lambda, ls.sr)
    status = solve!(red, s)
    status == NotPositiveDefinite || copyto!(ls.xr, getx(red))
    return status
end

# Sparse S, then CHOLMOD Cholesky falling back to LDLᵀ, or LDLᵀ for :ldl: the sparse Schur system's rule
# The sparse reduced normal equations are solved here, not through a DampedLinearSolve sparse system: the reduction
# writes one triangle of S, which a DampedLinearSolve sparse system (full symmetric storage) would need mirrored every
# solve, and its LDLᵀ is LDLFactorizations' full-matrix path, not the upper-triangle one used here, so the iterates
# would change in rounding. The persistent CHOLMOD copy also saves a conversion per solve.
function reduced_solve!(ls::QRSchurLinearSystem, red::SparseNormalReduced, s::Union{StaticSymbol{:cholesky}, StaticSymbol{:ldl}}, lambda, s_r)
    reduced_normal_sparse!(red.S, red.g, ls.ws, red.pattern)
    nz = nonzeros(red.S)
    @inbounds for i in eachindex(s_r)
        nz[red.diag_inds[i]] += lambda * s_r[i]   # formed in Float64, narrowed
    end
    all(isfinite, nz) || return NotPositiveDefinite
    copyto!(red.rhs, red.g)
    if !(s isa StaticSymbol{:ldl}) && !isnothing(red.S_chol)
        refresh_cholmod!(red.S_chol, red.S)
        LinearAlgebra.cholesky!(red.S_factor, red.S_chol; check = false)
        if LinearAlgebra.issuccess(red.S_factor)
            cholmod_ldiv!(red.sol, red.S_factor, red.rhs)
            copyto!(ls.xr, red.sol)
            return Success
        end
    end
    F = ldl_factorize!(Symmetric(red.S, :U), red.S_factor_ldl)
    factorized(F) || return NotPositiveDefinite
    ldiv!(red.sol, F, red.rhs)
    copyto!(ls.xr, red.sol)
    return is_indefinite(F) ? Indefinite : Success
end
