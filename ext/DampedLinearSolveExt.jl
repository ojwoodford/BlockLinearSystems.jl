module DampedLinearSolveExt

# The linear-system contract of DampedLinearSolve.jl, for the Schur-complement systems of a SchurBlockHessian and
# the QR-eliminated systems of a SchurBlockJacobian, built on this package's block storage and reduction kernels.
# Loaded when DampedLinearSolve is (LDLFactorizations, a dependency of DampedLinearSolve, is then loaded too).

using DampedLinearSolve
import DampedLinearSolve: AbstractLinearSystem, solve!, dampen!, sharex, getA, getb, getx, getJ, getr, markstale!, SolveStatus, Success, NotPositiveDefinite, Indefinite, makehessiansystem, max_diag
import LDLFactorizations
import LDLFactorizations: ldl_analyze, ldl_factorize!, factorized
import BlockLinearSystems
import BlockLinearSystems: SchurBlockJacobian, QRReductionWorkspace, reduce_qr!, reduced_nrows, reduced_jacobian!, build_reduced_jacobian_pattern, reduced_normal_dense!, reduced_normal_sparse!, free_block_pairs, jtj_diag!, jtj_quadform
import BlockLinearSystems: SchurBlockHessian, SchurReductionWorkspace, SparseBlockLinearSystem, reset_schur_dense!, reset_schur_sparse!, reduce_schur_dense!, reduce_schur_sparse!, backsubstitute!, build_schur_pattern, build_r_to_s_map, retained_block_pairs, trial_workspace
using LinearAlgebra, SparseArrays, Static

# A solver is named by a Symbol or a StaticSymbol
linsolver_symbol(s::Symbol) = s
linsolver_symbol(s::StaticSymbol) = dynamic(s)

# Indefiniteness of an LDLᵀ factorization from its pivots: a pivot below -rtol times the largest pivot magnitude
# (the test DampedLinearSolve applies to its own LDLᵀ)
function is_indefinite(F::LDLFactorizations.LDLFactorization, rtol::Float64=1e-6)
    maxabs = 0.0
    minpivot = Inf
    @inbounds for d in F.d
        maxabs = max(maxabs, abs(d))
        minpivot = min(minpivot, d)
    end
    return minpivot < -rtol * maxabs
end

# Schur-complement systems for a SchurBlockHessian
include("DampedLinearSolveExt/schurhessian.jl")

# QR elimination on the Jacobian path
include("DampedLinearSolveExt/qrelimination.jl")

# The conformance hooks of both
include("DampedLinearSolveExt/conformance.jl")

end # module DampedLinearSolveExt
