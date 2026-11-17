module BlockLinearSystemsExt

# Schur-complement linear systems for bipartite problems, built on the
# block data structures and reduction kernels of BlockLinearSystems.jl. Loaded when both packages are.

using SecondOrderOptimization
using DampedLinearSolve
import DampedLinearSolve: AbstractLinearSystem, solve!, dampen!, sharex, getA, getb, getx, SolveStatus, Success, NotPositiveDefinite, Indefinite, makelinearsystem, max_diag, is_indefinite
import DampedLinearSolve: AbstractTallLinearSystem, DenseLDLWorkspace, is_indefinite_packed, solvermismatch, make_reduced, solve_reduced!, SPQRWorkspace, spqr_solve!
import DampedLinearSolve.LDLFactorizations: ldl_analyze, ldl_factorize!, factorized
import BlockLinearSystems
import BlockLinearSystems: SchurBlockJacobian, QRReductionWorkspace, reduce_qr!, reduced_nrows, reduced_jacobian!, build_reduced_jacobian_pattern, reduced_normal_dense!, reduced_normal_sparse!, free_block_pairs, jtj_diag!, jtj_quadform
import BlockLinearSystems: SchurBlockHessian, SchurReductionWorkspace, SparseBlockLinearSystem, reset_schur_dense!, reset_schur_sparse!, reduce_schur_dense!, reduce_schur_sparse!, backsubstitute!, build_schur_pattern, build_r_to_s_map, retained_block_pairs, trial_workspace
using LinearAlgebra, SparseArrays, Static

# Schur-complement systems for a SchurBlockHessian
include("DampedLinearSolveExt/schurhessian.jl")

# QR elimination on the Jacobian path
include("DampedLinearSolveExt/qrelimination.jl")

# The conformance hooks of both
include("DampedLinearSolveExt/conformance.jl")

end # module BlockLinearSystemsExt
