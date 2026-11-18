module BlockLinearSystems

using Static, StaticArrays, HybridArrays, SparseArrays, LinearAlgebra
using SIRBSDispatch: sirbsdispatch
# The linear-system contract of DampedLinearSolve, implemented for the Schur-complement systems of a SchurBlockHessian
# and the QR-eliminated systems of a SchurBlockJacobian. Its own zero! and copy_structure are used qualified.
import DampedLinearSolve
import DampedLinearSolve: AbstractLinearSystem, solve!, dampen!, sharex, getA, getb, getx, getJ, getr, markstale!, SolveStatus, Success, NotPositiveDefinite, Indefinite, makehessiansystem, max_diag
import LDLFactorizations
import LDLFactorizations: ldl_analyze, ldl_factorize!, factorized

# Public API: exported, documented and listed in docs/src/api.md (test/exports.jl enforces this list).
# Everything else is internal: not exported, not in the API reference.
export SparseBlockLinearSystem, DenseBlockLinearSystem, TransposedPattern, BlockRLE
export block, blocksizes, nblocks, blockrange, zero!, copy_structure, uniformscaling!
export symmetrifyfull, symmetrifyfull!, sparse_with_map, block_sparse_nnz, computestartindices, find_diag_inds
export SchurStructure, SchurBlockHessian, SchurReductionWorkspace, trial_workspace, blockorder, nretained
export build_schur_pattern, build_r_to_s_map, retained_block_pairs
export reset_schur_dense!, reset_schur_sparse!, reduce_schur_dense!, reduce_schur_sparse!, backsubstitute!
export addblock!, blockentry
export SchurJacobianStructure, SchurBlockJacobian, schur_jacobian_bytes, residual_block
export QRReductionWorkspace, reduce_qr!
export reduced_nrows, free_block_pairs, reduced_mul!, reduced_tmul!, reduced_residual!, reduced_jacobian!,
    build_reduced_jacobian_pattern
export reduced_normal_dense!, reduced_normal_sparse!, reduced_diagonal_blocks!, factor_diagonal_blocks!, solve_diagonal_blocks!, jtr!, jtj_diag!, jtj_quadform
export threadargs, fromthreadargs

include("utils.jl")
include("threading.jl")
include("BlockRLE.jl")
include("DenseBlockLinearSystem.jl")
include("SparseBlockLinearSystem.jl")
include("kernels.jl")
include("schur_structure.jl")
include("schur_hessian.jl")
include("schur_reduction.jl")
include("linalg.jl")
include("schur_jacobian.jl")
include("schur_description.jl")
include("qr_reduction.jl")
include("qr_reduced.jl")
include("diagonal_blocks.jl")
include("schur_assembly.jl")
include("linear_systems.jl")
include("schur_linear_systems.jl")
include("qr_linear_systems.jl")
include("linear_system_conformance.jl")
include("docstrings.jl")

end # module BlockLinearSystems
