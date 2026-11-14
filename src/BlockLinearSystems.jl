module BlockLinearSystems

using Static, StaticArrays, HybridArrays, SparseArrays, LinearAlgebra, Polyester

export SparseBlockLinearSystem, DenseBlockLinearSystem, TransposedPattern, BlockOffsets
export BlockRLE, nruns, runlength, runblocksize, matrixdimension, runblockrange, runmatrixrange, eachrun, sort_blocks_hierarchical
export block, validblock, zero!, copy_structure, uniformscaling!
export symmetrifyfull, symmetrifyfull!, symmetrifysparse, makesparseindices
export sparse_with_map, copyto!
export block_sparse_nnz
export blocksizes, nblocks, blockdata, eachstoredblock, blockrange
export SchurStructure, SchurBlockStructure, SchurBlockMatrix, SchurReductionWorkspace, trial_workspace
export expand_block_perm, sort_schur_blocks, build_schur_pattern, build_r_to_s_map, retained_block_pairs
export inv_sym_block, reduce_schur_dense!, reduce_schur_sparse!
export reset_schur_dense!, reset_schur_sparse!, backsubstitute!
export fast_bAb
export SchurJacobianStructure, SchurBlockJacobian, schur_jacobian_bytes, coupling_block, eliminated_block, residual_block, coupling_entry
export QRReductionWorkspace, reduce_qr!
export reduced_nrows, free_block_pairs, reduced_mul!, reduced_tmul!, reduced_residual!, reduced_jacobian!
export reduced_normal_dense!, reduced_normal_sparse!, reduced_diagonal_blocks!, jtr!, jtj_diag!

include("utils.jl")
include("BlockRLE.jl")
include("DenseBlockLinearSystem.jl")
include("SparseBlockLinearSystem.jl")
include("schur.jl")
include("schur_jacobian.jl")
include("qr_reduction.jl")
include("qr_reduced.jl")
include("docstrings.jl")

end # module BlockLinearSystems
