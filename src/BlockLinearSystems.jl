module BlockLinearSystems

using Static, StaticArrays, HybridArrays, SparseArrays, LinearAlgebra

export SparseBlockLinearSystem, DenseBlockLinearSystem, TransposedPattern, BlockOffsets
export BlockRLE, nruns, runlength, runblocksize, matrixdimension, runblockrange, runmatrixrange, eachrun, sort_blocks_hierarchical
export block, validblock, zero!, copy_structure, uniformscaling!
export symmetrifyfull, symmetrifyfull!, symmetrifysparse, makesparseindices
export sparse_with_map, copyto!
export block_sparse_nnz
export blocksizes, nblocks, blockdata, eachstoredblock
export SchurStructure, SchurBlockStructure, SchurBlockMatrix, SchurReductionWorkspace
export expand_block_perm, sort_schur_blocks, build_schur_covisibility, build_b_to_s_map, retained_block_pairs
export inv_sym_block, reduce_schur_dense!, reduce_schur_sparse!
export reset_schur_dense!, reset_schur_sparse!, merge_dense_S!, merge_sparse_S!, backsubstitute!
export fast_bAb

include("utils.jl")
include("BlockRLE.jl")
include("DenseBlockLinearSystem.jl")
include("SparseBlockLinearSystem.jl")
include("schur.jl")
include("docstrings.jl")

end # module BlockLinearSystems
