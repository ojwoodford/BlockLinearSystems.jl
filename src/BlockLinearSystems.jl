module BlockLinearSystems

using Static, StaticArrays, HybridArrays, SparseArrays, LinearAlgebra

export SparseBlockLinearSystem, DenseBlockLinearSystem, TransposedPattern, BlockOffsets
export BlockRLE, nruns, runlength, runblocksize, matrixdimension, runblockrange, runmatrixrange, eachrun, sort_blocks_hierarchical
export block, validblock, zero!, copy_structure, uniformscaling!
export symmetrifyfull, symmetrifyfull!, symmetrifysparse, makesparseindices
export sparse_with_map, copyto!
export block_sparse_nnz
export blocksizes, nblocks, blockdata, eachstoredblock

include("utils.jl")
include("BlockRLE.jl")
include("DenseBlockLinearSystem.jl")
include("SparseBlockLinearSystem.jl")
include("schur.jl")
include("docstrings.jl")

end # module BlockLinearSystems
