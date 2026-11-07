module BlockMatrices

using Static, StaticArrays, HybridArrays, SparseArrays, LinearAlgebra

export BlockSparseMatrix, BlockDenseMatrix
export block, validblock, zero!, copy_structure, uniformscaling!
export symmetrifyfull, symmetrifyfull!, symmetrifysparse, makesparseindices
export cacheindices, clearindicescache, updatelastrowsym!
export block_sparse_nnz, computestartindices, find_diag_inds, inplacecumsum!

include("utils.jl")
include("BlockDenseMatrix.jl")
include("BlockSparseMatrix.jl")
include("docstrings.jl")

end # module BlockMatrices
