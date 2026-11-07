module BlockLinearSystems

using Static, StaticArrays, HybridArrays, SparseArrays, LinearAlgebra

export SparseBlockLinearSystem, DenseBlockLinearSystem
export block, validblock, zero!, copy_structure, uniformscaling!
export symmetrifyfull, symmetrifyfull!, symmetrifysparse, makesparseindices
export block_sparse_nnz, computestartindices, find_diag_inds, inplacecumsum!

include("utils.jl")
include("DenseBlockLinearSystem.jl")
include("SparseBlockLinearSystem.jl")
include("docstrings.jl")

end # module BlockLinearSystems
