module SecondOrderOptimizationExt

using SecondOrderOptimization
import SecondOrderOptimization: AbstractLinearSystem, solve!, dampen!, copy_structure, sharex, getA, getb, getx, zero!, fast_bAb, SolveStatus, Success, NotPositiveDefinite, Indefinite, makelinearsystem
using BlockLinearSystems
import BlockLinearSystems: SchurBlockMatrix, SchurStructure, SchurBlockStructure, SchurReductionWorkspace, reset_schur_dense!, reset_schur_sparse!, reduce_schur_dense!, reduce_schur_sparse!, backsubstitute!, build_schur_covisibility, build_b_to_s_map
using LinearAlgebra, SparseArrays, Static

"""
    DenseSchurLinearSystem{T} <: AbstractLinearSystem

Linear system solving bipartite normal equations via dense Schur reduction.
"""
mutable struct DenseSchurLinearSystem{T} <: AbstractLinearSystem
    H::SchurBlockMatrix{T, Matrix{T}}
    b::Vector{T}
    x::Vector{T}
    b_c::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    b_p::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    x_c::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    x_p::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    g_S::Vector{T}
    S::Matrix{T}
    lambda::T
    scaling::Vector{T}
    scaling_c::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    scaling_p::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    workspace::SchurReductionWorkspace{T}
end

function DenseSchurLinearSystem(
    H::SchurBlockMatrix{T, Matrix{T}},
    b::Vector{T}, x::Vector{T},
    scaling::Vector{T}=ones(T, length(b));
    nchunks::Int=0
) where T
    nc = H.structure.nc
    tot_dim = length(b)
    @assert tot_dim == nc + sum(Int, H.structure.p_dims; init=0) "Dimensions mismatch"

    b_c = view(b, 1:nc)
    b_p = view(b, nc+1:tot_dim)
    x_c = view(x, 1:nc)
    x_p = view(x, nc+1:tot_dim)
    scaling_c = view(scaling, 1:nc)
    scaling_p = view(scaling, nc+1:tot_dim)

    g_S = zeros(T, nc)
    S = zeros(T, nc, nc)

    K = nchunks > 0 ? nchunks : (isempty(H.structure.chunks) ? 1 : length(H.structure.chunks[1]))
    ws = SchurReductionWorkspace{T}(H.structure, K; is_dense=true)

    return DenseSchurLinearSystem{T}(
        H, b, x, b_c, b_p, x_c, x_p, g_S, S, zero(T), scaling, scaling_c, scaling_p, ws
    )
end

"""
    SparseSchurLinearSystem{T, Ti} <: AbstractLinearSystem

Linear system solving bipartite normal equations via sparse Schur reduction and CHOLMOD sparse Cholesky.
"""
mutable struct SparseSchurLinearSystem{T, Ti} <: AbstractLinearSystem
    H::SchurBlockMatrix{T, SparseBlockLinearSystem{T}}
    b::Vector{T}
    x::Vector{T}
    b_c::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    b_p::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    x_c::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    x_p::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    g_S::Vector{T}
    S::SparseMatrixCSC{T, Ti}
    S_factor::SparseArrays.CHOLMOD.Factor{T}
    b_to_s_map::Vector{Int}
    diag_inds::Vector{Int}
    block_rowval::Vector{Ti}
    block_colptr::Vector{Ti}
    block_row_offsets::Vector{Int}
    col_heights::Vector{Int}
    lambda::T
    scaling::Vector{T}
    scaling_c::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    scaling_p::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    workspace::SchurReductionWorkspace{T}
end

function SparseSchurLinearSystem(
    H::SchurBlockMatrix{T, SparseBlockLinearSystem{T}},
    b::Vector{T}, x::Vector{T},
    scaling::Vector{T}=ones(T, length(b));
    nchunks::Int=0
) where T
    nc = H.structure.nc
    tot_dim = length(b)
    @assert tot_dim == nc + sum(Int, H.structure.p_dims; init=0) "Dimensions mismatch"

    b_c = view(b, 1:nc)
    b_p = view(b, nc+1:tot_dim)
    x_c = view(x, 1:nc)
    x_p = view(x, nc+1:tot_dim)
    scaling_c = view(scaling, 1:nc)
    scaling_p = view(scaling, nc+1:tot_dim)

    # 1. Build co-visibility graph and sparsity structures
    covis = build_schur_covisibility(H.structure)
    S = covis.S_sparse
    Ti = eltype(covis.block_rowval)

    # 2. Map B to S
    if H.B isa SparseBlockLinearSystem
        b_to_s_map = build_b_to_s_map(H.B, S, H.structure.boffsets_c)
    elseif H.B isa SparseMatrixCSC
        b_to_s_map = build_b_to_s_map(H.B, S)
    else
        B_sparse, _ = sparse_with_map(H.B; symmetric=true)
        b_to_s_map = build_b_to_s_map(B_sparse, S)
    end

    # 3. Pre-analyze CHOLMOD symbolic factor
    S_factor = SparseArrays.CHOLMOD.analyze(SparseArrays.CHOLMOD.Sparse(Symmetric(S, :U)))

    g_S = zeros(T, nc)

    K = nchunks > 0 ? nchunks : (isempty(H.structure.chunks) ? 1 : length(H.structure.chunks[1]))
    ws = SchurReductionWorkspace{T}(H.structure, K; is_dense=false, nnz_S=nnz(S))

    return SparseSchurLinearSystem{T, Ti}(
        H, b, x, b_c, b_p, x_c, x_p, g_S, S, S_factor,
        b_to_s_map, covis.diag_inds,
        covis.block_rowval, covis.block_colptr, covis.block_row_offsets, covis.col_heights,
        zero(T), scaling, scaling_c, scaling_p, ws
    )
end

# -----------------------------------------------------------------------------
# Interface Contracts
# -----------------------------------------------------------------------------
SecondOrderOptimization.getA(ls::DenseSchurLinearSystem) = ls.H
SecondOrderOptimization.getb(ls::DenseSchurLinearSystem) = ls.b
SecondOrderOptimization.getx(ls::DenseSchurLinearSystem) = ls.x

SecondOrderOptimization.getA(ls::SparseSchurLinearSystem) = ls.H
SecondOrderOptimization.getb(ls::SparseSchurLinearSystem) = ls.b
SecondOrderOptimization.getx(ls::SparseSchurLinearSystem) = ls.x

SecondOrderOptimization.fast_bAb(H::SchurBlockMatrix, dx::AbstractVector) = BlockLinearSystems.fast_bAb(H, dx)

function SecondOrderOptimization.dampen!(ls::Union{DenseSchurLinearSystem{T}, SparseSchurLinearSystem{T}}, lambda::Real) where T
    ls.lambda = convert(T, lambda)
    return ls
end

function SecondOrderOptimization.dampen!(ls::Union{DenseSchurLinearSystem{T}, SparseSchurLinearSystem{T}}, lambda::Real, scaling) where T
    ls.lambda = convert(T, lambda)
    if scaling !== nothing
        copyto!(ls.scaling, scaling)
    end
    return ls
end

function SecondOrderOptimization.zero!(ls::DenseSchurLinearSystem)
    fill!(ls.b, zero(eltype(ls.b)))
    fill!(ls.x, zero(eltype(ls.x)))
    fill!(ls.H.B, zero(eltype(ls.b)))
    fill!(ls.H.C_data, zero(eltype(ls.b)))
    fill!(ls.H.E_data, zero(eltype(ls.b)))
    return ls
end

function SecondOrderOptimization.zero!(ls::SparseSchurLinearSystem)
    fill!(ls.b, zero(eltype(ls.b)))
    fill!(ls.x, zero(eltype(ls.x)))
    zero!(ls.H.B)
    fill!(ls.H.C_data, zero(eltype(ls.b)))
    fill!(ls.H.E_data, zero(eltype(ls.b)))
    return ls
end

function SecondOrderOptimization.solve!(ls::DenseSchurLinearSystem, solver_opt::Any=static(:auto))
    reset_schur_dense!(ls.S, ls.H.B, ls.scaling_c, ls.lambda, ls.workspace)
    success = reduce_schur_dense!(
        ls.S, ls.g_S, ls.b_c, ls.b_p, ls.scaling_p, ls.lambda,
        ls.H.structure, ls.H.C_data, ls.H.E_data, ls.workspace
    )
    if !success
        return NotPositiveDefinite
    end

    cholfac = cholesky(Symmetric(ls.S, :U); check=false)
    if !issuccess(cholfac)
        return NotPositiveDefinite
    end

    ldiv!(ls.x_c, cholfac, ls.g_S)
    backsubstitute!(ls.x_p, ls.x_c, ls.H.structure, ls.H.E_data, ls.workspace.C_inv_data, ls.workspace.z_data)
    return Success
end
SecondOrderOptimization.solve!(ls::DenseSchurLinearSystem, s::Symbol) = SecondOrderOptimization.solve!(ls, static(s))

function SecondOrderOptimization.solve!(ls::SparseSchurLinearSystem, solver_opt::Any=static(:auto))
    B_nzval = if ls.H.B isa SparseBlockLinearSystem
        ls.H.B.data
    elseif ls.H.B isa SparseMatrixCSC
        ls.H.B.nzval
    else
        error("Unsupported B type for SparseSchurLinearSystem: $(typeof(ls.H.B))")
    end
    reset_schur_sparse!(
        ls.S.nzval, B_nzval, ls.b_to_s_map, ls.diag_inds, ls.scaling_c, ls.lambda, ls.workspace
    )
    success = reduce_schur_sparse!(
        ls.S, ls.g_S, ls.b_c, ls.b_p, ls.scaling_p, ls.lambda,
        ls.H.structure, ls.H.C_data, ls.H.E_data,
        ls.block_colptr, ls.block_rowval, ls.block_row_offsets, ls.col_heights,
        ls.workspace
    )
    if !success
        return NotPositiveDefinite
    end

    LinearAlgebra.cholesky!(ls.S_factor, Symmetric(ls.S, :U))
    if !LinearAlgebra.issuccess(ls.S_factor)
        return NotPositiveDefinite
    end

    ls.x_c .= ls.S_factor \ ls.g_S
    backsubstitute!(ls.x_p, ls.x_c, ls.H.structure, ls.H.E_data, ls.workspace.C_inv_data, ls.workspace.z_data)
    return Success
end
SecondOrderOptimization.solve!(ls::SparseSchurLinearSystem, s::Symbol) = SecondOrderOptimization.solve!(ls, static(s))

function SecondOrderOptimization.copy_structure(ls::DenseSchurLinearSystem{T}) where T
    nc = ls.H.structure.nc
    b = copy(ls.b)
    x = copy(ls.x)
    scaling = copy(ls.scaling)
    ws_trial = SchurReductionWorkspace{T}(
        ls.workspace.W_scratch,
        similar(ls.workspace.C_inv_data),
        similar(ls.workspace.z_data),
        ls.workspace.thread_g_S,
        ls.workspace.thread_dense_S,
        ls.workspace.thread_nzvals,
        Base.Threads.Atomic{Int}(0)
    )
    return DenseSchurLinearSystem{T}(
        ls.H, b, x,
        view(b, 1:nc), view(b, nc+1:length(b)),
        view(x, 1:nc), view(x, nc+1:length(x)),
        similar(ls.g_S), similar(ls.S), ls.lambda, scaling,
        view(scaling, 1:nc), view(scaling, nc+1:length(scaling)),
        ws_trial
    )
end


function SecondOrderOptimization.copy_structure(ls::SparseSchurLinearSystem{T, Ti}) where {T, Ti}
    nc = ls.H.structure.nc
    b = copy(ls.b)
    x = copy(ls.x)
    scaling = copy(ls.scaling)
    ws_trial = SchurReductionWorkspace{T}(
        ls.workspace.W_scratch,
        similar(ls.workspace.C_inv_data),
        similar(ls.workspace.z_data),
        ls.workspace.thread_g_S,
        ls.workspace.thread_dense_S,
        ls.workspace.thread_nzvals,
        Base.Threads.Atomic{Int}(0)
    )
    return SparseSchurLinearSystem{T, Ti}(
        ls.H, b, x,
        view(b, 1:nc), view(b, nc+1:length(b)),
        view(x, 1:nc), view(x, nc+1:length(x)),
        similar(ls.g_S), copy(ls.S), ls.S_factor,
        ls.b_to_s_map, ls.diag_inds,
        ls.block_rowval, ls.block_colptr, ls.block_row_offsets, ls.col_heights,
        ls.lambda, scaling,
        view(scaling, 1:nc), view(scaling, nc+1:length(scaling)),
        ws_trial
    )
end

BlockLinearSystems.copy_structure(ls::DenseSchurLinearSystem) = SecondOrderOptimization.copy_structure(ls)
BlockLinearSystems.copy_structure(ls::SparseSchurLinearSystem) = SecondOrderOptimization.copy_structure(ls)

function SecondOrderOptimization.sharex(ls::DenseSchurLinearSystem{T}, from) where T
    ls.x === from.x && return ls
    nc = ls.H.structure.nc
    x = from.x
    return DenseSchurLinearSystem{T}(
        ls.H, ls.b, x,
        ls.b_c, ls.b_p,
        view(x, 1:nc), view(x, nc+1:length(x)),
        ls.g_S, ls.S, ls.lambda, ls.scaling,
        ls.scaling_c, ls.scaling_p,
        ls.workspace
    )
end

function SecondOrderOptimization.sharex(ls::SparseSchurLinearSystem{T, Ti}, from) where {T, Ti}
    ls.x === from.x && return ls
    nc = ls.H.structure.nc
    x = from.x
    return SparseSchurLinearSystem{T, Ti}(
        ls.H, ls.b, x,
        ls.b_c, ls.b_p,
        view(x, 1:nc), view(x, nc+1:length(x)),
        ls.g_S, ls.S, ls.S_factor,
        ls.b_to_s_map, ls.diag_inds,
        ls.block_rowval, ls.block_colptr, ls.block_row_offsets, ls.col_heights,
        ls.lambda, ls.scaling,
        ls.scaling_c, ls.scaling_p,
        ls.workspace
    )
end

function SecondOrderOptimization.makelinearsystem(x0, H::SchurBlockMatrix{T}, linsolver) where T
    nc = H.structure.nc
    np = sum(Int, H.structure.p_dims; init=0)
    tot = nc + np
    b = zeros(T, tot)
    x = zeros(T, tot)
    sym = SecondOrderOptimization.linsolver_symbol(linsolver)
    if sym === :sparse_schur || H.B isa SparseBlockLinearSystem || H.B isa AbstractSparseMatrix
        return SparseSchurLinearSystem(H, b, x)
    else
        return DenseSchurLinearSystem(H, b, x)
    end
end

end # module SecondOrderOptimizationExt
