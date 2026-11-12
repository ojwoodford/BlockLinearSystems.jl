module SecondOrderOptimizationExt

using SecondOrderOptimization
import SecondOrderOptimization: AbstractLinearSystem, solve!, dampen!, copy_structure, sharex, getA, getb, getx, zero!, fast_bAb, SolveStatus, Success, NotPositiveDefinite, Indefinite, makelinearsystem, max_diag, updatescaling!, AnyMarquardtData, is_indefinite
import SecondOrderOptimization.LDLFactorizations: ldl_analyze, ldl_factorize!, factorized
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
    SparseSchurLinearSystem{T, Ti, TL} <: AbstractLinearSystem

Linear system solving bipartite normal equations via sparse Schur reduction, CHOLMOD sparse Cholesky, and LDL factorizations.
"""
mutable struct SparseSchurLinearSystem{T, Ti, TL} <: AbstractLinearSystem
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
    S_factor_ldl::TL
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
    nchunks::Int=0,
    covis=nothing
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

    # 1. Build co-visibility graph and sparsity structures. Retained-retained couplings stored in B
    # that do not go through an eliminated block need slots in S too.
    # A precomputed `covis` (the result of an earlier build_schur_covisibility call with the same
    # extra blocks) can be passed in to avoid rebuilding it.
    if covis === nothing
        covis = build_schur_covisibility(H.structure; extra_blocks=BlockLinearSystems.retained_block_pairs(H.B))
    end
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

    # 3. Pre-analyze CHOLMOD symbolic factor and LDLFactorizations symbolic factor
    S_factor = SparseArrays.CHOLMOD.analyze(SparseArrays.CHOLMOD.Sparse(Symmetric(S, :U)))
    S_factor_ldl = ldl_analyze(Symmetric(S, :U))

    g_S = zeros(T, nc)

    K = nchunks > 0 ? nchunks : (isempty(H.structure.chunks) ? 1 : length(H.structure.chunks[1]))
    ws = SchurReductionWorkspace{T}(H.structure, K; is_dense=false, nnz_S=nnz(S))

    return SparseSchurLinearSystem{T, Ti, typeof(S_factor_ldl)}(
        H, b, x, b_c, b_p, x_c, x_p, g_S, S, S_factor, S_factor_ldl,
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

function SecondOrderOptimization.max_diag(H::SchurBlockMatrix{T}) where T
    m = zero(T)
    if H.B isa Matrix
        for i in 1:H.structure.nc
            m = max(m, abs(H.B[i, i]))
        end
    elseif H.B isa SparseBlockLinearSystem
        for b in 1:H.structure.ncblocks
            ptr = H.B.indices[b, b]
            if ptr > 0
                db = Int(H.structure.c_dims[b])
                for k in 1:db
                    m = max(m, abs(H.B.data[ptr + (k - 1) * (db + 1)]))
                end
            end
        end
    end
    for l in 1:H.structure.npblocks
        po = H.structure.ppoffsets[l]
        dp = Int(H.structure.p_dims[l])
        for k in 1:dp
            m = max(m, abs(H.C_data[po + (k - 1) * (dp + 1)]))
        end
    end
    return m
end

SecondOrderOptimization.max_diag(ls::Union{DenseSchurLinearSystem, SparseSchurLinearSystem}) = SecondOrderOptimization.max_diag(ls.H)

function SecondOrderOptimization.updatescaling!(lmd::AnyMarquardtData, ls::DenseSchurLinearSystem{T}) where T
    if lmd.lambda == 0
        lmd.lambda = 1e-6
    end
    B = ls.H.B
    C_data = ls.H.C_data
    struct_info = ls.H.structure
    nc = struct_info.nc
    m = zero(T)

    for i in 1:nc
        @inbounds d = lmd.scaling[i] = max(lmd.scaling[i], abs(B[i, i]))
        m = max(m, d)
    end
    for l in 1:struct_info.npblocks
        po = struct_info.ppoffsets[l]
        p_off = struct_info.boffsets_p[l]
        dp = Int(struct_info.p_dims[l])
        for i in 1:dp
            c_ii = C_data[po + (i - 1)*(dp + 1)]
            @inbounds d = lmd.scaling[nc + p_off + i - 1] = max(lmd.scaling[nc + p_off + i - 1], abs(c_ii))
            m = max(m, d)
        end
    end
    lmd.scaling .= max.(lmd.scaling, max(m * 1e-8, 1e-12))
end

function SecondOrderOptimization.updatescaling!(lmd::AnyMarquardtData, ls::SparseSchurLinearSystem{T}) where T
    if lmd.lambda == 0
        lmd.lambda = 1e-6
    end
    B_sys = ls.H.B
    C_data = ls.H.C_data
    struct_info = ls.H.structure
    nc = struct_info.nc
    m = zero(T)

    for b in 1:struct_info.ncblocks
        ptr = B_sys.indices[b, b]
        db = Int(struct_info.c_dims[b])
        bo = struct_info.boffsets_c[b]
        if ptr > 0
            for i in 1:db
                b_ii = B_sys.data[ptr + (i - 1)*(db + 1)]
                @inbounds d = lmd.scaling[bo + i - 1] = max(lmd.scaling[bo + i - 1], abs(b_ii))
                m = max(m, d)
            end
        end
    end
    for l in 1:struct_info.npblocks
        po = struct_info.ppoffsets[l]
        p_off = struct_info.boffsets_p[l]
        dp = Int(struct_info.p_dims[l])
        for i in 1:dp
            c_ii = C_data[po + (i - 1)*(dp + 1)]
            @inbounds d = lmd.scaling[nc + p_off + i - 1] = max(lmd.scaling[nc + p_off + i - 1], abs(c_ii))
            m = max(m, d)
        end
    end
    lmd.scaling .= max.(lmd.scaling, max(m * 1e-8, 1e-12))
end

function SecondOrderOptimization.zero!(ls::DenseSchurLinearSystem)
    fill!(ls.b, zero(eltype(ls.b)))
    BlockLinearSystems.zero!(ls.H)
    return ls
end

function SecondOrderOptimization.zero!(ls::SparseSchurLinearSystem)
    fill!(ls.b, zero(eltype(ls.b)))
    BlockLinearSystems.zero!(ls.H)
    return ls
end

# Build the reduced system S and g_S from scratch. Called at the start of each solve, and again if an
# in-place factorization has overwritten S and a fallback factorization is needed.
function build_reduced!(ls::DenseSchurLinearSystem)
    reset_schur_dense!(ls.S, ls.H.B, ls.scaling_c, ls.lambda, ls.workspace)
    return reduce_schur_dense!(
        ls.S, ls.g_S, ls.b_c, ls.b_p, ls.scaling_p, ls.lambda,
        ls.H.structure, ls.H.C_data, ls.H.E_data, ls.workspace
    )
end

function solve_bunchkaufman!(ls::DenseSchurLinearSystem)
    bk = bunchkaufman!(Symmetric(ls.S, :U); check=false)
    issuccess(bk) || return NotPositiveDefinite
    ldiv!(ls.x_c, bk, ls.g_S)
    is_indefinite(bk) && return Indefinite
    return Success
end

function SecondOrderOptimization.solve!(ls::DenseSchurLinearSystem, solver_opt::Any=static(:auto))
    build_reduced!(ls) || return NotPositiveDefinite

    if solver_opt isa StaticSymbol{:ldl}
        status = solve_bunchkaufman!(ls)
    else
        # Cholesky first, whether or not there is damping. It works in place on S (which is rebuilt on
        # every solve anyway); only if it fails do we rebuild S and fall back to Bunch-Kaufman.
        cholfac = cholesky!(Symmetric(ls.S, :U); check=false)
        if issuccess(cholfac)
            ldiv!(ls.x_c, cholfac, ls.g_S)
            status = Success
        else
            build_reduced!(ls) || return NotPositiveDefinite
            status = solve_bunchkaufman!(ls)
        end
    end
    status == Success || return status

    backsubstitute!(ls.x_p, ls.x_c, ls.H.structure, ls.H.E_data, ls.workspace.C_inv_data, ls.workspace.z_data)
    return Success
end
SecondOrderOptimization.solve!(ls::DenseSchurLinearSystem, s::Symbol) = SecondOrderOptimization.solve!(ls, static(s))

# `ldiv!(x, ::CHOLMOD.Factor, b)` only exists from Julia 1.12; earlier versions need the allocating `\\`.
@static if VERSION >= v"1.12"
    cholmod_ldiv!(x, F::SparseArrays.CHOLMOD.Factor, b) = ldiv!(x, F, b)
else
    cholmod_ldiv!(x, F::SparseArrays.CHOLMOD.Factor, b) = copyto!(x, F \ b)
end

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

    if solver_opt isa StaticSymbol{:ldl}
        F = ldl_factorize!(Symmetric(ls.S, :U), ls.S_factor_ldl)
        if !factorized(F)
            return NotPositiveDefinite
        end
        ldiv!(ls.x_c, F, ls.g_S)
        if is_indefinite(F)
            return Indefinite
        end
    else
        LinearAlgebra.cholesky!(ls.S_factor, Symmetric(ls.S, :U); check=false)
        if !LinearAlgebra.issuccess(ls.S_factor)
            F = ldl_factorize!(Symmetric(ls.S, :U), ls.S_factor_ldl)
            if !factorized(F)
                return NotPositiveDefinite
            end
            ldiv!(ls.x_c, F, ls.g_S)
            if is_indefinite(F)
                return Indefinite
            end
        else
            cholmod_ldiv!(ls.x_c, ls.S_factor, ls.g_S)
        end
    end

    backsubstitute!(ls.x_p, ls.x_c, ls.H.structure, ls.H.E_data, ls.workspace.C_inv_data, ls.workspace.z_data)
    return Success
end
SecondOrderOptimization.solve!(ls::SparseSchurLinearSystem, s::Symbol) = SecondOrderOptimization.solve!(ls, static(s))

# `quadratic_model_reduction` is intentionally not specialised: SecondOrderOptimization's generic method
# (0.5 * fast_bAb(getA(ls), dx) + fast_dot(getb(ls), dx)) is the undamped model change, and works with
# the `fast_bAb` for `SchurBlockMatrix` defined above. (An earlier override added the damping term with
# the wrong sign, and used a damping value the iterators have already reset to zero when a step is scored.)

function SecondOrderOptimization.copy_structure(ls::DenseSchurLinearSystem{T}) where T
    # New storage for the Hessian and gradient (the original is kept for retries); x is shared
    nc = ls.H.structure.nc
    b = copy(ls.b)
    x = ls.x
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
        BlockLinearSystems.copy_structure(ls.H), b, x,
        view(b, 1:nc), view(b, nc+1:length(b)),
        view(x, 1:nc), view(x, nc+1:length(x)),
        similar(ls.g_S), similar(ls.S), ls.lambda, scaling,
        view(scaling, 1:nc), view(scaling, nc+1:length(scaling)),
        ws_trial
    )
end


function SecondOrderOptimization.copy_structure(ls::SparseSchurLinearSystem{T, Ti, TL}) where {T, Ti, TL}
    # New storage for the Hessian and gradient (the original is kept for retries); x is shared
    nc = ls.H.structure.nc
    b = copy(ls.b)
    x = ls.x
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
    return SparseSchurLinearSystem{T, Ti, TL}(
        BlockLinearSystems.copy_structure(ls.H), b, x,
        view(b, 1:nc), view(b, nc+1:length(b)),
        view(x, 1:nc), view(x, nc+1:length(x)),
        similar(ls.g_S), copy(ls.S), ls.S_factor, ls.S_factor_ldl,
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

function SecondOrderOptimization.sharex(ls::SparseSchurLinearSystem{T, Ti, TL}, from) where {T, Ti, TL}
    ls.x === from.x && return ls
    nc = ls.H.structure.nc
    x = from.x
    return SparseSchurLinearSystem{T, Ti, TL}(
        ls.H, ls.b, x,
        ls.b_c, ls.b_p,
        view(x, 1:nc), view(x, nc+1:length(x)),
        ls.g_S, ls.S, ls.S_factor, ls.S_factor_ldl,
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
