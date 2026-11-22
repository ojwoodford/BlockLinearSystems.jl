# `linsolver` names the solver for the reduced system S. Only these are implemented for it; any other
# is rejected when the system is built, rather than silently treated as `:auto`.
const REDUCED_SOLVERS = (:auto, :cholesky, :ldl)
const ReducedSolver = Union{StaticSymbol{:auto}, StaticSymbol{:cholesky}, StaticSymbol{:ldl}}
function check_reduced_solver(linsolver)
    sym = linsolver_symbol(linsolver)
    sym in REDUCED_SOLVERS || throw(ArgumentError("linsolver = :$sym is not implemented for the reduced system of a SchurBlockHessian; use :auto or :cholesky (Cholesky, falling back to Bunch-Kaufman) or :ldl (Bunch-Kaufman)."))
    return nothing
end

# The element types a SchurBlockHessian system is built in (single-precision.md): the reduction, the reduced system and
# its factorization are in T; b and x are Float64.
const REDUCED_ELTYPES = (Float64, Float32)
# Whether SuiteSparse's CHOLMOD, as this Julia's SparseArrays binds it, factorizes in T: SparseArrays 1.10 converts a
# Float32 matrix to Float64, SparseArrays 1.13 factorizes it in Float32 (checked by building and solving)
reduced_cholmod(::Type{T}) where T = T === Float64 || T <: SparseArrays.CHOLMOD.VTypes
# Decision 10: the dense reduced solves (LAPACK Cholesky and Bunch-Kaufman, through DampedLinearSolve) offer Float32; of
# the sparse ones, LDLᵀ (LDLFactorizations) does, and :auto and :cholesky (CHOLMOD, falling back to LDLᵀ) only where
# CHOLMOD does. Anything else is an ArgumentError when the system is built, naming the solvers that do.
function check_reduced_eltype(H::SchurBlockHessian{T}, linsolver) where T
    T in REDUCED_ELTYPES || throw(ArgumentError("A SchurBlockHessian{$T} has element type $T; a linear system is built in Float64 or Float32."))
    sym = linsolver_symbol(linsolver)
    (H.R isa SparseBlockLinearSystem && sym in (:auto, :cholesky) && !reduced_cholmod(T)) &&
        throw(ArgumentError("linsolver = :$sym does not offer $T for the sparse reduced system of a SchurBlockHessian (CHOLMOD is Float64 only in SparseArrays $(pkgversion(SparseArrays)), on Julia $VERSION). The solver that offers $T for it is :ldl."))
    return nothing
end

"""
    DenseSchurLinearSystem{T} <: AbstractLinearSystem

Linear system solving bipartite normal equations via dense Schur reduction. `T` (Float64 or Float32) is the element
type of the Hessian, its reduction and its factorization; `b`, `x`, the damping and the scaling are Float64.
"""
mutable struct DenseSchurLinearSystem{T, R} <: AbstractLinearSystem
    H::SchurBlockHessian{T, Matrix{T}}
    b::Vector{Float64}
    x::Vector{Float64}
    b_r::SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int}}, true}
    b_e::SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int}}, true}
    x_r::SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int}}, true}
    x_e::SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int}}, true}
    red::R                            # the reduced system S x_r = g_S: a dense DampedLinearSolve system in T, undamped
    g_S::Vector{Float64}              # getb(red)
    S::Matrix{T}                      # getA(red): the reduction writes its upper triangle
    lambda::Float64
    scaling::Vector{Float64}
    scaling_r::SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int}}, true}
    scaling_e::SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int}}, true}
    workspace::SchurReductionWorkspace{T}
end

function DenseSchurLinearSystem(
    H::SchurBlockHessian{T, Matrix{T}},
    b::Vector{Float64}, x::Vector{Float64},
    scaling::Vector{Float64}=ones(length(b));
    nchunks::Int=0
) where T
    nr = H.structure.nr
    tot_dim = length(b)
    @assert tot_dim == nr + sum(Int, H.structure.e_dims; init=0) "Dimensions mismatch"

    b_r = view(b, 1:nr)
    b_e = view(b, nr+1:tot_dim)
    x_r = view(x, 1:nr)
    x_e = view(x, nr+1:tot_dim)
    scaling_r = view(scaling, 1:nr)
    scaling_e = view(scaling, nr+1:tot_dim)

    # Built for :ldl, so that it holds the Bunch-Kaufman workspace; :cholesky needs none. From a prototype of
    # H's element type, so that the reduced system is in T.
    red = makehessiansystem(zeros(T, nr, nr), static(:ldl))

    K = nchunks > 0 ? nchunks : (isempty(H.structure.chunks) ? 1 : length(H.structure.chunks[1]))
    ws = SchurReductionWorkspace{T}(H.structure, K)

    return DenseSchurLinearSystem{T, typeof(red)}(
        H, b, x, b_r, b_e, x_r, x_e, red, getb(red), getA(red), 0.0, scaling, scaling_r, scaling_e, ws
    )
end

"""
    SparseSchurLinearSystem{T, Ti, TL, TF, TC} <: AbstractLinearSystem

Linear system solving bipartite normal equations via sparse Schur reduction, CHOLMOD sparse Cholesky, and LDL factorizations.
`T` is the element type of the Hessian, the reduced system and its factorization; `b`, `x`, the damping and the
scaling are Float64, and the reduced solve narrows `g_S` into `rhs` and solves into `sol`, both in T.
"""
mutable struct SparseSchurLinearSystem{T, Ti, TL, TF, TC} <: AbstractLinearSystem
    H::SchurBlockHessian{T, SparseBlockLinearSystem{T}}
    b::Vector{Float64}
    x::Vector{Float64}
    b_r::SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int}}, true}
    b_e::SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int}}, true}
    x_r::SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int}}, true}
    x_e::SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int}}, true}
    g_S::Vector{Float64}
    S::SparseMatrixCSC{T, Ti}
    S_factor::TF                      # CHOLMOD.Factor (its type parameters differ between Julia versions); nothing where CHOLMOD does not offer T
    S_factor_ldl::TL
    S_chol::TC                        # CHOLMOD.Sparse copy of S, refreshed in place before each factorization; nothing likewise
    rhs::Vector{T}                    # g_S narrowed, for the factorization
    sol::Vector{T}                    # the factorization's solution x_r
    r_to_s_map::Vector{Int}
    diag_inds::Vector{Int}
    block_rowval::Vector{Ti}
    block_colptr::Vector{Ti}
    block_row_offsets::Vector{Int}
    col_heights::Vector{Int}
    lambda::Float64
    scaling::Vector{Float64}
    scaling_r::SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int}}, true}
    scaling_e::SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int}}, true}
    workspace::SchurReductionWorkspace{T}
end

function SparseSchurLinearSystem(
    H::SchurBlockHessian{T, SparseBlockLinearSystem{T}},
    b::Vector{Float64}, x::Vector{Float64},
    scaling::Vector{Float64}=ones(length(b));
    nchunks::Int=0,
    pattern=nothing,
    cholmod::Bool=true
) where T
    nr = H.structure.nr
    tot_dim = length(b)
    @assert tot_dim == nr + sum(Int, H.structure.e_dims; init=0) "Dimensions mismatch"

    b_r = view(b, 1:nr)
    b_e = view(b, nr+1:tot_dim)
    x_r = view(x, 1:nr)
    x_e = view(x, nr+1:tot_dim)
    scaling_r = view(scaling, 1:nr)
    scaling_e = view(scaling, nr+1:tot_dim)

    # 1. Build the reduced-system pattern and sparsity structures. Retained-retained couplings stored in R
    # that do not go through an eliminated block need slots in S too.
    # A precomputed `pattern` (the result of an earlier build_schur_pattern call with the same
    # extra blocks, typically carried by `H` itself) can be passed in to avoid rebuilding it. It is
    # used as structure only: it may be shared with other systems, so S gets its own copy of
    # `S_sparse`, whose values this system writes.
    if pattern === nothing
        pattern = build_schur_pattern(H.structure, T; extra_blocks=retained_block_pairs(H.R))
        S = pattern.S_sparse
    else
        S = copy(pattern.S_sparse)
    end
    Ti = eltype(pattern.block_rowval)

    # 2. Map R to S
    r_to_s_map = build_r_to_s_map(H.R, pattern, H.structure.boffsets_r)

    # 3. Pre-analyze CHOLMOD symbolic factor and LDLFactorizations symbolic factor
    # A caller can add an analysed factor to the pattern (`merge(pattern, (S_factor = fac,))`);
    # `hasproperty` on a NamedTuple is resolved at compile time.
    # CHOLMOD only where it offers T (`cholmod`): a Float32 system without it is solved by LDLᵀ only
    S_chol = cholmod ? SparseArrays.CHOLMOD.Sparse(Symmetric(S, :U)) : nothing
    S_factor = !cholmod ? nothing : hasproperty(pattern, :S_factor) ? pattern.S_factor : SparseArrays.CHOLMOD.analyze(S_chol)
    S_factor_ldl = ldl_analyze(Symmetric(S, :U))

    g_S = zeros(nr)

    K = nchunks > 0 ? nchunks : (isempty(H.structure.chunks) ? 1 : length(H.structure.chunks[1]))
    ws = SchurReductionWorkspace{T}(H.structure, K)

    return SparseSchurLinearSystem{T, Ti, typeof(S_factor_ldl), typeof(S_factor), typeof(S_chol)}(
        H, b, x, b_r, b_e, x_r, x_e, g_S, S, S_factor, S_factor_ldl, S_chol, zeros(T, nr), zeros(T, nr),
        r_to_s_map, pattern.diag_inds,
        pattern.block_rowval, pattern.block_colptr, pattern.block_row_offsets, pattern.col_heights,
        0.0, scaling, scaling_r, scaling_e, ws
    )
end

# -----------------------------------------------------------------------------
# Interface Contracts
# -----------------------------------------------------------------------------
DampedLinearSolve.getA(ls::DenseSchurLinearSystem) = ls.H
DampedLinearSolve.getb(ls::DenseSchurLinearSystem) = ls.b
DampedLinearSolve.getx(ls::DenseSchurLinearSystem) = ls.x

DampedLinearSolve.getA(ls::SparseSchurLinearSystem) = ls.H
DampedLinearSolve.getb(ls::SparseSchurLinearSystem) = ls.b
DampedLinearSolve.getx(ls::SparseSchurLinearSystem) = ls.x

# As for every linear system: `dampen!(ls, λ)` means λI and `dampen!(ls, λ, s)` means λ·Diagonal(s),
# each replacing the damping set before. The reduction always applies λ·scaling, so uniform damping
# is stored as scaling = 1.
function DampedLinearSolve.dampen!(ls::Union{DenseSchurLinearSystem, SparseSchurLinearSystem}, lambda::Real)
    ls.lambda = Float64(lambda)
    fill!(ls.scaling, 1.0)
    return ls
end

function DampedLinearSolve.dampen!(ls::Union{DenseSchurLinearSystem, SparseSchurLinearSystem}, lambda::Real, scaling)
    scaling === nothing && return DampedLinearSolve.dampen!(ls, lambda)
    ls.lambda = Float64(lambda)
    copyto!(ls.scaling, scaling)
    return ls
end

# A Float64, whatever the element type of H
function DampedLinearSolve.max_diag(H::SchurBlockHessian{T}) where T
    m = 0.0
    if H.R isa Matrix
        for i in 1:H.structure.nr
            m = max(m, Float64(abs(H.R[i, i])))
        end
    elseif H.R isa SparseBlockLinearSystem
        for b in 1:H.structure.nrblocks
            ptr = H.R.indices[b, b]
            if ptr > 0
                db = Int(H.structure.r_dims[b])
                for k in 1:db
                    m = max(m, Float64(abs(H.R.data[ptr + (k - 1) * (db + 1)])))
                end
            end
        end
    end
    for e in 1:H.structure.neblocks
        eo = H.structure.eoffsets[e]
        de = Int(H.structure.e_dims[e])
        for k in 1:de
            m = max(m, Float64(abs(H.E_data[eo + (k - 1) * (de + 1)])))
        end
    end
    return m
end

DampedLinearSolve.max_diag(ls::Union{DenseSchurLinearSystem, SparseSchurLinearSystem}) = DampedLinearSolve.max_diag(ls.H)
DampedLinearSolve.isfinitesystem(ls::Union{DenseSchurLinearSystem, SparseSchurLinearSystem}) =
    all(isfinite, ls.H.R isa Matrix ? ls.H.R : ls.H.R.data) && all(isfinite, ls.H.E_data) && all(isfinite, ls.H.C_data)

# The undamped diagonal of H, for the Marquardt scaling. Only the retained diagonal is read differently for
# dense and sparse R; a retained block whose diagonal block R does not store has a zero diagonal.
function DampedLinearSolve.getdiag!(d::AbstractVector, ls::Union{DenseSchurLinearSystem{T}, SparseSchurLinearSystem{T}}) where T
    st = ls.H.structure
    diag_retained!(d, ls.H.R, st)
    for e in 1:st.neblocks
        eo = st.eoffsets[e]
        o = st.nr + st.boffsets_e[e] - 1
        de = Int(st.e_dims[e])
        for i in 1:de
            d[o + i] = ls.H.E_data[eo + (i - 1) * (de + 1)]
        end
    end
    return d
end

function diag_retained!(d, R::Matrix, st)
    for i in 1:st.nr
        d[i] = R[i, i]
    end
    return d
end

function diag_retained!(d, R::SparseBlockLinearSystem{T}, st) where T
    for b in 1:st.nrblocks
        ptr = R.indices[b, b]
        db = Int(st.r_dims[b])
        bo = st.boffsets_r[b] - 1
        for i in 1:db
            d[bo + i] = ptr > 0 ? R.data[ptr + (i - 1) * (db + 1)] : zero(T)
        end
    end
    return d
end

# The dense reduced system is a DampedLinearSolve system, conditioned there with a scale computed at its first solve
# after a fill (decision 13); a new linearization is a new fill of it, so it is zeroed with the Hessian (its matrix is
# rewritten by every reduction, so only its conditioning scale is what this resets)
function DampedLinearSolve.zero!(ls::DenseSchurLinearSystem)
    fill!(ls.b, zero(eltype(ls.b)))
    BlockLinearSystems.zero!(ls.H)
    DampedLinearSolve.zero!(ls.red)
    return ls
end

function DampedLinearSolve.zero!(ls::SparseSchurLinearSystem)
    fill!(ls.b, zero(eltype(ls.b)))
    BlockLinearSystems.zero!(ls.H)
    return ls
end

# Build the reduced system S and g_S, with the damping, from scratch. Called at the start of each solve.
function build_reduced!(ls::DenseSchurLinearSystem)
    reset_schur_dense!(ls.S, ls.H.R, ls.scaling_r, ls.lambda, ls.workspace)
    return reduce_schur_dense!(
        ls.S, ls.g_S, ls.b_r, ls.b_e, ls.scaling_e, ls.lambda,
        ls.H.structure, ls.H.E_data, ls.H.C_data, ls.workspace
    )
end

# The reduced solve of a dense S, through the DampedLinearSolve system holding it (which factors a copy, so S
# is left as built): Cholesky, falling back to Bunch-Kaufman, for :auto and :cholesky; Bunch-Kaufman for :ldl
function solve_reduced_dense!(red, ::StaticSymbol{:ldl})
    return solve!(red, static(:ldl))
end
function solve_reduced_dense!(red, ::Union{StaticSymbol{:auto}, StaticSymbol{:cholesky}})
    status = solve!(red, static(:cholesky))
    status == Success && return status
    return solve!(red, static(:ldl))
end

DampedLinearSolve.solve!(ls::DenseSchurLinearSystem) = DampedLinearSolve.solve!(ls, static(:auto))
DampedLinearSolve.solve!(ls::DenseSchurLinearSystem, s::StaticSymbol) = check_reduced_solver(s)   # throws: not a ReducedSolver
function DampedLinearSolve.solve!(ls::DenseSchurLinearSystem, solver_opt::ReducedSolver)
    build_reduced!(ls) || return NotPositiveDefinite
    status = solve_reduced_dense!(ls.red, solver_opt)
    status == NotPositiveDefinite && return status
    # An Indefinite solve has solved too: the whole of x, as DampedLinearSolve's contract requires
    copyto!(ls.x_r, getx(ls.red))
    backsubstitute!(ls.x_e, ls.x_r, ls.H.structure, ls.H.C_data, ls.workspace)
    # A non-finite R or C can pass the factorization; a non-finite solution does not
    return all(isfinite, ls.x) ? status : NotPositiveDefinite
end
DampedLinearSolve.solve!(ls::DenseSchurLinearSystem, s::Symbol) = DampedLinearSolve.solve!(ls, static(s))

# Copy the values of S into the persistent CHOLMOD.Sparse in place. Its pattern (upper triangle of S, same
# order as S.nzval) never changes, so this avoids rebuilding a CHOLMOD.Sparse (pattern and values) per solve.
function refresh_cholmod!(chol::SparseArrays.CHOLMOD.Sparse, S::SparseMatrixCSC)
    cs = unsafe_load(pointer(chol))
    cs.nzmax == length(S.nzval) || error("CHOLMOD.Sparse pattern does not match S")
    GC.@preserve chol begin
        copyto!(unsafe_wrap(Array, Ptr{eltype(S)}(cs.x), cs.nzmax), S.nzval)
    end
    return chol
end

# `ldiv!(x, ::CHOLMOD.Factor, b)` only exists from Julia 1.12; earlier versions need the allocating `\\`.
@static if VERSION >= v"1.12"
    cholmod_ldiv!(x, F::SparseArrays.CHOLMOD.Factor, b) = ldiv!(x, F, b)
else
    cholmod_ldiv!(x, F::SparseArrays.CHOLMOD.Factor, b) = copyto!(x, F \ b)
end

DampedLinearSolve.solve!(ls::SparseSchurLinearSystem) = DampedLinearSolve.solve!(ls, static(:auto))
DampedLinearSolve.solve!(ls::SparseSchurLinearSystem, s::StaticSymbol) = check_reduced_solver(s)   # throws: not a ReducedSolver
function DampedLinearSolve.solve!(ls::SparseSchurLinearSystem, solver_opt::ReducedSolver)
    reset_schur_sparse!(
        ls.S.nzval, ls.H.R.data, ls.r_to_s_map, ls.diag_inds, ls.scaling_r, ls.lambda, ls.workspace
    )
    success = reduce_schur_sparse!(
        ls.S, ls.g_S, ls.b_r, ls.b_e, ls.scaling_e, ls.lambda,
        ls.H.structure, ls.H.E_data, ls.H.C_data,
        ls.block_colptr, ls.block_rowval, ls.block_row_offsets, ls.col_heights,
        ls.workspace
    )
    if !success
        return NotPositiveDefinite
    end

    status = Success
    copyto!(ls.rhs, ls.g_S)   # narrowed to T
    if solver_opt isa StaticSymbol{:ldl}
        F = ldl_factorize!(Symmetric(ls.S, :U), ls.S_factor_ldl)
        if !factorized(F)
            return NotPositiveDefinite
        end
        ldiv!(ls.sol, F, ls.rhs)
        is_indefinite(F) && (status = Indefinite)
    else
        refresh_cholmod!(ls.S_chol, ls.S)
        LinearAlgebra.cholesky!(ls.S_factor, ls.S_chol; check=false)
        if !LinearAlgebra.issuccess(ls.S_factor)
            F = ldl_factorize!(Symmetric(ls.S, :U), ls.S_factor_ldl)
            if !factorized(F)
                return NotPositiveDefinite
            end
            ldiv!(ls.sol, F, ls.rhs)
            is_indefinite(F) && (status = Indefinite)
        else
            cholmod_ldiv!(ls.sol, ls.S_factor, ls.rhs)
        end
    end
    copyto!(ls.x_r, ls.sol)

    # An Indefinite solve has solved too: the whole of x, as DampedLinearSolve's contract requires
    backsubstitute!(ls.x_e, ls.x_r, ls.H.structure, ls.H.C_data, ls.workspace)
    # A non-finite R or C can pass the factorization; a non-finite solution does not
    return all(isfinite, ls.x) ? status : NotPositiveDefinite
end
DampedLinearSolve.solve!(ls::SparseSchurLinearSystem, s::Symbol) = DampedLinearSolve.solve!(ls, static(s))

# `quadratic_model_change` is intentionally not specialised: DampedLinearSolve's generic method
# (0.5 * quadform(getA(ls), dx) + fast_dot(getb(ls), dx)) is the undamped model change. Its quadratic
# form falls back to LinearAlgebra.dot(dx, H, dx), which BlockLinearSystems defines for a
# SchurBlockHessian. (An earlier override added the damping term with the wrong sign, and used a
# damping value the iterators have already reset to zero when a step is scored.)

function DampedLinearSolve.copy_structure(ls::DenseSchurLinearSystem{T, R}) where {T, R}
    # New storage for the Hessian and gradient (the original is kept for retries); x is shared. The reduced
    # system's copy has its own S and g_S and shares its workspace, as the contract allows.
    nr = ls.H.structure.nr
    b = copy(ls.b)
    x = ls.x
    scaling = copy(ls.scaling)
    ws_trial = trial_workspace(ls.workspace)
    red = DampedLinearSolve.copy_structure(ls.red)
    return DenseSchurLinearSystem{T, R}(
        BlockLinearSystems.copy_structure(ls.H), b, x,
        view(b, 1:nr), view(b, nr+1:length(b)),
        view(x, 1:nr), view(x, nr+1:length(x)),
        red, getb(red), getA(red), ls.lambda, scaling,
        view(scaling, 1:nr), view(scaling, nr+1:length(scaling)),
        ws_trial
    )
end


function DampedLinearSolve.copy_structure(ls::SparseSchurLinearSystem{T, Ti, TL, TF, TC}) where {T, Ti, TL, TF, TC}
    # New storage for the Hessian and gradient (the original is kept for retries); x is shared
    nr = ls.H.structure.nr
    b = copy(ls.b)
    x = ls.x
    scaling = copy(ls.scaling)
    ws_trial = trial_workspace(ls.workspace)
    return SparseSchurLinearSystem{T, Ti, TL, TF, TC}(
        BlockLinearSystems.copy_structure(ls.H), b, x,
        view(b, 1:nr), view(b, nr+1:length(b)),
        view(x, 1:nr), view(x, nr+1:length(x)),
        similar(ls.g_S), copy(ls.S), ls.S_factor, ls.S_factor_ldl, isnothing(ls.S_chol) ? nothing : copy(ls.S_chol), ls.rhs, ls.sol,
        ls.r_to_s_map, ls.diag_inds,
        ls.block_rowval, ls.block_colptr, ls.block_row_offsets, ls.col_heights,
        ls.lambda, scaling,
        view(scaling, 1:nr), view(scaling, nr+1:length(scaling)),
        ws_trial
    )
end

BlockLinearSystems.copy_structure(ls::DenseSchurLinearSystem) = DampedLinearSolve.copy_structure(ls)
BlockLinearSystems.copy_structure(ls::SparseSchurLinearSystem) = DampedLinearSolve.copy_structure(ls)

function DampedLinearSolve.sharex(ls::DenseSchurLinearSystem{T, R}, from) where {T, R}
    ls.x === from.x && return ls
    nr = ls.H.structure.nr
    x = from.x
    return DenseSchurLinearSystem{T, R}(
        ls.H, ls.b, x,
        ls.b_r, ls.b_e,
        view(x, 1:nr), view(x, nr+1:length(x)),
        ls.red, ls.g_S, ls.S, ls.lambda, ls.scaling,
        ls.scaling_r, ls.scaling_e,
        ls.workspace
    )
end

function DampedLinearSolve.sharex(ls::SparseSchurLinearSystem{T, Ti, TL, TF, TC}, from) where {T, Ti, TL, TF, TC}
    ls.x === from.x && return ls
    nr = ls.H.structure.nr
    x = from.x
    return SparseSchurLinearSystem{T, Ti, TL, TF, TC}(
        ls.H, ls.b, x,
        ls.b_r, ls.b_e,
        view(x, 1:nr), view(x, nr+1:length(x)),
        ls.g_S, ls.S, ls.S_factor, ls.S_factor_ldl, ls.S_chol, ls.rhs, ls.sol,
        ls.r_to_s_map, ls.diag_inds,
        ls.block_rowval, ls.block_colptr, ls.block_row_offsets, ls.col_heights,
        ls.lambda, ls.scaling,
        ls.scaling_r, ls.scaling_e,
        ls.workspace
    )
end

# The element type of H, Float64 or Float32, is the system's: the reduction and the reduced factorization are in it;
# b and x are Float64. A reduced solver that does not offer it is an ArgumentError here (single-precision.md,
# decision 10).
function DampedLinearSolve.makehessiansystem(H::SchurBlockHessian{T}, linsolver) where T
    nr = H.structure.nr
    ne = sum(Int, H.structure.e_dims; init=0)
    tot = nr + ne
    b = zeros(tot)
    x = zeros(tot)
    check_reduced_solver(linsolver)
    check_reduced_eltype(H, linsolver)
    # The storage of R picks the dense or sparse Schur system. The sparse system reuses a Schur
    # pattern carried by H (BlockLinearSystems versions that predate the field have none); the dense
    # system has no use for one.
    pattern = hasfield(typeof(H), :pattern) ? H.pattern : nothing
    return H.R isa SparseBlockLinearSystem ? SparseSchurLinearSystem(H, b, x; pattern, cholmod = reduced_cholmod(T)) : DenseSchurLinearSystem(H, b, x)
end
DampedLinearSolve.systemeltype(::Union{DenseSchurLinearSystem{T}, SparseSchurLinearSystem{T}}) where T = T

# The prototype's checks and workspace key: a Float64 or Float32 SchurBlockHessian of the right size; another
# element type is rejected before any work.
function DampedLinearSolve.checkhessian(H::SchurBlockHessian{T}, n::Integer) where T
    T in REDUCED_ELTYPES || throw(ArgumentError("The Hessian prototype must be a SchurBlockHessian{Float64} or SchurBlockHessian{Float32}; got a $(typeof(H))."))
    m = H.structure.nr + sum(Int, H.structure.e_dims; init=0)
    n == m || throw(DimensionMismatch("Hessian prototype H is a SchurBlockHessian of $m variables, but the variables have $n degrees of freedom."))
    return nothing
end
# As on the Jacobian path, a workspace is reused for an H of the same type (which carries the reduced
# system's storage) and the same (identical) structure.
DampedLinearSolve.structurekey(H::SchurBlockHessian) = (typeof(H), H.structure)
