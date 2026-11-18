# `linsolver` names the solver for the reduced system S. Only these are implemented for it; any other
# is rejected when the system is built, rather than silently treated as `:auto`.
const REDUCED_SOLVERS = (:auto, :cholesky, :ldl)
const ReducedSolver = Union{StaticSymbol{:auto}, StaticSymbol{:cholesky}, StaticSymbol{:ldl}}
function check_reduced_solver(linsolver)
    sym = linsolver_symbol(linsolver)
    sym in REDUCED_SOLVERS || throw(ArgumentError("linsolver = :$sym is not implemented for the reduced system of a SchurBlockHessian; use :auto or :cholesky (Cholesky, falling back to Bunch-Kaufman) or :ldl (Bunch-Kaufman)."))
    return nothing
end

"""
    DenseSchurLinearSystem{T} <: AbstractLinearSystem

Linear system solving bipartite normal equations via dense Schur reduction.
"""
mutable struct DenseSchurLinearSystem{T, R} <: AbstractLinearSystem
    H::SchurBlockHessian{T, Matrix{T}}
    b::Vector{T}
    x::Vector{T}
    b_r::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    b_e::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    x_r::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    x_e::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    red::R                            # the reduced system S x_r = g_S: a dense DampedLinearSolve system, undamped
    g_S::Vector{T}                    # getb(red)
    S::Matrix{T}                      # getA(red): the reduction writes its upper triangle
    lambda::T
    scaling::Vector{T}
    scaling_r::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    scaling_e::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    workspace::SchurReductionWorkspace{T}
end

function DenseSchurLinearSystem(
    H::SchurBlockHessian{T, Matrix{T}},
    b::Vector{T}, x::Vector{T},
    scaling::Vector{T}=ones(T, length(b));
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

    # Built for :ldl, so that it holds the Bunch-Kaufman workspace; :cholesky needs none
    red = makehessiansystem(nr, static(:ldl))

    K = nchunks > 0 ? nchunks : (isempty(H.structure.chunks) ? 1 : length(H.structure.chunks[1]))
    ws = SchurReductionWorkspace{T}(H.structure, K)

    return DenseSchurLinearSystem{T, typeof(red)}(
        H, b, x, b_r, b_e, x_r, x_e, red, getb(red), getA(red), zero(T), scaling, scaling_r, scaling_e, ws
    )
end

"""
    SparseSchurLinearSystem{T, Ti, TL, TF, TC} <: AbstractLinearSystem

Linear system solving bipartite normal equations via sparse Schur reduction, CHOLMOD sparse Cholesky, and LDL factorizations.
"""
mutable struct SparseSchurLinearSystem{T, Ti, TL, TF, TC} <: AbstractLinearSystem
    H::SchurBlockHessian{T, SparseBlockLinearSystem{T}}
    b::Vector{T}
    x::Vector{T}
    b_r::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    b_e::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    x_r::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    x_e::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    g_S::Vector{T}
    S::SparseMatrixCSC{T, Ti}
    S_factor::TF                      # CHOLMOD.Factor (its type parameters differ between Julia versions)
    S_factor_ldl::TL
    S_chol::TC                        # CHOLMOD.Sparse copy of S, refreshed in place before each factorization
    r_to_s_map::Vector{Int}
    diag_inds::Vector{Int}
    block_rowval::Vector{Ti}
    block_colptr::Vector{Ti}
    block_row_offsets::Vector{Int}
    col_heights::Vector{Int}
    lambda::T
    scaling::Vector{T}
    scaling_r::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    scaling_e::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    workspace::SchurReductionWorkspace{T}
end

function SparseSchurLinearSystem(
    H::SchurBlockHessian{T, SparseBlockLinearSystem{T}},
    b::Vector{T}, x::Vector{T},
    scaling::Vector{T}=ones(T, length(b));
    nchunks::Int=0,
    pattern=nothing
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
    S_chol = SparseArrays.CHOLMOD.Sparse(Symmetric(S, :U))
    S_factor = hasproperty(pattern, :S_factor) ? pattern.S_factor : SparseArrays.CHOLMOD.analyze(S_chol)
    S_factor_ldl = ldl_analyze(Symmetric(S, :U))

    g_S = zeros(T, nr)

    K = nchunks > 0 ? nchunks : (isempty(H.structure.chunks) ? 1 : length(H.structure.chunks[1]))
    ws = SchurReductionWorkspace{T}(H.structure, K)

    return SparseSchurLinearSystem{T, Ti, typeof(S_factor_ldl), typeof(S_factor), typeof(S_chol)}(
        H, b, x, b_r, b_e, x_r, x_e, g_S, S, S_factor, S_factor_ldl, S_chol,
        r_to_s_map, pattern.diag_inds,
        pattern.block_rowval, pattern.block_colptr, pattern.block_row_offsets, pattern.col_heights,
        zero(T), scaling, scaling_r, scaling_e, ws
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
function DampedLinearSolve.dampen!(ls::Union{DenseSchurLinearSystem{T}, SparseSchurLinearSystem{T}}, lambda::Real) where T
    ls.lambda = convert(T, lambda)
    fill!(ls.scaling, one(T))
    return ls
end

function DampedLinearSolve.dampen!(ls::Union{DenseSchurLinearSystem{T}, SparseSchurLinearSystem{T}}, lambda::Real, scaling) where T
    scaling === nothing && return DampedLinearSolve.dampen!(ls, lambda)
    ls.lambda = convert(T, lambda)
    copyto!(ls.scaling, scaling)
    return ls
end

function DampedLinearSolve.max_diag(H::SchurBlockHessian{T}) where T
    m = zero(T)
    if H.R isa Matrix
        for i in 1:H.structure.nr
            m = max(m, abs(H.R[i, i]))
        end
    elseif H.R isa SparseBlockLinearSystem
        for b in 1:H.structure.nrblocks
            ptr = H.R.indices[b, b]
            if ptr > 0
                db = Int(H.structure.r_dims[b])
                for k in 1:db
                    m = max(m, abs(H.R.data[ptr + (k - 1) * (db + 1)]))
                end
            end
        end
    end
    for e in 1:H.structure.neblocks
        eo = H.structure.eoffsets[e]
        de = Int(H.structure.e_dims[e])
        for k in 1:de
            m = max(m, abs(H.E_data[eo + (k - 1) * (de + 1)]))
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

function DampedLinearSolve.zero!(ls::DenseSchurLinearSystem)
    fill!(ls.b, zero(eltype(ls.b)))
    BlockLinearSystems.zero!(ls.H)
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
    status == NotPositiveDefinite || copyto!(ls.x_r, getx(ls.red))   # an Indefinite solve has solved, too
    status == Success || return status

    backsubstitute!(ls.x_e, ls.x_r, ls.H.structure, ls.H.C_data, ls.workspace)
    # A non-finite R or C can pass the factorization; a non-finite solution does not
    return all(isfinite, ls.x) ? Success : NotPositiveDefinite
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

    if solver_opt isa StaticSymbol{:ldl}
        F = ldl_factorize!(Symmetric(ls.S, :U), ls.S_factor_ldl)
        if !factorized(F)
            return NotPositiveDefinite
        end
        ldiv!(ls.x_r, F, ls.g_S)
        if is_indefinite(F)
            return Indefinite
        end
    else
        refresh_cholmod!(ls.S_chol, ls.S)
        LinearAlgebra.cholesky!(ls.S_factor, ls.S_chol; check=false)
        if !LinearAlgebra.issuccess(ls.S_factor)
            F = ldl_factorize!(Symmetric(ls.S, :U), ls.S_factor_ldl)
            if !factorized(F)
                return NotPositiveDefinite
            end
            ldiv!(ls.x_r, F, ls.g_S)
            if is_indefinite(F)
                return Indefinite
            end
        else
            cholmod_ldiv!(ls.x_r, ls.S_factor, ls.g_S)
        end
    end

    backsubstitute!(ls.x_e, ls.x_r, ls.H.structure, ls.H.C_data, ls.workspace)
    # A non-finite R or C can pass the factorization; a non-finite solution does not
    return all(isfinite, ls.x) ? Success : NotPositiveDefinite
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
        similar(ls.g_S), copy(ls.S), ls.S_factor, ls.S_factor_ldl, copy(ls.S_chol),
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
        ls.g_S, ls.S, ls.S_factor, ls.S_factor_ldl, ls.S_chol,
        ls.r_to_s_map, ls.diag_inds,
        ls.block_rowval, ls.block_colptr, ls.block_row_offsets, ls.col_heights,
        ls.lambda, ls.scaling,
        ls.scaling_r, ls.scaling_e,
        ls.workspace
    )
end

function DampedLinearSolve.makehessiansystem(H::SchurBlockHessian{T}, linsolver) where T
    nr = H.structure.nr
    ne = sum(Int, H.structure.e_dims; init=0)
    tot = nr + ne
    b = zeros(T, tot)
    x = zeros(T, tot)
    check_reduced_solver(linsolver)
    # The storage of R picks the dense or sparse Schur system. The sparse system reuses a Schur
    # pattern carried by H (BlockLinearSystems versions that predate the field have none); the dense
    # system has no use for one.
    pattern = hasfield(typeof(H), :pattern) ? H.pattern : nothing
    return H.R isa SparseBlockLinearSystem ? SparseSchurLinearSystem(H, b, x; pattern) : DenseSchurLinearSystem(H, b, x)
end

# The prototype's checks and workspace key. The optimizer works in Float64 (as the Jacobian path's
# `checkjacobian` requires of a SchurBlockJacobian), so another element type is rejected before any work.
function DampedLinearSolve.checkhessian(H::SchurBlockHessian{T}, n::Integer) where T
    T === Float64 || throw(ArgumentError("The Hessian prototype must be Float64; got a $(typeof(H))."))
    m = H.structure.nr + sum(Int, H.structure.e_dims; init=0)
    n == m || throw(DimensionMismatch("Hessian prototype H is a SchurBlockHessian of $m variables, but the variables have $n degrees of freedom."))
    return nothing
end
# As on the Jacobian path, a workspace is reused for an H of the same type (which carries the reduced
# system's storage) and the same (identical) structure.
DampedLinearSolve.structurekey(H::SchurBlockHessian) = (typeof(H), H.structure)
