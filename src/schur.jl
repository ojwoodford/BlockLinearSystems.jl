using Static, StaticArrays, SparseArrays, LinearAlgebra

"""
    inv_sym_block(H::SMatrix{P, P, T}, lambda::T, scale::SVector{P, T}) where {P, T}

Invert a symmetric positive semi-definite block `H` with diagonal Levenberg-Marquardt damping
`H + lambda * diag(scale)` (`scale` is the damping diagonal itself, e.g. `|diag(H)|` for Marquardt scaling) in registers.
Returns `(invH, success)` where `success` is false if the block is numerically rank-deficient.
"""
# Relative tolerance used to declare a damped block numerically singular: 1e-12 for Float64, scaled
# with the precision of T.
@inline singular_tol(::Type{T}) where T = 4504 * eps(T)

@inline function inv_sym_block(H::SMatrix{1, 1, T, 1}, lambda::T, scale::SVector{1, T}) where T
    h11 = H[1, 1] + lambda * scale[1]
    if h11 <= eps(T) * abs(H[1, 1]) || !isfinite(h11)
        return (invH = zeros(SMatrix{1, 1, T, 1}), success = false)
    end
    return (invH = SMatrix{1, 1, T, 1}((inv(h11),)), success = true)
end

@inline function inv_sym_block(H::SMatrix{2, 2, T, 4}, lambda::T, scale::SVector{2, T}) where T
    H11 = H[1, 1] + lambda * scale[1]
    H22 = H[2, 2] + lambda * scale[2]
    H12 = H[1, 2]

    det = H11 * H22 - H12 * H12
    if det <= singular_tol(T) * (H11 * H22) || !isfinite(det)
        return (invH = zeros(SMatrix{2, 2, T, 4}), success = false)
    end
    inv_det = inv(det)
    invH = SMatrix{2, 2, T, 4}((H22 * inv_det, -H12 * inv_det, -H12 * inv_det, H11 * inv_det))
    return (invH = invH, success = true)
end

@inline function inv_sym_block(H::SMatrix{3, 3, T, 9}, lambda::T, scale::SVector{3, T}) where T
    H11 = H[1, 1] + lambda * scale[1]
    H22 = H[2, 2] + lambda * scale[2]
    H33 = H[3, 3] + lambda * scale[3]
    H12 = H[1, 2]; H13 = H[1, 3]; H23 = H[2, 3]

    c11 = H22 * H33 - H23 * H23
    c12 = H13 * H23 - H12 * H33
    c13 = H12 * H23 - H13 * H22
    c22 = H11 * H33 - H13 * H13
    c23 = H12 * H13 - H11 * H23
    c33 = H11 * H22 - H12 * H12

    det = H11 * c11 + H12 * c12 + H13 * c13
    diag_prod = H11 * H22 * H33

    if det <= singular_tol(T) * diag_prod || !isfinite(det)
        return (invH = zeros(SMatrix{3, 3, T, 9}), success = false)
    end

    inv_det = inv(det)
    invH = SMatrix{3, 3, T, 9}((
        c11 * inv_det, c12 * inv_det, c13 * inv_det,
        c12 * inv_det, c22 * inv_det, c23 * inv_det,
        c13 * inv_det, c23 * inv_det, c33 * inv_det
    ))
    return (invH = invH, success = true)
end

@inline function inv_sym_block(H::SMatrix{D, D, T}, lambda::T, scale::SVector{D, T}) where {D, T}
    Hd = H + lambda * SMatrix{D, D, T}(Diagonal(scale))
    chol = cholesky(Symmetric(Hd, :U); check=false)
    ok = issuccess(chol)
    if ok
        # Every pivot must be safely positive relative to its damped diagonal entry. (A determinant
        # test, as for the closed-form 1-3 dimensional blocks, would underflow for large blocks.)
        U = chol.U
        min_ratio = minimum(i -> abs2(U[i, i]) / Hd[i, i], ntuple(identity, Val(D)))
        ok = min_ratio > singular_tol(T) && isfinite(min_ratio)
    end
    ok || return (invH = zeros(SMatrix{D, D, T, D * D}), success = false)
    return (invH = SMatrix{D, D, T, D * D}(chol \ one(SMatrix{D, D, T})), success = true)
end

"""
    SchurStructure

Metadata describing bipartite variable partitioning, block offsets, and coupling sparsity.
"""
struct SchurStructure
    nr::Int                          # Total scalar dimension of retained variables
    nrblocks::Int                    # Number of retained blocks
    neblocks::Int                    # Number of eliminated blocks
    r_dims::Vector{UInt8}            # Dimension of each retained block
    e_dims::Vector{UInt8}            # Dimension of each eliminated block
    boffsets_r::Vector{Int}          # Scalar start index of each retained block in x_r
    boffsets_e::Vector{Int}          # Scalar start index of each eliminated block in x_e
    eoffsets::Vector{Int}           # Start offset in flat E_data for each E_e block
    eliminated_ptr::Vector{Int}               # Eliminated-block-major CSR pointer for coupling entries
    entry_retained::Vector{Int}             # Retained block index for each coupling entry (sorted ascending per eliminated block!)
    coffsets::Vector{Int}           # Start offset in flat C_data for each coupling block
    max_coupled_dim::Int                 # Maximum over eliminated blocks e of sum_{r coupled to e} d_r
    e_groups::Vector{UnitRange{Int}} # Groups of eliminated blocks by uniform dimension
    e_sizes::Vector{Int}             # Dimension of the eliminated blocks in each group
    chunks::Vector{Vector{UnitRange{Int}}} # chunks[group_idx][chunk_id]
end

# Largest supported dimension of an eliminated block
const MAX_ELIMINATED_DIM = 255

function SchurStructure(
    r_dims::AbstractVector{<:Integer},
    e_dims::AbstractVector{<:Integer},
    couplings::AbstractVector{<:AbstractVector{<:Integer}};
    nchunks::Int=1
)
    nrblocks = length(r_dims)
    neblocks = length(e_dims)
    length(couplings) == neblocks || throw(DimensionMismatch("Number of coupling lists ($(length(couplings))) must match the number of eliminated blocks ($neblocks)"))
    nchunks >= 1 || throw(ArgumentError("nchunks must be at least 1, got $nchunks"))

    # Validate inputs here, rather than failing (or silently going wrong) deep inside the reduction
    for (i, d) in enumerate(r_dims)
        1 <= d <= 255 || throw(ArgumentError("Retained block $i has dimension $d; dimensions must be between 1 and 255"))
    end
    for (e, d) in enumerate(e_dims)
        1 <= d <= MAX_ELIMINATED_DIM || throw(ArgumentError("Eliminated block $e has dimension $d; supported dimensions are 1 to $MAX_ELIMINATED_DIM"))
    end
    seen = zeros(Int, nrblocks)   # seen[r] == e once eliminated block e has been coupled to retained block r
    for (e, obs) in enumerate(couplings)
        for r in obs
            1 <= r <= nrblocks || throw(ArgumentError("Eliminated block $e references retained block $r, but there are only $nrblocks retained blocks"))
            seen[r] == e && throw(ArgumentError("Retained block $r is listed more than once in the couplings of eliminated block $e"))
            seen[r] = e
        end
    end

    r_dims_u8 = UInt8.(r_dims)
    e_dims_u8 = UInt8.(e_dims)

    nr = sum(Int, r_dims; init=0)

    boffsets_r = Vector{Int}(undef, nrblocks)
    if nrblocks > 0
        boffsets_r[1] = 1
        for i in 2:nrblocks
            boffsets_r[i] = boffsets_r[i-1] + Int(r_dims[i-1])
        end
    end

    boffsets_e = Vector{Int}(undef, neblocks)
    if neblocks > 0
        boffsets_e[1] = 1
        for i in 2:neblocks
            boffsets_e[i] = boffsets_e[i-1] + Int(e_dims[i-1])
        end
    end

    eoffsets = Vector{Int}(undef, neblocks)
    if neblocks > 0
        eoffsets[1] = 1
        for i in 2:neblocks
            eoffsets[i] = eoffsets[i-1] + Int(e_dims[i-1])^2
        end
    end

    eliminated_ptr = Vector{Int}(undef, neblocks + 1)
    eliminated_ptr[1] = 1
    for e in 1:neblocks
        eliminated_ptr[e+1] = eliminated_ptr[e] + length(couplings[e])
    end
    num_couplings = eliminated_ptr[end] - 1

    entry_retained = Vector{Int}(undef, num_couplings)
    coffsets = Vector{Int}(undef, num_couplings)
    curr_c_off = 1
    idx = 1
    for e in 1:neblocks
        de = Int(e_dims[e])
        sorted_retained = sort(couplings[e])
        for r in sorted_retained
            entry_retained[idx] = r
            coffsets[idx] = curr_c_off
            curr_c_off += Int(r_dims[r]) * de
            idx += 1
        end
    end

    max_coupled_dim = 0
    for e in 1:neblocks
        obs_dim = sum(Int(r_dims[r]) for r in couplings[e]; init=0)
        max_coupled_dim = max(max_coupled_dim, obs_dim)
    end

    e_groups = UnitRange{Int}[]
    e_sizes = Int[]
    if neblocks > 0
        curr_dim = Int(e_dims[1])
        start_idx = 1
        for e in 2:neblocks
            d = Int(e_dims[e])
            if d != curr_dim
                push!(e_groups, start_idx:e-1)
                push!(e_sizes, curr_dim)
                curr_dim = d
                start_idx = e
            end
        end
        push!(e_groups, start_idx:neblocks)
        push!(e_sizes, curr_dim)
    end

    chunks = Vector{Vector{UnitRange{Int}}}(undef, length(e_groups))
    for (g_idx, grp) in enumerate(e_groups)
        len = length(grp)
        group_chunks = Vector{UnitRange{Int}}(undef, nchunks)
        chunk_sz = cld(len, nchunks)
        for k in 1:nchunks
            first_idx = grp.start + (k - 1) * chunk_sz
            last_idx = min(grp.stop, first_idx + chunk_sz - 1)
            group_chunks[k] = first_idx <= last_idx ? (first_idx:last_idx) : (first_idx:first_idx-1)
        end
        chunks[g_idx] = group_chunks
    end

    return SchurStructure(
        nr, nrblocks, neblocks,
        r_dims_u8, e_dims_u8, boffsets_r, boffsets_e,
        eoffsets, eliminated_ptr, entry_retained, coffsets,
        max_coupled_dim, e_groups, e_sizes, chunks
    )
end

"""
    SchurReductionWorkspace{T}

Pre-allocated scratch memory and caches for parallel Schur reduction and back-substitution.

The reduction runs in two phases. Phase 1 (parallel over chunks of eliminated blocks) factors every
`E_e`, caching `E_inv`, `z_e` and `W_er = E_inv * C_er` (`W_data`, laid out like `C_data`). Phase 2
(parallel over `col_chunks`, disjoint ranges of retained blocks) has each chunk write straight into the
columns of `S` (and entries of `g_S`) that it owns, so no per-chunk copies of `S` are needed and nothing
has to be merged afterwards.
"""
struct SchurReductionWorkspace{T}
    W_scratch::Vector{Matrix{T}}          # per phase-1 chunk: max_De x max_De scratch for blocks above D_STATIC (0 x 0 otherwise)
    W_data::Vector{T}                     # W_er = E_e^-1 C_er for every coupling, same layout as C_data
    E_inv_data::Vector{T}                 # Cached E_e(lambda)^-1: length sum_e d_e^2
    z_data::Vector{T}                     # Cached z_e = E_e(lambda)^-1 b_e: length sum_e d_e
    retained_ptr::Vector{Int}                  # Retained-block-major CSR pointer into retained_entry
    retained_entry::Vector{Int}                # Coupling entries (indices into entry_retained) of each retained block, ascending
    entry_eliminated::Vector{Int}           # Eliminated block of each coupling entry
    col_chunks::Vector{UnitRange{Int}}    # Disjoint ranges of retained blocks, one per phase-2 chunk
    failed_block::Base.Threads.Atomic{Int}
end

"Split the retained blocks `1:n` into `K` contiguous ranges of roughly equal work (couplings plus the diagonal)."
function partition_columns(retained_ptr::Vector{Int}, K::Int)
    n = length(retained_ptr) - 1
    total = retained_ptr[end] - retained_ptr[1] + n
    ranges = UnitRange{Int}[]
    start = 1
    for k in 1:K
        stop = start - 1
        if k == K
            stop = n
        else
            target = (total * k) / K
            while stop < n && (retained_ptr[stop + 2] - retained_ptr[1] + (stop + 1)) <= target
                stop += 1
            end
        end
        push!(ranges, start:stop)
        start = stop + 1
    end
    return ranges
end

function SchurReductionWorkspace{T}(
    structure::SchurStructure, nchunks::Int;
    is_dense::Bool=true, nnz_S::Int=0   # accepted for compatibility: no per-chunk copies of S are needed any more
) where T
    max_De = maximum(structure.e_sizes; init=3)

    # One scratch matrix per phase-1 chunk, only used for blocks above D_STATIC
    nscratch = max(nchunks, maximum(length, structure.chunks; init=1))
    W_scratch = [Matrix{T}(undef, max_De > D_STATIC ? max_De : 0, max_De > D_STATIC ? max_De : 0) for _ in 1:nscratch]

    total_r_size = sum(Int(d)^2 for d in structure.e_dims; init=0)
    total_e_scalars = sum(Int(d) for d in structure.e_dims; init=0)
    E_inv_data = Vector{T}(undef, total_r_size)
    z_data = Vector{T}(undef, total_e_scalars)

    ncoup = length(structure.entry_retained)
    total_c = 0
    for e in 1:structure.neblocks, c in structure.eliminated_ptr[e]:(structure.eliminated_ptr[e + 1] - 1)
        total_c += Int(structure.r_dims[structure.entry_retained[c]]) * Int(structure.e_dims[e])
    end
    W_data = Vector{T}(undef, total_c)

    # Retained-block-major index of the couplings
    entry_eliminated = Vector{Int}(undef, ncoup)
    counts = zeros(Int, structure.nrblocks + 1)
    for e in 1:structure.neblocks, c in structure.eliminated_ptr[e]:(structure.eliminated_ptr[e + 1] - 1)
        entry_eliminated[c] = e
        counts[structure.entry_retained[c] + 1] += 1
    end
    retained_ptr = cumsum(vcat(1, counts[2:end]))
    retained_entry = Vector{Int}(undef, ncoup)
    fill_pos = copy(retained_ptr)
    for c in 1:ncoup   # ascending c means ascending eliminated block within each retained block
        r = structure.entry_retained[c]
        retained_entry[fill_pos[r]] = c
        fill_pos[r] += 1
    end

    return SchurReductionWorkspace{T}(
        W_scratch, W_data, E_inv_data, z_data,
        retained_ptr, retained_entry, entry_eliminated, partition_columns(retained_ptr, max(nchunks, 1)),
        Base.Threads.Atomic{Int}(0)
    )
end

"""
    trial_workspace(ws)

A workspace for a second linear system with the same structure: own `E_inv`/`z` caches and failure flag;
the scratch (`W_data`, `W_scratch`) is only live within one solve and is shared.
"""
trial_workspace(ws::SchurReductionWorkspace{T}) where T = SchurReductionWorkspace{T}(
    ws.W_scratch, ws.W_data, similar(ws.E_inv_data), similar(ws.z_data),
    ws.retained_ptr, ws.retained_entry, ws.entry_eliminated, ws.col_chunks, Base.Threads.Atomic{Int}(0)
)

const SchurBlockStructure = SchurStructure

"""
    SchurBlockMatrix(structure, R, E_data, C_data; pattern=nothing)

Block Hessian structure partitioned into the retained blocks `R`,
eliminated blocks `E_data`, coupling blocks `C_data`,
and bipartite indexing `structure::SchurStructure`.

`pattern` optionally carries the result of [`build_schur_pattern`](@ref) for this structure, so that
code which receives only the matrix can build a reduced system without recomputing it. `nothing` (the
default) means "build it when needed". The pattern must come from
`build_schur_pattern(structure; extra_blocks=retained_block_pairs(R))`, or from a superset of those
pairs; only its size is checked (`size(pattern.S_sparse, 1) == structure.nr`, else `ArgumentError`).

The pattern is shared between every system built from `H`, including its [`copy_structure`](@ref)
copies, so treat it as read-only structure: in particular, copy `pattern.S_sparse` before using it as
value storage. Callers may attach extra entries to the named tuple (for example
`merge(pattern, (S_factor = fac,))`); this package ignores entries it does not know.

The positional form `SchurBlockMatrix(R, E_data, C_data, structure)` (field order) is also accepted.
"""
struct SchurBlockMatrix{T, RType}
    R::RType                         # Matrix{T} (dense) or SparseBlockLinearSystem (sparse)
    E_data::Vector{T}                # Contiguous storage for block-diagonal E (each E_e column-major): sum_e d_e^2
    C_data::Vector{T}                # Contiguous storage for block-sparse C (each C_er column-major, d_e x d_r): sum_c (d_e * d_r)
    structure::SchurStructure
    # Deliberately abstractly typed, and not a type parameter: a third parameter would make the field
    # declarations of downstream code abstract and its hot paths type-unstable. This is read only when a
    # linear system is constructed (a function barrier); assembly, reduction and fast_bAb never touch it.
    pattern::Union{Nothing, NamedTuple}
end

function SchurBlockMatrix(structure::SchurStructure, R::RType, E_data::Vector{T}, C_data::Vector{T};
                          pattern::Union{Nothing, NamedTuple}=nothing) where {T, RType}
    if pattern !== nothing && size(pattern.S_sparse, 1) != structure.nr
        throw(ArgumentError("pattern.S_sparse has size $(size(pattern.S_sparse, 1)), expected structure.nr = $(structure.nr)"))
    end
    return SchurBlockMatrix{T, RType}(R, E_data, C_data, structure, pattern)
end

SchurBlockMatrix{T, RType}(R, E_data, C_data, structure) where {T, RType} =
    SchurBlockMatrix{T, RType}(R, E_data, C_data, structure, nothing)
SchurBlockMatrix(R::RType, E_data::Vector{T}, C_data::Vector{T}, structure::SchurStructure) where {T, RType} =
    SchurBlockMatrix{T, RType}(R, E_data, C_data, structure, nothing)

function zero!(H::SchurBlockMatrix)
    if H.R isa Matrix
        fill!(H.R, zero(eltype(H.R)))
    else
        zero!(H.R)
    end
    fill!(H.E_data, zero(eltype(H.E_data)))
    fill!(H.C_data, zero(eltype(H.C_data)))
    return H
end

"""
    copy_structure(H::SchurBlockMatrix)

Return a new `SchurBlockMatrix` with the same structure but freshly allocated, zeroed storage for
`R`, `E_data` and `C_data`. Nothing is shared with `H` except the immutable `SchurStructure` and the
(read-only) `pattern`.
"""
function copy_structure(H::SchurBlockMatrix{T}) where T
    R = H.R isa AbstractMatrix ? zeros(T, size(H.R)) : copy_structure(H.R)
    return SchurBlockMatrix(H.structure, R, zeros(T, length(H.E_data)), zeros(T, length(H.C_data)); pattern=H.pattern)
end

# Block dimensions up to D_STATIC use statically sized (register) arithmetic, dispatched on StaticInt;
# larger ones use a dynamic path with a per-chunk scratch matrix. StaticArrays' Cholesky stops beating
# LAPACK at around this size, and every extra static size costs compile time.
const D_STATIC = 8

# Block dimension as an Int, from either a StaticInt or an Int
@inline dimval(::StaticInt{D}) where D = D
@inline dimval(d::Int) = d

@inline make_svec(arr::AbstractVector{T}, off::Int, ::StaticInt{D}) where {T, D} =
    SVector{D, T}(ntuple(i -> @inbounds(arr[off + i - 1]), Val(D)))

@inline make_smat(arr::AbstractVector{T}, off::Int, ::StaticInt{D}) where {T, D} =
    SMatrix{D, D, T, D * D}(ntuple(i -> @inbounds(arr[off + i - 1]), Val(D * D)))

# dot(a[ia:ia+de-1], b[ib:ib+de-1]) with a static or run-time dimension
@inline function dot_dim(a::Vector{T}, ia::Int, b::Vector{T}, ib::Int, s_de::Union{StaticInt, Int}) where T
    acc = zero(T)
    @inbounds for i in 0:dimval(s_de) - 1
        acc += a[ia + i] * b[ib + i]
    end
    return acc
end

# Call `f(s_de)` with the block dimension `de` as a StaticInt when `de <= D_STATIC`, else as an Int
@inline function dispatch_dim(f::F, de::Int) where F
    de == 1 && return f(static(1))
    de == 2 && return f(static(2))
    de == 3 && return f(static(3))
    de == 4 && return f(static(4))
    de == 5 && return f(static(5))
    de == 6 && return f(static(6))
    de == 7 && return f(static(7))
    de == 8 && return f(static(8))
    return f(de)
end

# Threading helper. Polyester (the same mechanism NLLSModel.jl uses for assembly, so the two never compete
# for threads) has low per-call overhead, but loops still run serially unless more than one thread is
# available *and* there is enough work to be worth splitting.
const SCHUR_THREAD_MIN_WORK = Ref(4096)

@inline use_threads(nitems::Integer, work::Integer=nitems) = nitems > 1 && work >= SCHUR_THREAD_MIN_WORK[] && Threads.nthreads() > 1

# Chunk 1 accumulates straight into the reduced system (`S` or `S.nzval`); only chunks 2..K need their
# own buffers, which are merged in afterwards. With a single chunk no extra memory is needed at all.
@inline chunk_buffer(target::AbstractArray, extra::Vector, k::Int) = k == 1 ? target : @inbounds(extra[k - 1])

# Kept out of line so that the serial path does not pay for the closure Polyester builds
@noinline function batch_foreach(f::F, n::Int) where F
    Polyester.@batch for i in 1:n
        f(i)
    end
    return nothing
end

@inline function foreach_index(f::F, n::Int, threaded::Bool) where F
    if threaded
        batch_foreach(f, n)
    else
        for i in 1:n
            f(i)
        end
    end
    return nothing
end

"""
    factor_eliminated!(E_inv_data, z_data, W_data, scratch, structure, E_data, C_data, b_e, scaling_e, lambda, e, s_de)

Phase 1 for eliminated block `e`: invert `E_e + lambda * diag(scaling)`, cache `E_inv` and
`z_e = E_inv * b_e`, and store `W_er = E_inv * C_er` for every coupled retained block in `W_data`.
Returns `false` if the damped block is numerically singular or not positive definite.

`s_de` is a `StaticInt` for blocks of up to `D_STATIC` dimensions (register arithmetic), otherwise the
`Int` dimension, which uses a LAPACK Cholesky on the `scratch` matrix (O(d^3) per eliminated block plus
O(d^2) per coupling row; no allocation).
"""
@inline function factor_eliminated!(
    E_inv_data::Vector{T}, z_data::Vector{T}, W_data::Vector{T}, ::Matrix{T},
    structure::SchurStructure, E_data::Vector{T}, C_data::Vector{T},
    b_e::AbstractVector{T}, scaling_e::AbstractVector{T}, lambda::T, e::Int, s_de::StaticInt{De}
) where {T, De}
    eo = structure.eoffsets[e]
    sc_off = structure.boffsets_e[e]

    res = inv_sym_block(make_smat(E_data, eo, s_de), lambda, make_svec(scaling_e, sc_off, s_de))
    res.success || return false
    E_inv = res.invH
    @inbounds for i in 1:(De * De)
        E_inv_data[eo + i - 1] = E_inv[i]
    end

    z_e = E_inv * make_svec(b_e, sc_off, s_de)
    @inbounds for i in 1:De
        z_data[sc_off + i - 1] = z_e[i]
    end

    @inbounds for c in structure.eliminated_ptr[e]:(structure.eliminated_ptr[e + 1] - 1)
        co = structure.coffsets[c]
        dr = Int(structure.r_dims[structure.entry_retained[c]])
        for ir in 1:dr
            w_ir = E_inv * make_svec(C_data, co + (ir - 1) * De, s_de)
            for i in 1:De
                W_data[co + (ir - 1) * De + i - 1] = w_ir[i]
            end
        end
    end
    return true
end

@inline function factor_eliminated!(
    E_inv_data::Vector{T}, z_data::Vector{T}, W_data::Vector{T}, scratch::Matrix{T},
    structure::SchurStructure, E_data::Vector{T}, C_data::Vector{T},
    b_e::AbstractVector{T}, scaling_e::AbstractVector{T}, lambda::T, e::Int, de::Int
) where T
    eo = structure.eoffsets[e]
    sc_off = structure.boffsets_e[e]

    M = view(scratch, 1:de, 1:de)
    @inbounds for j in 1:de, i in 1:de
        M[i, j] = E_data[eo + (j - 1) * de + i - 1]
    end
    @inbounds for i in 1:de
        M[i, i] += lambda * scaling_e[sc_off + i - 1]
    end
    # Cholesky; every pivot must be safely positive relative to its (damped) diagonal entry. The damped
    # diagonal entry is recomputed from the inputs.
    _, info = LinearAlgebra.LAPACK.potrf!('U', M)
    info == 0 || return false
    @inbounds for i in 1:de
        hii = E_data[eo + (i - 1) * (de + 1)] + lambda * scaling_e[sc_off + i - 1]
        ratio = abs2(M[i, i]) / hii
        (isfinite(ratio) && ratio > singular_tol(T)) || return false
    end
    LinearAlgebra.LAPACK.potri!('U', M)

    @inbounds for j in 1:de, i in 1:j
        v = M[i, j]
        E_inv_data[eo + (j - 1) * de + i - 1] = v
        E_inv_data[eo + (i - 1) * de + j - 1] = v
    end

    @inbounds for i in 1:de
        acc = zero(T)
        for j in 1:de
            acc += E_inv_data[eo + (j - 1) * de + i - 1] * b_e[sc_off + j - 1]
        end
        z_data[sc_off + i - 1] = acc
    end

    @inbounds for c in structure.eliminated_ptr[e]:(structure.eliminated_ptr[e + 1] - 1)
        co = structure.coffsets[c]
        dr = Int(structure.r_dims[structure.entry_retained[c]])
        for ir in 1:dr
            row_off = co + (ir - 1) * de
            for i in 1:de
                acc = zero(T)
                for j in 1:de
                    acc += E_inv_data[eo + (j - 1) * de + i - 1] * C_data[row_off + j - 1]
                end
                W_data[row_off + i - 1] = acc
            end
        end
    end
    return true
end

"""
    factor_eliminated_group!(ws..., e_chunk, s_de) -> Bool

Phase 1 over a chunk of eliminated blocks of uniform dimension. On failure the lowest failing
eliminated block index is recorded in `failed_block`.
"""
function factor_eliminated_group!(
    E_inv_data::Vector{T}, z_data::Vector{T}, W_data::Vector{T}, scratch::Matrix{T},
    failed_block::Base.Threads.Atomic{Int},
    structure::SchurStructure, E_data::Vector{T}, C_data::Vector{T},
    b_e::AbstractVector{T}, scaling_e::AbstractVector{T}, lambda::T,
    e_chunk::UnitRange{Int}, s_de::Union{StaticInt, Int}
) where T
    @inbounds for e in e_chunk
        failed_block[] != 0 && return false
        if !factor_eliminated!(E_inv_data, z_data, W_data, scratch, structure, E_data, C_data, b_e, scaling_e, lambda, e, s_de)
            old = failed_block[]
            while (old == 0 || e < old)
                old_val = Threads.atomic_cas!(failed_block, old, e)
                old_val == old && break
                old = failed_block[]
            end
            return false
        end
    end
    return true
end

# Contribution of coupling entry c2 (eliminated block e, retained block r2) to g_S and to the column
# block r2 of S: g_S[r2] -= C_er2' z_e and S[r1, r2] -= W_er1' C_er2 for every r1 <= r2 coupled to e.
@inline function accumulate_entry_dense!(
    S::Matrix{T}, g_S::Vector{T}, structure::SchurStructure, C_data::Vector{T}, W_data::Vector{T}, z_data::Vector{T},
    c2::Int, e::Int, s_de::Union{StaticInt, Int}
) where T
    de = dimval(s_de)
    r2 = structure.entry_retained[c2]
    dr2 = Int(structure.r_dims[r2])
    ro2 = structure.boffsets_r[r2]
    co2 = structure.coffsets[c2]
    sc_off = structure.boffsets_e[e]
    @inbounds for ir in 1:dr2
        g_S[ro2 + ir - 1] -= dot_dim(C_data, co2 + (ir - 1) * de, z_data, sc_off, s_de)
    end
    @inbounds for c1 in structure.eliminated_ptr[e]:c2
        r1 = structure.entry_retained[c1]
        dr1 = Int(structure.r_dims[r1])
        ro1 = structure.boffsets_r[r1]
        wo1 = structure.coffsets[c1]
        for col in 1:dr2
            c_off = co2 + (col - 1) * de
            nrow = r1 == r2 ? col : dr1      # diagonal block: upper triangle only
            for row in 1:nrow
                S[ro1 + row - 1, ro2 + col - 1] -= dot_dim(W_data, wo1 + (row - 1) * de, C_data, c_off, s_de)
            end
        end
    end
    return nothing
end

"""
    accumulate_dense_columns!(S, g_S, structure, C_data, W_data, z_data, retained_ptr, retained_entry, entry_eliminated, r_range)

Phase 2 for the retained blocks in `r_range`: write all contributions to their columns of the dense upper
triangle of `S` and to `g_S` directly. Chunks with disjoint `r_range` never write the same entry.
"""
function accumulate_dense_columns!(
    S::Matrix{T}, g_S::Vector{T}, structure::SchurStructure, C_data::Vector{T}, W_data::Vector{T}, z_data::Vector{T},
    retained_ptr::Vector{Int}, retained_entry::Vector{Int}, entry_eliminated::Vector{Int}, r_range::UnitRange{Int}
) where T
    @inbounds for r2 in r_range
        for k in retained_ptr[r2]:(retained_ptr[r2 + 1] - 1)
            c2 = retained_entry[k]
            e = entry_eliminated[c2]
            dispatch_dim(Int(structure.e_dims[e])) do s_de
                accumulate_entry_dense!(S, g_S, structure, C_data, W_data, z_data, c2, e, s_de)
            end
        end
    end
    return nothing
end

@inline function accumulate_entry_sparse!(
    nzval::Vector{T}, g_S::Vector{T}, structure::SchurStructure, C_data::Vector{T}, W_data::Vector{T}, z_data::Vector{T},
    S_colptr::Vector{Ti}, block_colptr::Vector{Ti}, block_rowval::Vector{Ti}, block_row_offsets::Vector{Int}, col_heights::Vector{Int},
    c2::Int, e::Int, s_de::Union{StaticInt, Int}
) where {T, Ti}
    de = dimval(s_de)
    r2 = structure.entry_retained[c2]
    dr2 = Int(structure.r_dims[r2])
    ro2 = structure.boffsets_r[r2]
    co2 = structure.coffsets[c2]
    sc_off = structure.boffsets_e[e]
    @inbounds for ir in 1:dr2
        g_S[ro2 + ir - 1] -= dot_dim(C_data, co2 + (ir - 1) * de, z_data, sc_off, s_de)
    end
    col_base = S_colptr[ro2]
    stride = col_heights[r2]
    b_row_idx = block_colptr[r2]
    b_row_end = block_colptr[r2 + 1] - 1
    @inbounds for c1 in structure.eliminated_ptr[e]:c2
        r1 = structure.entry_retained[c1]
        dr1 = Int(structure.r_dims[r1])
        wo1 = structure.coffsets[c1]
        # monotone merge-walk down the block column
        while b_row_idx <= b_row_end && block_rowval[b_row_idx] < r1
            b_row_idx += 1
        end
        @assert b_row_idx <= b_row_end && block_rowval[b_row_idx] == r1 "Sparse pattern missing entry ($r1, $r2)"
        row_offset = block_row_offsets[b_row_idx]
        for col in 1:dr2
            c_off = co2 + (col - 1) * de
            col_ptr = col_base + (col - 1) * stride + row_offset
            nrow = r1 == r2 ? col : dr1
            for row in 1:nrow
                nzval[col_ptr + row - 1] -= dot_dim(W_data, wo1 + (row - 1) * de, C_data, c_off, s_de)
            end
        end
    end
    return nothing
end

"""
    accumulate_sparse_columns!(nzval, g_S, structure, C_data, W_data, z_data, retained_ptr, retained_entry, entry_eliminated,
                               S_colptr, block_colptr, block_rowval, block_row_offsets, col_heights, r_range)

Sparse counterpart of [`accumulate_dense_columns!`](@ref), writing into `S.nzval`.
"""
function accumulate_sparse_columns!(
    nzval::Vector{T}, g_S::Vector{T}, structure::SchurStructure, C_data::Vector{T}, W_data::Vector{T}, z_data::Vector{T},
    retained_ptr::Vector{Int}, retained_entry::Vector{Int}, entry_eliminated::Vector{Int},
    S_colptr::Vector{Ti}, block_colptr::Vector{Ti}, block_rowval::Vector{Ti},
    block_row_offsets::Vector{Int}, col_heights::Vector{Int}, r_range::UnitRange{Int}
) where {T, Ti}
    @inbounds for r2 in r_range
        for k in retained_ptr[r2]:(retained_ptr[r2 + 1] - 1)
            c2 = retained_entry[k]
            e = entry_eliminated[c2]
            dispatch_dim(Int(structure.e_dims[e])) do s_de
                accumulate_entry_sparse!(nzval, g_S, structure, C_data, W_data, z_data, S_colptr, block_colptr, block_rowval,
                                         block_row_offsets, col_heights, c2, e, s_de)
            end
        end
    end
    return nothing
end

"""
    reset_schur_dense!(S, R, scaling_r, lambda, ws)

Reset dense reduced system S = R + lambda * diag(scaling_r) and the failure flag.
"""
function reset_schur_dense!(
    S::Matrix{T}, R::Matrix{T}, scaling_r::AbstractVector{T},
    lambda::T, ws::SchurReductionWorkspace{T}
) where T
    copyto!(S, R)
    @inbounds for (i, sc) in enumerate(scaling_r)
        S[i, i] += lambda * sc
    end
    ws.failed_block[] = 0
    return nothing
end

"""
    reset_schur_sparse!(nzval, R_nzval, r_to_s_map, diag_inds, scaling_r, lambda, ws)

Reset sparse reduced system S = R + lambda * diag(scaling_r) and the failure flag.
"""
function reset_schur_sparse!(
    nzval::Vector{T}, R_nzval::Vector{T}, r_to_s_map::Vector{Int},
    diag_inds::Vector{Int}, scaling_r::AbstractVector{T},
    lambda::T, ws::SchurReductionWorkspace{T}
) where T
    fill!(nzval, zero(T))
    @inbounds for i in eachindex(r_to_s_map)
        idx = r_to_s_map[i]
        if idx > 0
            nzval[idx] = R_nzval[i]
        end
    end
    @inbounds for (i, idx) in enumerate(diag_inds)
        nzval[idx] += lambda * scaling_r[i]
    end
    ws.failed_block[] = 0
    return nothing
end

# Phase 1 for all groups of eliminated blocks; false if any block failed
function factor_all_eliminated!(
    structure::SchurStructure, E_data::Vector{T}, C_data::Vector{T}, b_e::AbstractVector{T},
    scaling_e::AbstractVector{T}, lambda::T, ws::SchurReductionWorkspace{T}
) where T
    for (g_idx, e_range) in enumerate(structure.e_groups)
        chunks = structure.chunks[g_idx]
        threaded = use_threads(length(chunks), length(structure.entry_retained))
        dispatch_dim(structure.e_sizes[g_idx]) do s_de
            foreach_index(length(chunks), threaded) do k
                factor_eliminated_group!(ws.E_inv_data, ws.z_data, ws.W_data, ws.W_scratch[k], ws.failed_block,
                                        structure, E_data, C_data, b_e, scaling_e, lambda, chunks[k], s_de)
            end
        end
        ws.failed_block[] != 0 && return false
    end
    return true
end

"""
    reduce_schur_dense!(S, g_S, b_r, b_e, scaling_e, lambda, structure, E_data, C_data, ws)

Two-phase Schur reduction into the dense upper triangle of `S` (already holding `R` plus damping) and `g_S`.
"""
function reduce_schur_dense!(
    S::Matrix{T}, g_S::Vector{T}, b_r::AbstractVector{T}, b_e::AbstractVector{T},
    scaling_e::AbstractVector{T}, lambda::T, structure::SchurStructure,
    E_data::Vector{T}, C_data::Vector{T}, ws::SchurReductionWorkspace{T}
) where T
    factor_all_eliminated!(structure, E_data, C_data, b_e, scaling_e, lambda, ws) || return false
    copyto!(g_S, b_r)
    nk = length(ws.col_chunks)
    foreach_index(nk, use_threads(nk, length(structure.entry_retained))) do k
        accumulate_dense_columns!(S, g_S, structure, C_data, ws.W_data, ws.z_data, ws.retained_ptr, ws.retained_entry, ws.entry_eliminated, ws.col_chunks[k])
    end
    return true
end

"""
    reduce_schur_sparse!(S, g_S, b_r, b_e, scaling_e, lambda, structure, E_data, C_data,
                         block_colptr, block_rowval, block_row_offsets, col_heights, ws)

Two-phase Schur reduction into `S.nzval` (already holding `R` plus damping) and `g_S`.
"""
function reduce_schur_sparse!(
    S::SparseMatrixCSC{T, Ti}, g_S::Vector{T}, b_r::AbstractVector{T}, b_e::AbstractVector{T},
    scaling_e::AbstractVector{T}, lambda::T, structure::SchurStructure,
    E_data::Vector{T}, C_data::Vector{T},
    block_colptr::Vector{Ti}, block_rowval::Vector{Ti},
    block_row_offsets::Vector{Int}, col_heights::Vector{Int},
    ws::SchurReductionWorkspace{T}
) where {T, Ti}
    factor_all_eliminated!(structure, E_data, C_data, b_e, scaling_e, lambda, ws) || return false
    copyto!(g_S, b_r)
    nk = length(ws.col_chunks)
    foreach_index(nk, use_threads(nk, length(structure.entry_retained))) do k
        accumulate_sparse_columns!(S.nzval, g_S, structure, C_data, ws.W_data, ws.z_data, ws.retained_ptr, ws.retained_entry, ws.entry_eliminated,
                                   S.colptr, block_colptr, block_rowval, block_row_offsets, col_heights, ws.col_chunks[k])
    end
    return true
end

"""
    backsubstitute!(x_e, x_r, structure, C_data, E_inv_data, z_data)

Recover step Delta x_e for all eliminated blocks independently using cached E_inv and z:
`x_e = z_e - E_e^-1 * (sum_r C_er x_r)` for each eliminated block e. The coupling sum is accumulated first
and `E_inv` is applied once per eliminated block.
"""
function backsubstitute!(
    x_e::AbstractVector{T}, x_r::AbstractVector{T},
    structure::SchurStructure, C_data::Vector{T},
    E_inv_data::Vector{T}, z_data::Vector{T}
) where T
    threaded = use_threads(structure.neblocks, length(structure.entry_retained))
    for (g_idx, e_range) in enumerate(structure.e_groups)
        de = structure.e_sizes[g_idx]
        chunks = structure.chunks[g_idx]
        # Function barrier on the (static) block dimension
        dispatch_dim(de) do s_de
            foreach_index(length(chunks), threaded) do k
                backsubstitute_group!(x_e, x_r, structure, C_data, E_inv_data, z_data, chunks[k], s_de)
            end
        end
    end
    return nothing
end

function backsubstitute_group!(
    x_e::AbstractVector{T}, x_r::AbstractVector{T},
    structure::SchurStructure, C_data::Vector{T},
    E_inv_data::Vector{T}, z_data::Vector{T},
    e_chunk::UnitRange{Int}, s_de::StaticInt{De}
) where {T, De}
    eliminated_ptr = structure.eliminated_ptr
    entry_retained = structure.entry_retained
    coffsets = structure.coffsets
    boffsets_r = structure.boffsets_r
    boffsets_e = structure.boffsets_e
    eoffsets = structure.eoffsets
    r_dims = structure.r_dims

    @inbounds for e in e_chunk
        e_off = boffsets_e[e]

        # t = sum over the retained blocks r coupled to e of C_er * x_r
        t = zero(SVector{De, T})
        for c in eliminated_ptr[e]:(eliminated_ptr[e + 1] - 1)
            r = entry_retained[c]
            dr = Int(r_dims[r])
            ro = boffsets_r[r]
            co = coffsets[c]
            for ir in 1:dr
                t += make_svec(C_data, co + (ir - 1) * De, s_de) * x_r[ro + ir - 1]
            end
        end

        xe = make_svec(z_data, e_off, s_de) - make_smat(E_inv_data, eoffsets[e], s_de) * t
        for i in 1:De
            x_e[e_off + i - 1] = xe[i]
        end
    end
    return nothing
end

# Blocks above D_STATIC dimensions: same computation with run-time loops and one small temporary per call
function backsubstitute_group!(
    x_e::AbstractVector{T}, x_r::AbstractVector{T},
    structure::SchurStructure, C_data::Vector{T},
    E_inv_data::Vector{T}, z_data::Vector{T},
    e_chunk::UnitRange{Int}, de::Int
) where T
    t = Vector{T}(undef, de)
    @inbounds for e in e_chunk
        e_off = structure.boffsets_e[e]
        eo = structure.eoffsets[e]
        fill!(t, zero(T))
        for c in structure.eliminated_ptr[e]:(structure.eliminated_ptr[e + 1] - 1)
            r = structure.entry_retained[c]
            dr = Int(structure.r_dims[r])
            ro = structure.boffsets_r[r]
            co = structure.coffsets[c]
            for ir in 1:dr
                xc = x_r[ro + ir - 1]
                for i in 1:de
                    t[i] += C_data[co + (ir - 1) * de + i - 1] * xc
                end
            end
        end
        for i in 1:de
            acc = z_data[e_off + i - 1]
            for j in 1:de
                acc -= E_inv_data[eo + (j - 1) * de + i - 1] * t[j]
            end
            x_e[e_off + i - 1] = acc
        end
    end
    return nothing
end

"""
    expand_block_perm(block_perm::AbstractVector{<:Integer}, block_sizes::AbstractVector{<:Integer}) -> Vector{Int}

Expand a block-level permutation into a scalar-level permutation vector.
"""
function expand_block_perm(block_perm::AbstractVector{<:Integer}, block_sizes::AbstractVector{<:Integer})
    nblocks = length(block_sizes)
    @assert length(block_perm) == nblocks "block_perm and block_sizes must have equal length"
    total_dim = sum(Int, block_sizes; init=0)
    scalar_perm = Vector{Int}(undef, total_dim)
    boffsets = Vector{Int}(undef, nblocks)
    boffsets[1] = 1
    for i in 2:nblocks
        boffsets[i] = boffsets[i - 1] + Int(block_sizes[i - 1])
    end
    cursor = 1
    for b in block_perm
        bo = boffsets[b]
        bs = Int(block_sizes[b])
        for j in 1:bs
            scalar_perm[cursor] = bo + j - 1
            cursor += 1
        end
    end
    return scalar_perm
end

"""
    sort_schur_blocks(is_eliminated::AbstractVector{Bool}, block_sizes::AbstractVector{<:Integer}, couplings::AbstractVector{<:AbstractVector{<:Integer}}) -> Vector{Int}

Two-pass variable ordering:
- Pass 1: Retained variables (role 0) ordered by dimension descending; eliminated variables (role 1) grouped by dimension descending.
- Pass 2: Within each eliminated dimension group, sort eliminated blocks by their first coupled retained block index r_min(e) = min_{r coupled to e} r.
Returns a block permutation vector mapping old block indices to new positions.
"""
function sort_schur_blocks(
    is_eliminated::AbstractVector{Bool},
    block_sizes::AbstractVector{<:Integer},
    couplings::AbstractVector{<:AbstractVector}
)
    nblocks = length(is_eliminated)
    @assert length(block_sizes) == nblocks "is_eliminated and block_sizes must have equal length"

    # Retained blocks (sorted by dimension descending)
    r_indices = findall(!, is_eliminated)
    sort!(r_indices, by = i -> -Int(block_sizes[i]))

    # Eliminable blocks
    e_indices = findall(identity, is_eliminated)
    # Group by dimension descending, then within dimension by r_min
    sort!(e_indices, by = function(i)
        de = Int(block_sizes[i])
        coupled = couplings[i]
        r_min = isempty(coupled) ? typemax(Int) : minimum(coupled)
        return (-de, r_min)
    end)

    return [r_indices; e_indices]
end

"""
    build_schur_pattern(structure::SchurStructure, [T=Float64]; extra_blocks=()) -> NamedTuple

Compute block-level and scalar-level sparsity patterns of the reduced system S, where two retained blocks are linked if they are coupled to a common eliminated block.
Returns `(block_pattern, block_colptr, block_rowval, block_row_offsets, col_heights, S_sparse, diag_inds)`;
`S_sparse` is a zero-valued `SparseMatrixCSC{T}`.

The pattern contains every pair of retained blocks that share an eliminated block, plus the diagonal.
Retained-retained couplings that do not go through an eliminated block (direct couplings between
retained blocks that share no eliminated block) have no such slot, so list them in `extra_blocks`, an
iterable of `(i, j)` retained block index pairs (in either order; `i == j` is allowed and ignored).
Use [`retained_block_pairs`](@ref) to obtain them from the block pattern of `R`.
"""
function build_schur_pattern(structure::SchurStructure, ::Type{T}=Float64; extra_blocks=()) where T
    nrblocks = structure.nrblocks
    neblocks = structure.neblocks
    r_dims = structure.r_dims
    eliminated_ptr = structure.eliminated_ptr
    entry_retained = structure.entry_retained
    boffsets_r = structure.boffsets_r
    nr = structure.nr

    # 1. Build the block pattern of the reduced system (upper triangle). Collect (I, J) pairs and build the
    # sparse matrix once: inserting entries one at a time into a SparseMatrixCSC is O(nnz^2).
    I_cov = Int[]
    J_cov = Int[]
    for e in 1:neblocks
        c_start = eliminated_ptr[e]
        c_end = eliminated_ptr[e + 1] - 1
        for idx2 in c_start:c_end
            r2 = entry_retained[idx2]
            for idx1 in c_start:idx2
                push!(I_cov, entry_retained[idx1])
                push!(J_cov, r2)
            end
        end
    end
    for (i, j) in extra_blocks
        (1 <= i <= nrblocks && 1 <= j <= nrblocks) || throw(ArgumentError("Extra block pair ($i, $j) is outside the $nrblocks retained blocks"))
        push!(I_cov, min(i, j))
        push!(J_cov, max(i, j))
    end
    for r in 1:nrblocks
        push!(I_cov, r)
        push!(J_cov, r)
    end
    block_pattern = sparse(I_cov, J_cov, true, nrblocks, nrblocks, |)

    block_colptr = Vector{Int}(block_pattern.colptr)
    block_rowval = Vector{Int}(block_pattern.rowval)
    nnz_blocks = nnz(block_pattern)

    # 2. Compute scalar row offsets and column heights
    col_heights = zeros(Int, nrblocks)
    block_row_offsets = zeros(Int, nnz_blocks)
    for r2 in 1:nrblocks
        h = 0
        for idx in block_colptr[r2]:(block_colptr[r2 + 1] - 1)
            r1 = block_rowval[idx]
            block_row_offsets[idx] = h
            h += Int(r_dims[r1])
        end
        col_heights[r2] = h
    end

    # 3. Build scalar SparseMatrixCSC S pattern
    I_list = Int[]
    J_list = Int[]
    for r2 in 1:nrblocks
        for idx in block_colptr[r2]:(block_colptr[r2 + 1] - 1)
            r1 = block_rowval[idx]
            for r_col in 1:r_dims[r2], r_row in 1:r_dims[r1]
                push!(I_list, boffsets_r[r1] + r_row - 1)
                push!(J_list, boffsets_r[r2] + r_col - 1)
            end
        end
    end
    S_sparse = sparse(I_list, J_list, zeros(T, length(I_list)), nr, nr)

    # 4. Compute diagonal indices in S_sparse.nzval
    diag_inds = zeros(Int, nr)
    for j in 1:nr
        for idx in S_sparse.colptr[j]:(S_sparse.colptr[j + 1] - 1)
            if S_sparse.rowval[idx] == j
                diag_inds[j] = idx
                break
            end
        end
        @assert diag_inds[j] != 0 "Missing diagonal entry for retained scalar $j"
    end

    return (
        block_pattern = block_pattern,
        block_colptr = block_colptr,
        block_rowval = block_rowval,
        block_row_offsets = block_row_offsets,
        col_heights = col_heights,
        S_sparse = S_sparse,
        diag_inds = diag_inds
    )
end

"""
    retained_block_pairs(R::SparseBlockLinearSystem) -> Vector{Tuple{Int, Int}}

Off-diagonal stored blocks of `R` as `(row_block, col_block)` pairs. Pass them as `extra_blocks` to
[`build_schur_pattern`](@ref) so the reduced system has a slot for every coupling in `R`.
"""
function retained_block_pairs(R::SparseBlockLinearSystem)
    cols, rows, _ = SparseArrays.findnz(R.indicestransposed)
    return [(Int(brow), Int(bcol)) for (bcol, brow) in zip(cols, rows) if brow != bcol]
end

"""
    build_r_to_s_map(R_sparse::SparseMatrixCSC, S_sparse::SparseMatrixCSC) -> Vector{Int}

Build mapping vector from R_sparse.nzval to S_sparse.nzval.
"""
function build_r_to_s_map(R_sparse::SparseMatrixCSC, S_sparse::SparseMatrixCSC)
    r_to_s = Int[]
    for j in 1:R_sparse.n
        for k in R_sparse.colptr[j]:(R_sparse.colptr[j + 1] - 1)
            i = R_sparse.rowval[k]
            s_k = find_in_column(S_sparse, i, j)
            @assert s_k != 0 "R entry ($i, $j) not present in S sparsity pattern"
            push!(r_to_s, s_k)
        end
    end
    return r_to_s
end

# Index of `row` in column `j` of the sorted-pattern CSC matrix `S` (0 if absent), by binary search
@inline function find_in_column(S::SparseMatrixCSC, row::Integer, j::Integer)
    lo = Int(S.colptr[j])
    hi = Int(S.colptr[j + 1]) - 1
    idx = lo + searchsortedfirst(view(S.rowval, lo:hi), row) - 1
    return (idx <= hi && S.rowval[idx] == row) ? idx : 0
end

"""
    build_r_to_s_map(R_sys::SparseBlockLinearSystem, S_sparse::SparseMatrixCSC, boffsets_r::Vector{Int}) -> Vector{Int}

Build mapping vector from R_sys.data to S_sparse.nzval for symmetric upper-triangle reduction.
Entries of `R_sys.data` that are not mapped (the strict lower triangle of diagonal blocks) are 0.
"""
function build_r_to_s_map(R_sys::SparseBlockLinearSystem, S_sparse::SparseMatrixCSC, boffsets_r::Vector{Int})
    r_to_s = zeros(Int, length(R_sys.data))
    for (bcol, brow, data_start) in zip(SparseArrays.findnz(R_sys.indicestransposed)...)
        drow = Int(R_sys.rowblocksizes[brow])
        dcol = Int(R_sys.columnblocksizes[bcol])

        if brow == bcol
            for col in 1:dcol
                scol = boffsets_r[brow] + col - 1
                for row in 1:col
                    srow = boffsets_r[brow] + row - 1
                    s_k = find_in_column(S_sparse, srow, scol)
                    @assert s_k != 0 "Missing diagonal entry ($srow, $scol) in S"
                    r_to_s[data_start + (col - 1)*drow + row - 1] = s_k
                end
            end
        elseif brow > bcol
            for col in 1:dcol
                srow = boffsets_r[bcol] + col - 1
                for row in 1:drow
                    scol = boffsets_r[brow] + row - 1
                    s_k = find_in_column(S_sparse, srow, scol)
                    @assert s_k != 0 "Missing entry ($srow, $scol) in S"
                    r_to_s[data_start + (col - 1)*drow + row - 1] = s_k
                end
            end
        end
    end
    return r_to_s
end

"""
    build_r_to_s_map(R_sys::SparseBlockLinearSystem, pattern, boffsets_r::Vector{Int}) -> Vector{Int}

As above, but using the block-level pattern in `pattern` (the result of [`build_schur_pattern`](@ref)):
one block-level search per stored block, then the scalar position follows from `block_row_offsets` and
`col_heights` with integer arithmetic.
"""
function build_r_to_s_map(R_sys::SparseBlockLinearSystem, pattern::NamedTuple, boffsets_r::Vector{Int})
    S_colptr = pattern.S_sparse.colptr
    r_to_s = zeros(Int, length(R_sys.data))
    for (bcol, brow, data_start) in zip(SparseArrays.findnz(R_sys.indicestransposed)...)
        brow >= bcol || continue   # only lower-triangle blocks are stored in R
        drow = Int(R_sys.rowblocksizes[brow])
        dcol = Int(R_sys.columnblocksizes[bcol])
        # Block (bcol, brow) of the upper-triangle pattern: block column brow, block row bcol
        rng = pattern.block_colptr[brow]:(pattern.block_colptr[brow + 1] - 1)
        k = searchsortedfirst(view(pattern.block_rowval, rng), bcol)
        (k <= length(rng) && pattern.block_rowval[rng[k]] == bcol) || throw(AssertionError("Block ($bcol, $brow) of R is not present in the pattern of S"))
        row_offset = pattern.block_row_offsets[rng[k]]
        base = S_colptr[boffsets_r[brow]] + row_offset
        stride = pattern.col_heights[brow]
        if brow == bcol
            for col in 1:dcol, row in 1:col
                r_to_s[data_start + (col - 1)*drow + row - 1] = base + (col - 1) * stride + row - 1
            end
        else
            for col in 1:dcol, row in 1:drow
                r_to_s[data_start + (col - 1)*drow + row - 1] = base + (row - 1) * stride + col - 1
            end
        end
    end
    return r_to_s
end

"""
    fast_bAb(H::SchurBlockMatrix, dx::AbstractVector) -> Real

Compute quadratic form dx' * H * dx in O(N) operations without materializing the full Hessian.
`dx` may have any element type; the result has the promoted type.

The retained part (`R`) is walked block by block. The eliminated part (`E` and the couplings `C`) is where
nearly all the work is, and is processed per group of eliminated blocks of uniform dimension with that
dimension known statically (3.4x faster than per-block run-time sizes on a bundle-adjustment-like
benchmark with 200 retained and 20000 eliminated blocks). Specialising the retained part by run of retained block sizes was measured too: it saved nothing
measurable (that part takes ~0.01 ms) and cost 0.2-2 s of extra compilation, so it is not done.
"""
function fast_bAb(H::SchurBlockMatrix{T}, dx::AbstractVector) where T
    Tacc = promote_type(T, eltype(dx))
    structure = H.structure
    nr = structure.nr
    dx_r = view(dx, 1:nr)
    dx_e = view(dx, nr+1:length(dx))
    
    # 1. dx_r' * R * dx_r
    total = zero(Tacc)
    if H.R isa Matrix{T}
        R = H.R
        @inbounds for j in 1:nr
            val = zero(T)
            for i in 1:j
                r_val = R[i, j]
                if i == j
                    val += r_val * dx_r[i]
                else
                    val += r_val * dx_r[i]
                    total += r_val * dx_r[j] * dx_r[i]
                end
            end
            total += dx_r[j] * val
        end
    elseif H.R isa SparseBlockLinearSystem{T}
        # Walk the stored block pattern directly (column i of indicestransposed lists the stored
        # column-blocks j of block row i); each block is column-major at data[ptr:ptr+drow*dcol-1].
        R_sys = H.R
        boffsets = structure.boffsets_r
        data = R_sys.data
        colptr = R_sys.indicestransposed.colptr
        rowval = R_sys.indicestransposed.rowval
        nzval = R_sys.indicestransposed.nzval
        @inbounds for i in 1:length(R_sys.rowblocksizes)
            row_start = boffsets[i]
            drow = R_sys.rowblocksizes[i]
            for k in colptr[i]:(colptr[i + 1] - 1)
                j = rowval[k]
                ptr = nzval[k]
                col_start = boffsets[j]
                dcol = R_sys.columnblocksizes[j]
                blk_total = zero(T)
                for jc in 1:dcol
                    dx_jc = dx_r[col_start + jc - 1]
                    col_ptr = ptr + (jc - 1) * drow - 1
                    col_total = zero(T)
                    for ir in 1:drow
                        col_total += data[col_ptr + ir] * dx_r[row_start + ir - 1]
                    end
                    blk_total += col_total * dx_jc
                end
                total += (i == j ? blk_total : 2 * blk_total)
            end
        end
    end

    # 2. 2 * dx_e' * C * dx_r and 3. dx_e' * E * dx_e, per group of eliminated blocks of uniform dimension.
    # (`z`, a typed zero, is captured by the closure rather than the type `Tacc`: Julia 1.10 widens a captured
    # type to `DataType`, which made the result type inferred as `Any` and boxed it on every call.)
    z = zero(Tacc)
    for (g_idx, e_range) in enumerate(structure.e_groups)
        total += dispatch_dim(structure.e_sizes[g_idx]) do s_de
            fast_bAb_eliminated(H, dx_r, dx_e, e_range, s_de, z)
        end
    end

    return total
end

# C and E contributions of the eliminated blocks in `e_range`, all of dimension `s_de` (static up to D_STATIC)
function fast_bAb_eliminated(H::SchurBlockMatrix{T}, dx_r, dx_e, e_range::UnitRange{Int}, s_de::StaticInt, z::Tacc) where {T, Tacc}
    structure = H.structure
    D = dimval(s_de)
    part = z
    @inbounds for e in e_range
        e_off = structure.boffsets_e[e]
        xp = make_svec(dx_e, e_off, s_de)
        part += dot(xp, make_smat(H.E_data, structure.eoffsets[e], s_de) * xp)
        for c in structure.eliminated_ptr[e]:(structure.eliminated_ptr[e + 1] - 1)
            r = structure.entry_retained[c]
            dr = Int(structure.r_dims[r])
            ro = structure.boffsets_r[r]
            co = structure.coffsets[c]
            for ir in 1:dr
                part += 2 * dx_r[ro + ir - 1] * dot(make_svec(H.C_data, co + (ir - 1) * D, s_de), xp)
            end
        end
    end
    return part
end

# Blocks above D_STATIC dimensions: the same computation with run-time loops
function fast_bAb_eliminated(H::SchurBlockMatrix{T}, dx_r, dx_e, e_range::UnitRange{Int}, de::Int, z::Tacc) where {T, Tacc}
    structure = H.structure
    part = z
    @inbounds for e in e_range
        e_off = structure.boffsets_e[e]
        eo = structure.eoffsets[e]
        for col in 1:de
            xcol = dx_e[e_off + col - 1]
            for row in 1:de
                part += H.E_data[eo + (col - 1) * de + row - 1] * xcol * dx_e[e_off + row - 1]
            end
        end
        for c in structure.eliminated_ptr[e]:(structure.eliminated_ptr[e + 1] - 1)
            r = structure.entry_retained[c]
            dr = Int(structure.r_dims[r])
            ro = structure.boffsets_r[r]
            co = structure.coffsets[c]
            for ir in 1:dr
                acc = zero(Tacc)
                for i in 1:de
                    acc += H.C_data[co + (ir - 1) * de + i - 1] * dx_e[e_off + i - 1]
                end
                part += 2 * dx_r[ro + ir - 1] * acc
            end
        end
    end
    return part
end
