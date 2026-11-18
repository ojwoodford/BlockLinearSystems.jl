"""
    SchurBlockHessian{T, RType}(structure::SchurStructure; pattern=nothing)
    SchurBlockHessian(R, like::SchurBlockHessian)
    SchurBlockHessian(structure, R, E_data, C_data; pattern=nothing)

A Hessian (or any symmetric matrix) laid out for elimination by `structure`: the retained part `R`, the
eliminated blocks `E_data` and the coupling blocks `C_data`.

`SchurBlockHessian{T, RType}(structure; pattern)` allocates the storage and zeroes it. `RType = Matrix{T}` gives
a dense retained part; `RType = SparseBlockLinearSystem{T}` a block-sparse one, on the block pattern of the
reduced system (the pairs of retained blocks that share an eliminated block or a row block of the description),
taken from `pattern` if given, otherwise built and kept in the `pattern` field. Any other `RType` is an
`ArgumentError`.

`SchurBlockHessian(R, like)` is a system on the retained storage `R` that shares `like`'s `E_data`, `C_data`,
structure and pattern, and allocates nothing (for per-thread accumulation of the retained part). `R` must be
of the same kind and size as `like.R` (`DimensionMismatch` otherwise).

**Fields.** `R`, `E_data`, `C_data`, `structure` and `pattern` may be read for operations on a whole part:
zero it, add another system's part of the same structure to it, copy it, compare structures by identity.
Where a block lives inside them is not public: fill the system with [`addblock!`](@ref).

`pattern` optionally carries the result of [`build_schur_pattern`](@ref) for this structure, so that
code which receives only the matrix can build a reduced system without recomputing it. `nothing` (the
default) means "build it when needed". The pattern must come from
`build_schur_pattern(structure; extra_blocks=retained_block_pairs(R))`, or from a superset of those
pairs; only its size is checked (`size(pattern.S_sparse, 1) == structure.nr`, else `ArgumentError`).

The pattern is shared between every system built from `H`, including its [`copy_structure`](@ref)
copies, so treat it as read-only structure: in particular, copy `pattern.S_sparse` before using it as
value storage. Callers may attach extra entries to the named tuple (for example
`merge(pattern, (S_factor = fac,))`); this package ignores entries it does not know.

The positional form `SchurBlockHessian(R, E_data, C_data, structure)` (field order) is also accepted.
"""
struct SchurBlockHessian{T, RType}
    R::RType                         # Matrix{T} (dense) or SparseBlockLinearSystem (sparse)
    E_data::Vector{T}                # Contiguous storage for block-diagonal E (each E_e column-major): sum_e d_e^2
    C_data::Vector{T}                # Contiguous storage for block-sparse C (each C_er column-major, d_e x d_r): sum_c (d_e * d_r)
    structure::SchurStructure
    # Deliberately abstractly typed, and not a type parameter: a third parameter would make the field
    # declarations of downstream code abstract and its hot paths type-unstable. This is read only when a
    # linear system is constructed (a function barrier); assembly, reduction and `dot` never touch it.
    pattern::Union{Nothing, NamedTuple}
end

function SchurBlockHessian(structure::SchurStructure, R::RType, E_data::Vector{T}, C_data::Vector{T};
                          pattern::Union{Nothing, NamedTuple}=nothing) where {T, RType}
    if pattern !== nothing && size(pattern.S_sparse, 1) != structure.nr
        throw(ArgumentError("pattern.S_sparse has size $(size(pattern.S_sparse, 1)), expected structure.nr = $(structure.nr)"))
    end
    return SchurBlockHessian{T, RType}(R, E_data, C_data, structure, pattern)
end

SchurBlockHessian{T, RType}(R, E_data, C_data, structure) where {T, RType} =
    SchurBlockHessian{T, RType}(R, E_data, C_data, structure, nothing)
SchurBlockHessian(R::RType, E_data::Vector{T}, C_data::Vector{T}, structure::SchurStructure) where {T, RType} =
    SchurBlockHessian{T, RType}(R, E_data, C_data, structure, nothing)

# Storage of the eliminated blocks and of the coupling blocks of a structure
eliminated_storage_length(st::SchurStructure) = sum(d -> Int(d)^2, st.e_dims; init=0)
function coupling_storage_length(st::SchurStructure)
    n = 0
    for e in 1:st.neblocks, c in st.eliminated_ptr[e]:st.eliminated_ptr[e + 1] - 1
        n += Int(st.e_dims[e]) * Int(st.r_dims[st.entry_retained[c]])
    end
    return n
end

function SchurBlockHessian{T, RType}(structure::SchurStructure; pattern::Union{Nothing, NamedTuple}=nothing) where {T, RType}
    (RType === Matrix{T} || RType === SparseBlockLinearSystem{T}) ||
        throw(ArgumentError("Retained storage $RType is not supported; use Matrix{$T} or SparseBlockLinearSystem{$T}"))
    if RType === Matrix{T}
        R = zeros(T, structure.nr, structure.nr)
    else
        pattern === nothing && (pattern = build_schur_pattern(structure, T))
        R = SparseBlockLinearSystem{T}(TransposedPattern(pattern.block_pattern), structure.r_dims, structure.r_dims)
    end
    E_data = zeros(T, eliminated_storage_length(structure))
    C_data = zeros(T, coupling_storage_length(structure))
    return SchurBlockHessian(structure, R, E_data, C_data; pattern)
end

function SchurBlockHessian(R, like::SchurBlockHessian{T, RType}) where {T, RType}
    R isa RType || throw_retained_mismatch(typeof(R), RType)
    same_retained_layout(R, like.R) || throw_retained_mismatch(R, like.R)
    return SchurBlockHessian{T, RType}(R, like.E_data, like.C_data, like.structure, like.pattern)
end

same_retained_layout(R::Matrix, L::Matrix) = size(R) == size(L)
same_retained_layout(R::SparseBlockLinearSystem, L::SparseBlockLinearSystem) =
    length(R.data) == length(L.data) && (R.indicestransposed === L.indicestransposed ||
        (R.rowblocksizes == L.rowblocksizes && R.indicestransposed == L.indicestransposed))
@noinline throw_retained_mismatch(::Type{A}, ::Type{B}) where {A, B} =
    throw(DimensionMismatch("The retained storage is a $A, but the system's is a $B"))
@noinline throw_retained_mismatch(R::Matrix, L::Matrix) =
    throw(DimensionMismatch("The retained storage has size $(size(R)), but the system's has size $(size(L))"))
@noinline throw_retained_mismatch(R, L) =
    throw(DimensionMismatch("The retained storage has another block layout than the system's"))

function zero!(H::SchurBlockHessian)
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
    copy_structure(H::SchurBlockHessian)

Return a new `SchurBlockHessian` with the same structure but freshly allocated, zeroed storage for
`R`, `E_data` and `C_data`. Nothing is shared with `H` except the immutable `SchurStructure` and the
(read-only) `pattern`.
"""
function copy_structure(H::SchurBlockHessian{T}) where T
    R = H.R isa AbstractMatrix ? zeros(T, size(H.R)) : copy_structure(H.R)
    return SchurBlockHessian(H.structure, R, zeros(T, length(H.E_data)), zeros(T, length(H.C_data)); pattern=H.pattern)
end
