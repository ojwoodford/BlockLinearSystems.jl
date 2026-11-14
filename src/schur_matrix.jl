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
    # linear system is constructed (a function barrier); assembly, reduction and `dot` never touch it.
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
