using Static, StaticArrays

# Block sizes are *stored* compactly as `T` (normally `UInt8`, hence the 1-255 limit), but every accessor
# returns `Int` so that products such as `rows * cols` can never wrap around in the storage type.
struct BlockRLE{T<:Integer, L<:Integer} <: AbstractVector{Int}
    runlengths::Vector{L}
    blocksizes::Vector{T}
    cumblocks::Vector{L}
    cummatrixdims::Vector{Int}
    flat::Vector{T}          # per-block sizes, one entry per block: O(1) getindex (and checked block access)

    # Unchecked: the runs are already validated, merged and free of zero lengths. Every public route checks first.
    function BlockRLE{T, L}(::Val{:unchecked}, merged_rl::Vector{L}, merged_bs::Vector{T}) where {T<:Integer, L<:Integer}
        cumblocks = Vector{L}(undef, length(merged_rl) + 1)
        cummatrixdims = Vector{Int}(undef, length(merged_rl) + 1)
        cumblocks[1] = one(L)
        cummatrixdims[1] = 1

        for i in eachindex(merged_rl)
            cumblocks[i + 1] = cumblocks[i] + merged_rl[i]
            cummatrixdims[i + 1] = cummatrixdims[i] + Int(merged_rl[i]) * Int(merged_bs[i])
        end

        flat = Vector{T}(undef, Int(cumblocks[end]) - 1)
        for i in eachindex(merged_rl)
            fill!(view(flat, Int(cumblocks[i]):Int(cumblocks[i + 1]) - 1), merged_bs[i])
        end

        return new{T, L}(merged_rl, merged_bs, cumblocks, cummatrixdims, flat)
    end
end

# Largest supported block dimension
const MAX_BLOCK_DIM = 255

# Whether d is a valid block dimension: the one comparison against the limit
@inline valid_block_dim(d::Integer) = 1 <= d <= MAX_BLOCK_DIM

# Throw unless d is a valid block dimension. `i` is the index the caller passed: of a block, or (run = true) of a run.
function check_block_dim(d::Integer, i::Integer; run::Bool=false)
    valid_block_dim(d) && return nothing
    what = run ? "Run $i has block dimension $d" : "Block $i has dimension $d"
    throw(ArgumentError("$what; dimensions must be between 1 and $MAX_BLOCK_DIM"))
end

# From runs: the one checked route for run-length input. Each size is checked before it is narrowed to T, zero-length
# runs are skipped (their sizes unchecked), and adjacent runs of the same size are merged.
function BlockRLE{T, L}(runlengths::AbstractVector{<:Integer}, blocksizes::AbstractVector{<:Integer}) where {T<:Integer, L<:Integer}
    length(runlengths) == length(blocksizes) || throw(DimensionMismatch("runlengths and blocksizes must have identical length"))
    merged_rl = L[]
    merged_bs = T[]
    for (k, rl, bs) in zip(1:length(runlengths), runlengths, blocksizes)
        rl < 0 && throw(ArgumentError("Run lengths must be non-negative, got $rl"))
        rl == 0 && continue
        check_block_dim(bs, k; run=true)
        if !isempty(merged_bs) && merged_bs[end] == bs
            merged_rl[end] += rl
        else
            push!(merged_rl, rl)
            push!(merged_bs, bs)
        end
    end
    return BlockRLE{T, L}(Val(:unchecked), merged_rl, merged_bs)
end

# Run-length encode block sizes that have already been checked
function encode_block_sizes(::Type{T}, vec::AbstractVector{<:Integer}) where T<:Integer
    rl = Int[]
    bs = T[]
    for d in vec
        if !isempty(bs) && bs[end] == d
            rl[end] += 1
        else
            push!(rl, 1)
            push!(bs, d)
        end
    end
    return BlockRLE{T, Int}(Val(:unchecked), rl, bs)
end

# Outer constructors
BlockRLE{T}(runlengths::AbstractVector{<:Integer}, blocksizes::AbstractVector{<:Integer}) where T<:Integer =
    BlockRLE{T, Int}(runlengths, blocksizes)

function BlockRLE(runlengths::AbstractVector{<:Integer}, blocksizes::AbstractVector{<:Integer})
    return BlockRLE{UInt8}(runlengths, blocksizes)
end

function BlockRLE(runs::AbstractVector{<:Pair{<:Integer, <:Integer}})
    rl = [first(p) for p in runs]
    bs = [last(p) for p in runs]
    return BlockRLE(rl, bs)
end

function BlockRLE(runs::AbstractVector{<:Tuple{<:Integer, <:Integer}})
    rl = [first(t) for t in runs]
    bs = [last(t) for t in runs]
    return BlockRLE(rl, bs)
end

"""
    BlockRLE(vec::AbstractVector{<:Integer})
    BlockRLE{T}(vec::AbstractVector{<:Integer}) where T<:Integer

Constructs a `BlockRLE` by run-length encoding a flat vector of block sizes.
"""
function BlockRLE{T}(vec::AbstractVector{<:Integer}) where T<:Integer
    for (i, d) in enumerate(vec)
        check_block_dim(d, i)
    end
    return encode_block_sizes(T, vec)
end

BlockRLE(vec::AbstractVector{<:Integer}) = BlockRLE{UInt8}(vec)
BlockRLE(rle::BlockRLE) = rle
BlockRLE{T}(rle::BlockRLE{T}) where T<:Integer = rle
BlockRLE{T}(rle::BlockRLE{T2, L}) where {T<:Integer, T2<:Integer, L} = BlockRLE{T, L}(rle.runlengths, rle.blocksizes)
BlockRLE{T, L}(rle::BlockRLE{T, L}) where {T<:Integer, L<:Integer} = rle
BlockRLE{T, L}(rle::BlockRLE) where {T<:Integer, L<:Integer} = BlockRLE{T, L}(rle.runlengths, rle.blocksizes)

# Base.AbstractVector Interface
Base.size(rle::BlockRLE) = (Int(rle.cumblocks[end] - 1),)
Base.length(rle::BlockRLE) = Int(rle.cumblocks[end] - 1)
Base.IndexStyle(::Type{<:BlockRLE}) = IndexLinear()

@inline function Base.getindex(rle::BlockRLE, i::Int)
    @boundscheck checkbounds(rle, i)
    return @inbounds Int(rle.flat[i])
end

# Fast reductions
Base.sum(rle::BlockRLE) = rle.cummatrixdims[end] - 1

Base.copy(rle::BlockRLE{T, L}) where {T, L} = BlockRLE{T, L}(copy(rle.runlengths), copy(rle.blocksizes))

function Base.collect(rle::BlockRLE)
    out = Vector{Int}(undef, length(rle))
    idx = 1
    for k in 1:nruns(rle)
        sz = Int(rle.blocksizes[k])
        len = rle.runlengths[k]
        fill!(view(out, idx:(idx + len - 1)), sz)
        idx += len
    end
    return out
end

Base.Vector(rle::BlockRLE) = collect(rle)
Base.Vector{T}(rle::BlockRLE) where T = convert(Vector{T}, collect(rle))
Base.convert(::Type{Vector{T}}, rle::BlockRLE) where T = convert(Vector{T}, collect(rle))
Base.convert(::Type{Vector}, rle::BlockRLE) = collect(rle)
Base.convert(::Type{BlockRLE}, v::AbstractVector{<:Integer}) = BlockRLE(v)
Base.convert(::Type{BlockRLE{T}}, v::AbstractVector{<:Integer}) where T = BlockRLE{T}(v)

function Base.:(==)(a::BlockRLE, b::BlockRLE)
    if length(a) != length(b) || matrixdimension(a) != matrixdimension(b)
        return false
    end
    if a.runlengths == b.runlengths && a.blocksizes == b.blocksizes
        return true
    end
    return invoke(==, Tuple{AbstractVector, AbstractVector}, a, b)
end

# RLE Specific API
nruns(rle::BlockRLE) = length(rle.runlengths)
runlength(rle::BlockRLE, k::Integer) = rle.runlengths[k]
runblocksize(rle::BlockRLE, k::Integer) = Int(rle.blocksizes[k])
matrixdimension(rle::BlockRLE) = rle.cummatrixdims[end] - 1

@inline function runblockrange(rle::BlockRLE, k::Integer)
    @boundscheck (1 <= k <= nruns(rle)) || throw(BoundsError(rle.runlengths, k))
    return @inbounds Int(rle.cumblocks[k]):Int(rle.cumblocks[k+1] - 1)
end

@inline function runmatrixrange(rle::BlockRLE, k::Integer)
    @boundscheck (1 <= k <= nruns(rle)) || throw(BoundsError(rle.runlengths, k))
    return @inbounds rle.cummatrixdims[k]:(rle.cummatrixdims[k+1] - 1)
end

function computestartindices(rle::BlockRLE)
    n = length(rle)
    startind = Vector{Int}(undef, n)
    idx = 1
    curr = 1
    for k in 1:nruns(rle)
        sz = Int(rle.blocksizes[k])
        len = rle.runlengths[k]
        for _ in 1:len
            startind[idx] = curr
            curr += sz
            idx += 1
        end
    end
    return startind
end

function eachrun(rle::BlockRLE)
    return (
        (run = k,
         blockrange = runblockrange(rle, k),
         blocksize = Int(rle.blocksizes[k]),
         matrixrange = runmatrixrange(rle, k))
        for k in 1:nruns(rle)
    )
end

# Coercion helper ensuring BlockRLE{UInt8, Int} representation
as_block_rle(v::BlockRLE{UInt8, Int}) = v
as_block_rle(v::BlockRLE) = BlockRLE{UInt8, Int}(v)
as_block_rle(v::AbstractVector{<:Integer}) = BlockRLE(v)
const as_blocksizes = as_block_rle
