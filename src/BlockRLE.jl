using Static, StaticArrays

struct BlockRLE{T<:Integer, L<:Integer} <: AbstractVector{T}
    runlengths::Vector{L}
    blocksizes::Vector{T}
    cumblocks::Vector{L}
    cummatrixdims::Vector{Int}

    function BlockRLE{T, L}(runlengths::Vector{L}, blocksizes::Vector{T}) where {T<:Integer, L<:Integer}
        length(runlengths) == length(blocksizes) || throw(DimensionMismatch("runlengths and blocksizes must have identical length"))
        
        # Merge adjacent runs of identical block size and filter zero-length runs
        merged_rl = L[]
        merged_bs = T[]
        for i in eachindex(runlengths)
            rl = runlengths[i]
            bs = blocksizes[i]
            rl < 0 && throw(ArgumentError("Run lengths must be non-negative, got $rl"))
            rl == 0 && continue
            (bs > 0 && bs <= 255) || throw(ArgumentError("Block dimensions must be between 1 and 255, got $bs"))
            if !isempty(merged_bs) && merged_bs[end] == bs
                merged_rl[end] += rl
            else
                push!(merged_rl, rl)
                push!(merged_bs, bs)
            end
        end

        cumblocks = Vector{L}(undef, length(merged_rl) + 1)
        cummatrixdims = Vector{Int}(undef, length(merged_rl) + 1)
        cumblocks[1] = one(L)
        cummatrixdims[1] = 1

        for i in eachindex(merged_rl)
            cumblocks[i + 1] = cumblocks[i] + merged_rl[i]
            cummatrixdims[i + 1] = cummatrixdims[i] + Int(merged_rl[i]) * Int(merged_bs[i])
        end

        return new{T, L}(merged_rl, merged_bs, cumblocks, cummatrixdims)
    end
end

# Outer constructors
function BlockRLE{T}(runlengths::AbstractVector{<:Integer}, blocksizes::AbstractVector{<:Integer}) where T<:Integer
    all(b -> 0 < b <= 255, blocksizes) || throw(ArgumentError("Block dimensions must be between 1 and 255"))
    rl = convert(Vector{Int}, runlengths)
    bs = convert(Vector{T}, blocksizes)
    return BlockRLE{T, Int}(rl, bs)
end

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
    all(b -> 0 < b <= 255, vec) || throw(ArgumentError("Block dimensions must be between 1 and 255"))
    isempty(vec) && return BlockRLE{T, Int}(Int[], T[])
    rl = Int[]
    bs = T[]
    current_size = convert(T, vec[1])
    current_len = 1
    for i in 2:length(vec)
        val = convert(T, vec[i])
        if val == current_size
            current_len += 1
        else
            push!(rl, current_len)
            push!(bs, current_size)
            current_size = val
            current_len = 1
        end
    end
    push!(rl, current_len)
    push!(bs, current_size)
    return BlockRLE{T, Int}(rl, bs)
end

BlockRLE(vec::AbstractVector{<:Integer}) = BlockRLE{UInt8}(vec)
BlockRLE(rle::BlockRLE) = rle
BlockRLE{T}(rle::BlockRLE{T}) where T<:Integer = rle
BlockRLE{T}(rle::BlockRLE{T2, L}) where {T<:Integer, T2<:Integer, L} = BlockRLE{T, L}(copy(rle.runlengths), convert(Vector{T}, rle.blocksizes))

# Base.AbstractVector Interface
Base.size(rle::BlockRLE) = (Int(rle.cumblocks[end] - 1),)
Base.length(rle::BlockRLE) = Int(rle.cumblocks[end] - 1)
Base.IndexStyle(::Type{<:BlockRLE}) = IndexLinear()

@inline function Base.getindex(rle::BlockRLE, i::Int)
    @boundscheck checkbounds(rle, i)
    # Binary search to find run index
    run_idx = searchsortedlast(rle.cumblocks, i)
    return @inbounds rle.blocksizes[run_idx]
end

# Fast reductions
Base.sum(rle::BlockRLE) = rle.cummatrixdims[end] - 1

Base.copy(rle::BlockRLE{T, L}) where {T, L} = BlockRLE{T, L}(copy(rle.runlengths), copy(rle.blocksizes))

function Base.collect(rle::BlockRLE{T}) where T
    out = Vector{T}(undef, length(rle))
    idx = 1
    for k in 1:nruns(rle)
        sz = rle.blocksizes[k]
        len = rle.runlengths[k]
        fill!(view(out, idx:(idx + len - 1)), sz)
        idx += len
    end
    return out
end

Base.Vector(rle::BlockRLE{T}) where T = collect(rle)
Base.Vector{T}(rle::BlockRLE) where T = convert(Vector{T}, collect(rle))
Base.convert(::Type{Vector{T}}, rle::BlockRLE{T}) where T = collect(rle)
Base.convert(::Type{Vector{T}}, rle::BlockRLE) where T = convert(Vector{T}, collect(rle))
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
runblocksize(rle::BlockRLE, k::Integer) = rle.blocksizes[k]
matrixdimension(rle::BlockRLE) = rle.cummatrixdims[end] - 1

@inline function runblockrange(rle::BlockRLE, k::Integer)
    @boundscheck (1 <= k <= nruns(rle)) || throw(BoundsError(rle.runlengths, k))
    return @inbounds Int(rle.cumblocks[k]):Int(rle.cumblocks[k+1] - 1)
end

@inline function runmatrixrange(rle::BlockRLE, k::Integer)
    @boundscheck (1 <= k <= nruns(rle)) || throw(BoundsError(rle.runlengths, k))
    return @inbounds rle.cummatrixdims[k]:(rle.cummatrixdims[k+1] - 1)
end

@inline function blockrange(rle::BlockRLE, i::Integer)
    @boundscheck (1 <= i <= length(rle)) || throw(BoundsError(rle, i))
    run_idx = searchsortedlast(rle.cumblocks, i)
    sz = Int(@inbounds rle.blocksizes[run_idx])
    bstart = @inbounds rle.cummatrixdims[run_idx] + (i - @inbounds rle.cumblocks[run_idx]) * sz
    return bstart:(bstart + sz - 1)
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
         blocksize = rle.blocksizes[k],
         matrixrange = runmatrixrange(rle, k))
        for k in 1:nruns(rle)
    )
end

function sort_blocks_hierarchical(roles::AbstractVector, dimensions::AbstractVector{<:Integer}; rev_dim::Bool=true)
    length(roles) == length(dimensions) || throw(DimensionMismatch("roles and dimensions must have equal length"))
    n = length(roles)
    perm = sortperm(1:n, by = i -> (roles[i], rev_dim ? -Int(dimensions[i]) : Int(dimensions[i])))
    sorted_dims = dimensions[perm]
    rle = BlockRLE(sorted_dims)
    return perm, rle
end

# Internal coercion helper
as_blocksizes(v::BlockRLE{UInt8}) = v
as_blocksizes(v::BlockRLE) = BlockRLE{UInt8}(v)
function as_blocksizes(v::AbstractVector{<:Integer})
    all(i -> 0 < i <= 255, v) || throw(ArgumentError("Block dimensions must be between 1 and 255"))
    return v isa AbstractVector{UInt8} ? v : convert(Vector{UInt8}, v)
end
