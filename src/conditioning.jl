# Conditioning of the eliminated systems over every variable, before the elimination (single-precision.md, decision 13;
# SP-B2). With S = Diagonal(s) over all the variables in the system's order, retained and eliminated alike, the
# reduction, the reduced solve and the back-substitution run on the scaled system (S H S, S b on the Hessian side; the
# columns of J scaled, J S, on the Jacobian side, with the damping in the scaled variables) and x = S y is returned for
# every variable. The scale is applied where the kernels read the stored values (each scaled entry formed in Float64 and
# narrowed once), so H, J, b and r stay exactly as filled, the queries read them unscaled, and nothing is allocated or
# copied for it. Scaling by powers of two is exact, so with no overflow or underflow the result is that of the
# unconditioned elimination, scaled; what it changes is the range of the entries (in Float32) and the decisions made by
# comparing magnitudes: the reduced solvers' pivoting and their rank and indefiniteness tests, and the sparse reduced
# systems, which DampedLinearSolve does not condition (the dense reduced systems are DampedLinearSolve systems and are
# conditioned there as well, on the reduced matrix).
#
# The scale is DampedLinearSolve's rule (decision 13, with SP-D9's recomputation): s_j is the power of two nearest, on a
# log scale, to 1/(1e-12 + √|d_j + λ₀·D_jj|), d_j the diagonal as filled (on the Jacobian side the squared column norms
# of J) and λ₀·D the damping of the first solve after a fill, reused for later dampings of the same fill and recomputed
# when, for some variable, |d_j + λ·D_jj|·s_j² lies outside [2^-16, 2^16]. DampedLinearSolve keeps this rule internal
# (rules.md 12), so it is restated here with the same constants: a second copy of one decision (rules.md 4), recorded
# for the maintainer. The 1e-12 and the 2^16 are starting values (rules.md 26; improvements.md item 5).
#
# Off (the internal switch, for tests; and always for LSMR, whose iterates a diagonal scaling would change, as for
# DampedLinearSolve's Krylov solves) the kernels are handed `UnitScale`, whose entries are the constant 1.0, so the
# kernels' operations are exactly those without conditioning.

const CONDITIONING_EPS = 1e-12
const RESCALE_BOUND = 2.0^16
const SQRT_HALF = sqrt(0.5)

# The scale of one variable from its damped diagonal entry d (DampedLinearSolve's `conditioning_scale`)
@inline function conditioning_scale(d::Real, ::Type{T}) where T
    dd = abs(Float64(d))
    isfinite(dd) || return 1.0
    f, e = frexp(1 / (CONDITIONING_EPS + sqrt(dd)))
    s = f >= SQRT_HALF ? ldexp(1.0, e) : ldexp(1.0, e - 1)
    return max(s, Float64(floatmin(T)))
end

# A scale of ones, for an unconditioned solve: indexing it gives the constant 1.0
struct UnitScale <: AbstractVector{Float64}
    n::Int
end
Base.size(u::UnitScale) = (u.n,)
Base.@propagate_inbounds Base.getindex(::UnitScale, ::Int) = 1.0

# The conditioning state of one fill: the scale `s` over all n variables, the diagonal it was computed from (`d`, as
# filled), whether it must be computed at the next solve (`stale`: set by `zero!`, and on the Jacobian side by
# `markstale!`), and the number of recomputations at a later damping (`nrescale`, for tests). `on` is the internal
# switch. A `copy_structure` copy has its own; a `sharex` form shares it, as it shares H or J.
mutable struct EliminationConditioning
    const s::Vector{Float64}
    const d::Vector{Float64}
    const on::Bool
    stale::Bool
    nrescale::Int
end
EliminationConditioning(n::Integer, on::Bool) = EliminationConditioning(ones(n), zeros(n), on, true, 0)

# The scale for a solve at the damping λ·Diagonal(sc) (`sc` a vector), computing it at the first solve of a fill from
# the diagonal `diag!(d)` writes, and recomputing it if the damping has moved far from the one it was computed at. For a
# conditioned system only (`c.on`): an unconditioned one is handed `UnitScale`s by its solve.
function prepare_scale!(c::EliminationConditioning, diag!::F, lambda::Float64, sc::AbstractVector, ::Type{T}) where {F, T}
    s, d = c.s, c.d
    lambda = max(lambda, 0.0)
    if c.stale
        diag!(d)
        @inbounds for j in eachindex(s)
            s[j] = conditioning_scale(d[j] + lambda * sc[j], T)
        end
        c.stale = false
    else
        drifted = false
        @inbounds for j in eachindex(s)
            dj = d[j] + lambda * sc[j]
            v = abs(dj) * (s[j] * s[j])
            if !(inv(RESCALE_BOUND) <= v <= RESCALE_BOUND) && conditioning_scale(dj, T) != s[j]
                drifted = true
                break
            end
        end
        if drifted
            @inbounds for j in eachindex(s)
                s[j] = conditioning_scale(d[j] + lambda * sc[j], T)
            end
            c.nrescale += 1
        end
    end
    return s
end

# x = S y
function unscale!(x::AbstractVector, s::AbstractVector)
    @inbounds for i in eachindex(x)
        x[i] *= s[i]
    end
    return x
end
unscale!(x::AbstractVector, ::UnitScale) = x

# An entry of T scaled by the Float64 factor f, formed in Float64 and narrowed once
@inline scaled(::Type{T}, a, f::Float64) where T = T(Float64(a) * f)

# The coupling column C[off : off+d-1] (eliminated rows `so`, … of the scale `s`, times the retained variable's factor
# f), each entry scaled: a static vector for a static dimension, read lazily otherwise. With a unit scale these are the
# stored entries.
@inline scaled_column(C::Vector{T}, off::Int, s::AbstractVector, so::Int, f::Float64, ::StaticInt{D}) where {T, D} =
    SVector{D, T}(ntuple(i -> @inbounds(scaled(T, C[off + i - 1], s[so + i - 1] * f)), Val(D)))
struct ScaledColumn{T, V <: AbstractVector{T}, S <: AbstractVector}
    C::V
    off::Int
    s::S
    so::Int
    f::Float64
end
Base.@propagate_inbounds Base.getindex(c::ScaledColumn{T}, i::Int) where T = scaled(T, c.C[c.off + i - 1], c.s[c.so + i - 1] * c.f)
Base.eltype(::Type{<:ScaledColumn{T}}) where T = T
@inline scaled_column(C::Vector{T}, off::Int, s::AbstractVector, so::Int, f::Float64, ::Int) where T = ScaledColumn{T, Vector{T}, typeof(s)}(C, off, s, so, f)

# dot(a[ia:ia+de-1], v[1:de]) (or v against b) with a static or run-time dimension, one operand a (scaled) column,
# accumulated in the promoted element type (Float64 for a Float32 block against a Float64 vector)
@inline function dot_col(a::Vector, ia::Int, v, s_de::Union{StaticInt, Int})
    acc = zero(promote_type(eltype(a), eltype(v)))
    @inbounds for i in 0:dimval(s_de) - 1
        acc += a[ia + i] * v[i + 1]
    end
    return acc
end
@inline function dot_col(v, b::Vector, ib::Int, s_de::Union{StaticInt, Int})
    acc = zero(promote_type(eltype(v), eltype(b)))
    @inbounds for i in 0:dimval(s_de) - 1
        acc += v[i + 1] * b[ib + i]
    end
    return acc
end

# g = S_r b_r
scaled_copy!(g::AbstractVector, b::AbstractVector, ::UnitScale) = copyto!(g, b)
function scaled_copy!(g::AbstractVector, b::AbstractVector, s::AbstractVector)
    @inbounds for i in eachindex(g)
        g[i] = b[i] * s[i]
    end
    return g
end

# S ← S_r S S_r in place, on the stored entries of a sparse S (each formed in Float64 and narrowed)
function scale_sparse!(S::SparseMatrixCSC{T}, s::AbstractVector) where T
    rv, nz = rowvals(S), nonzeros(S)
    @inbounds for j in axes(S, 2), k in nzrange(S, j)
        nz[k] = scaled(T, nz[k], s[rv[k]] * s[j])
    end
    return S
end
