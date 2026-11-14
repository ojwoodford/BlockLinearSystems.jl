using Static, StaticArrays, LinearAlgebra

# ------------------------------------------------------------------------------------------------------------------
# Workspace and the elimination kernel
# ------------------------------------------------------------------------------------------------------------------

# QRReductionWorkspace{T}(structure::SchurJacobianStructure)
# QRReductionWorkspace(J::SchurBlockJacobian)
# Factored tables and scratch for QR elimination.
struct QRReductionWorkspace{T}
    structure::SchurJacobianStructure
    factored::Vector{T}                   # Qᵀ T_e for every table (free tables copied), same layout as the raw data
    scratch::Vector{Vector{T}}            # Per chunk: max_De scratch for blocks above D_STATIC (empty otherwise)
    failed_block::Base.Threads.Atomic{Int}
end

function QRReductionWorkspace{T}(jst::SchurJacobianStructure) where T
    scratch = [Vector{T}(undef, qr_scratch_length(jst)) for _ in 1:qr_nscratch(jst)]
    return QRReductionWorkspace{T}(jst, zeros(T, table_length(jst)), scratch, Base.Threads.Atomic{Int}(0))
end
QRReductionWorkspace(J::SchurBlockJacobian{T}) where T = QRReductionWorkspace{T}(J.structure)

# trial_workspace(ws::QRReductionWorkspace)
# A workspace for a second system with the same structure: its own factored tables and failure flag, shared scratch.
trial_workspace(ws::QRReductionWorkspace{T}) where T =
    QRReductionWorkspace{T}(ws.structure, similar(ws.factored), ws.scratch, Base.Threads.Atomic{Int}(0))

# Unrolled over the columns for a static dimension; stops (returning false) as soon as f(k) returns false
@generated function unrolled_all(f::F, ::StaticInt{D}) where {F, D}
    return quote
        Base.Cartesian.@nexprs $D k -> (f(k) || return false)
        return true
    end
end
@inline function unrolled_all(f::F, d::Int) where F
    for k in 1:d
        f(k) || return false
    end
    return true
end

# Copy table e from raw to factored, write its damping rows and QR-factor its eliminated columns in place,
# applying Qᵀ to the rest of the table. The reflectors are not kept: the eliminated columns end as [U; 0].
# Returns false for a non-finite table or a numerically singular U.
@inline function qr_table!(
    F::Vector{T}, Rw::Vector{T}, jst::SchurJacobianStructure,
    scaling_e::AbstractVector{T}, lambda::T, e::Int, s_de::Union{StaticInt, Int}
) where T
    de = dimval(s_de)
    @inbounds begin
        M = jst.table_nrows[e]
        C = jst.table_ncols[e]
        off = jst.table_offsets[e]
        me = M - de
        ce = C - de - 1                  # retained columns
        bad = zero(T)
        for i in off:off + M * C - 1
            x = Rw[i]
            F[i] = x
            bad += x - x                 # NaN for an infinite or NaN entry, zero otherwise
        end
        isnan(bad) && return false
        for j in 1:C, i in me + 1:M
            F[off + (j - 1) * M + i - 1] = zero(T)
        end
        so = jst.schur.boffsets_e[e]
        for k in 1:de
            F[off + (ce + k - 1) * M + me + k - 1] = sqrt(lambda * scaling_e[so + k - 1])
        end
    end
    # Relative singularity threshold on the diagonal of U, scaled by the number of rows: the rounding left on an
    # exactly dependent column grows with the row count
    tol = M * eps(T)
    return unrolled_all(s_de) do k
        @inbounds begin
            cb = off + (ce + k - 1) * M - 1      # F[cb + i] is row i of eliminated column k
            s_top = zero(T)
            for i in 1:k - 1
                s_top += abs2(F[cb + i])
            end
            s_sub = zero(T)
            for i in k:M
                s_sub += abs2(F[cb + i])
            end
            nrm = sqrt(s_sub)
            full = sqrt(s_top + s_sub)          # norm of the original (damped) column: Q preserves it
            a = F[cb + k]
            beta = a >= zero(T) ? -nrm : nrm
            (isfinite(beta) && abs(beta) > tol * full) || return false
            scale = inv(a - beta)
            for i in k + 1:M
                F[cb + i] *= scale
            end
            tau = (beta - a) / beta
            F[cb + k] = beta
            # Apply H = I - tau v v' (v = [1; F[cb+k+1:cb+M]]) to every other column not yet triangularised
            for j in 1:C
                (ce + 1 <= j <= ce + k) && continue
                yb = off + (j - 1) * M - 1
                s = F[yb + k]
                for i in k + 1:M
                    s += F[cb + i] * F[yb + i]
                end
                s *= tau
                F[yb + k] -= s
                for i in k + 1:M
                    F[yb + i] -= s * F[cb + i]
                end
            end
            for i in k + 1:M
                F[cb + i] = zero(T)
            end
        end
        return true
    end
end

@inline function record_failure!(failed_block::Base.Threads.Atomic{Int}, e::Int)
    old = failed_block[]
    while (old == 0 || e < old)
        old_val = Threads.atomic_cas!(failed_block, old, e)
        old_val == old && break
        old = failed_block[]
    end
    return nothing
end

function qr_group!(
    F::Vector{T}, Rw::Vector{T}, failed_block::Base.Threads.Atomic{Int}, jst::SchurJacobianStructure,
    scaling_e::AbstractVector{T}, lambda::T, e_chunk::UnitRange{Int}, s_de::Union{StaticInt, Int}
) where T
    @inbounds for e in e_chunk
        failed_block[] != 0 && return false
        if !qr_table!(F, Rw, jst, scaling_e, lambda, e, s_de)
            record_failure!(failed_block, e)
            return false
        end
    end
    return true
end

# reduce_qr!(ws::QRReductionWorkspace, J::SchurBlockJacobian, scaling_e, lambda) -> Bool
# QR elimination of every eliminated block.
function reduce_qr!(ws::QRReductionWorkspace{T}, J::SchurBlockJacobian{T}, scaling_e::AbstractVector{T}, lambda_in::Real) where T
    lambda = T(lambda_in)
    jst = ws.structure
    J.structure === jst || throw(ArgumentError("The Jacobian and the workspace were built for different structures"))
    st = jst.schur
    length(scaling_e) == sum(Int, st.e_dims; init=0) || throw(DimensionMismatch("scaling_e has length $(length(scaling_e)), expected $(sum(Int, st.e_dims; init=0))"))
    ws.failed_block[] = 0
    F, Rw = ws.factored, J.data
    for (g_idx, e_range) in enumerate(st.e_groups)
        chunks = st.chunks[g_idx]
        threaded = qr_threaded(jst, length(chunks))
        dispatch_dim(st.e_sizes[g_idx]) do s_de
            foreach_index(length(chunks), threaded) do k
                qr_group!(F, Rw, ws.failed_block, jst, scaling_e, lambda, chunks[k], s_de)
            end
        end
        ws.failed_block[] != 0 && return false
    end
    # Free tables need no factorization
    o1 = jst.table_offsets[st.neblocks + 1]
    n = jst.table_offsets[end] - o1
    n > 0 && copyto!(F, o1, Rw, o1, n)
    return true
end
