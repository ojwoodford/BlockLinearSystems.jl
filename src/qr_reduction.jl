# ------------------------------------------------------------------------------------------------------------------
# Workspace and the elimination kernel
# ------------------------------------------------------------------------------------------------------------------

# QRReductionWorkspace{T}(structure::SchurJacobianStructure)
# QRReductionWorkspace(J::SchurBlockJacobian)
# Factored tables and scratch for QR elimination.
# The factored tables have the raw tables' layout (`raw_offsets`: no residual column); the residual column goes
# through the elimination in Float64, in `fres` (`J.r`'s layout, `residual_offsets`), to which the T reflectors are
# applied in Float64 (SP-B6): Qᵀr, whose top rows are the back-substitution's q and the rest the reduced residual r̃.
struct QRReductionWorkspace{T}
    structure::SchurJacobianStructure
    factored::Vector{T}                   # Qᵀ T_e for every table (free tables copied), without the residual column
    fres::Vector{Float64}                 # Qᵀ r_e for every table (free tables copied), in Float64
    scratch::Vector{Vector{Float64}}      # Per chunk: max_De scratch for blocks above D_STATIC (empty otherwise)
    failed_block::Base.Threads.Atomic{Int}
end

function QRReductionWorkspace{T}(jst::SchurJacobianStructure) where T
    scratch = [Vector{Float64}(undef, qr_scratch_length(jst)) for _ in 1:qr_nscratch(jst)]
    return QRReductionWorkspace{T}(jst, zeros(T, raw_length(jst)), zeros(residual_length(jst)), scratch, Base.Threads.Atomic{Int}(0))
end
QRReductionWorkspace(J::SchurBlockJacobian{T}) where T = QRReductionWorkspace{T}(J.structure)

# trial_workspace(ws::QRReductionWorkspace)
# A workspace for a second system with the same structure: its own factored tables and failure flag, shared scratch.
trial_workspace(ws::QRReductionWorkspace{T}) where T =
    QRReductionWorkspace{T}(ws.structure, similar(ws.factored), similar(ws.fres), ws.scratch, Base.Threads.Atomic{Int}(0))

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

# Copy table e from raw to factored (its columns from the raw table into F, its residual from the residual vector
# into the Float64 Fr), write its damping rows and QR-factor its eliminated columns in place,
# applying Qᵀ to the rest of the table and, in Float64, to its residual. The reflectors are not kept in the table: the
# eliminated columns end as [U; 0]; a refined system (`rf`, qr_refinement.jl) keeps them in its own storage first.
# Returns false for a non-finite table or a numerically singular U.
# Conditioned (conditioning.jl; `cs` the scale over all the variables, ones when off), the table's columns are scaled as
# they are copied and the damping rows are √(λ·sc)·s: the elimination is that of J S.
@inline function qr_table!(
    F::Vector{T}, Fr::Vector{Float64}, Rw::Vector{T}, Rr::Vector{Float64}, jst::SchurJacobianStructure,
    scaling_e::AbstractVector, lambda::Float64, e::Int, s_de::Union{StaticInt, Int}, cs::AbstractVector, rf = nothing
) where T
    de = dimval(s_de)
    @inbounds begin
        M = jst.table_nrows[e]
        C = jst.table_ncols[e] - 1       # columns in F (the residual is in Fr)
        off = jst.raw_offsets[e]
        rq = jst.residual_offsets[e] - 1 # Fr[rq + i] is row i of the residual
        me = M - de
        ce = C - de                      # retained columns
        copy_table!(F, Fr, Rw, Rr, jst, e, cs) || return false
        for j in 1:C, i in me + 1:M
            F[off + (j - 1) * M + i - 1] = zero(T)
        end
        for i in me + 1:M
            Fr[rq + i] = 0.0
        end
        so = jst.schur.boffsets_e[e]
        nr = jst.schur.nr
        for k in 1:de
            F[off + (ce + k - 1) * M + me + k - 1] = T(sqrt(lambda * scaling_e[so + k - 1]) * cs[nr + so + k - 1])   # formed in Float64
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
            # and to the residual, in Float64: the same operations (for T = Float64, the same values as when the
            # residual was the table's last column)
            s = Fr[rq + k]
            for i in k + 1:M
                s += F[cb + i] * Fr[rq + i]
            end
            s *= tau
            Fr[rq + k] -= s
            for i in k + 1:M
                Fr[rq + i] -= s * F[cb + i]
            end
            keep_reflector!(rf, F, cb, M, e, k, so, tau)
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
        # Retry only when another thread changed failed_block between the read and the CAS: a race that no
        # deterministic test can force, so these two lines are excluded from coverage.
        old = failed_block[]  # COV_EXCL_LINE
    end  # COV_EXCL_LINE
    return nothing
end

function qr_group!(
    F::Vector{T}, Fr::Vector{Float64}, Rw::Vector{T}, Rr::Vector{Float64}, failed_block::Base.Threads.Atomic{Int}, jst::SchurJacobianStructure,
    scaling_e::AbstractVector, lambda::Float64, e_chunk::UnitRange{Int}, s_de::Union{StaticInt, Int}, s::AbstractVector, rf = nothing
) where T
    @inbounds for e in e_chunk
        fb = failed_block[]                  # a lower failure elsewhere: no block above it can be the lowest
        fb != 0 && e > fb && return false
        if !qr_table!(F, Fr, Rw, Rr, jst, scaling_e, lambda, e, s_de, s, rf)
            record_failure!(failed_block, e)
            return false
        end
    end
    return true
end

# reduce_qr!(ws::QRReductionWorkspace, J::SchurBlockJacobian, scaling_e, lambda, s = ones) -> Bool
# QR elimination of every eliminated block; conditioned, of J S, with `s` the scale over all the variables. With `rf`
# (internal: a refined system's storage, qr_refinement.jl) the tables' reflectors are kept there.
function reduce_qr!(ws::QRReductionWorkspace{T}, J::SchurBlockJacobian{T}, scaling_e::AbstractVector, lambda_in::Real,
                    s::AbstractVector=UnitScale(0), rf = nothing) where T
    lambda = Float64(lambda_in)
    jst = ws.structure
    J.structure === jst || throw(ArgumentError("The Jacobian and the workspace were built for different structures"))
    st = jst.schur
    length(scaling_e) == sum(Int, st.e_dims; init=0) || throw(DimensionMismatch("scaling_e has length $(length(scaling_e)), expected $(sum(Int, st.e_dims; init=0))"))
    check_damping(lambda_in, scaling_e, :scaling_e)
    ws.failed_block[] = 0
    F, Fr, Rw, Rr = ws.factored, ws.fres, J.data, J.r
    for g_idx in 1:nruns(st.e_dims)
        chunks = st.chunks[g_idx]
        threaded = qr_threaded(jst, length(chunks))
        dispatch_dim(runblocksize(st.e_dims, g_idx)) do s_de
            foreach_index(length(chunks), threaded) do k
                qr_group!(F, Fr, Rw, Rr, ws.failed_block, jst, scaling_e, lambda, chunks[k], s_de, s, rf)
            end
        end
        ws.failed_block[] != 0 && return false
    end
    # Free tables need no factorization
    for t in st.neblocks + 1:jst.ntables
        copy_table!(F, Fr, Rw, Rr, jst, t, s)
    end
    return true
end

# Copy table t into the factored storage: its columns from the raw table (the same layout), and its residual from the
# residual vector into the Float64 Fr (the same layout). Returns false if an entry is not finite. Conditioned, each
# column is scaled by its variable's s as it is copied (formed in Float64, narrowed once).
@inline function copy_table!(F::Vector{T}, Fr::Vector{Float64}, Rw::Vector{T}, Rr::Vector{Float64}, jst::SchurJacobianStructure, t::Int,
                             s::AbstractVector=UnitScale(0)) where T
    @inbounds begin
        M = jst.table_nrows[t]
        C = jst.table_ncols[t] - 1
        off = jst.raw_offsets[t]
        bad = zero(T)
        if s isa UnitScale
            for i in off:off + M * C - 1
                x = Rw[i]
                F[i] = x
                bad += x - x             # NaN for an infinite or NaN entry, zero otherwise
            end
        else
            for i in off:off + M * C - 1
                x = Rw[i]
                bad += x - x
            end
            scale_table_columns!(F, Rw, jst, t, s)
        end
        rb = jst.residual_offsets[t]
        badr = 0.0
        for i in rb:rb + M - 1
            x = Rr[i]
            Fr[i] = x
            badr += x - x
        end
    end
    return !isnan(bad) && !isnan(badr)
end

# F's columns of table t, from the raw table Rw scaled by their variables' s: the retained blocks' columns, then (for an
# eliminated block's table) its eliminated columns
@inline function scale_table_columns!(F::Vector{T}, Rw::Vector{T}, jst::SchurJacobianStructure, t::Int, s::AbstractVector) where T
    st = jst.schur
    @inbounds begin
        M = jst.table_nrows[t]
        C = jst.table_ncols[t] - 1
        off = jst.raw_offsets[t]
        for u in jst.table_ptr[t]:jst.table_ptr[t + 1] - 1
            r = jst.entry_retained[u]
            ro = st.boffsets_r[r]
            for k in 1:Int(st.r_dims[r])
                f = s[ro + k - 1]
                cb = off + (jst.entry_col[u] + k - 2) * M
                for i in cb:cb + M - 1
                    F[i] = scaled(T, Rw[i], f)
                end
            end
        end
        if t <= st.neblocks
            de = Int(st.e_dims[t])
            eo = st.nr + st.boffsets_e[t] - 1
            for k in 1:de
                f = s[eo + k]
                cb = off + (C - de + k - 1) * M
                for i in cb:cb + M - 1
                    F[i] = scaled(T, Rw[i], f)
                end
            end
        end
    end
    return F
end
