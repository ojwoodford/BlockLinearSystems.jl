# Mixed-precision iterative refinement of the QR-eliminated solve over the whole system, the tables included (SP-B7),
# for an element type T narrower than Float64 with the dense reduced system solved by QR (`:auto`, `:qr`). Float64
# systems, and every other reduced solver, are not refined: the system holds no refinement (`nothing`), and the solve
# selects it by dispatch.
#
# The method is DampedLinearSolve SP-D11's, Björck's refinement of the augmented system with the residual carried as an
# unknown (Björck, BIT 7, 1967; Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., §20.5), applied to the
# whole conditioned, damped problem min ‖A y − c‖, x = S y, whose rows are every table's rows (its data rows, then for
# an eliminated table its d_e damping rows √(λ s_e)·S_e) and the retained variables' damping rows √(λ s_r)·S_r, and
# c = [r; 0]. The elimination and the reduced solve are its QR, A = Q R:
#     Q = (each eliminated table's reflectors, on its rows) · Q̃ (the reduced system's, on the reduced rows: every
#         table's rows below its top d_e, and the retained damping rows),
#     R = [U B; 0 R̃] (U block diagonal, one d_e × d_e triangle per eliminated block; B the top rows of the tables'
#         retained columns; R̃ the reduced system's factor).
# The reduced system's factors come from DampedLinearSolve's `qrfactors` (SP-D12), in the reduced system's own
# variables, which are these conditioned ones. One step forms, in Float64, from the tables as filled (J S exact in
# Float64: the scale is powers of two), r, the scale and the damping,
#     f₁ = c − ρ − A y,   f₂ = −Aᵀρ,
# and solves for the correction with the T factors applied in Float64:
#     h = R⁻ᵀ f₂;  (d₁, d₂) = Qᵀ f₁;  δy = R⁻¹(d₁ − h);  δρ = Q (h, d₂).
#
# The start (SP-B7's amendment): y₀ is the solve as it is (the eliminated solve on the reduced system's refined x_r),
# and ρ₀ = Q (0, d₂) for d = Qᵀc, the factors' residual (not c − A y₀, whose first step would be the corrected seminormal
# equations). Björck's own start (y₀ from the factors alone, which recomputes the reduced solution the reduced solve
# has already refined) is kept for the tests that chose between them.
#
# The tables' reflectors are kept here, not in the factored tables: `jtr!`, `jtj_diag!` and `jtj_quadform` of a
# QRReductionWorkspace read the eliminated columns as [U; 0]. The number of steps is a starting value (rules.md 26;
# improvements.md item 5).
const ELIMINATION_REFINEMENT_STEPS = 1

# The refinement's storage, allocated with the system (Float32 with the dense reduced QR only): the tables' reflectors
# (`V`, the vectors below U of every eliminated column, laid out as the eliminated columns, M_e × d_e per table from
# `voff[e]`; `tau`, one per eliminated variable) and Float64 vectors: ρ and f₁ over every row of A (J.r's layout, then
# the n_r retained damping rows), h, the correction δy and x = S y over the n variables, and w over the reduced rows
# (the reduced system's m̃ + n_r).
struct EliminationRefinement{T}
    V::Vector{T}
    tau::Vector{T}
    voff::Vector{Int}
    rho::Vector{Float64}
    f::Vector{Float64}
    h::Vector{Float64}
    dy::Vector{Float64}
    xs::Vector{Float64}
    w::Vector{Float64}
end
function EliminationRefinement{T}(jst::SchurJacobianStructure) where T
    st = jst.schur
    ne = sum(Int, st.e_dims; init=0)
    n = st.nr + ne
    voff = Vector{Int}(undef, st.neblocks + 1)
    voff[1] = 1
    for e in 1:st.neblocks
        voff[e + 1] = voff[e] + jst.table_nrows[e] * Int(st.e_dims[e])
    end
    nf = residual_length(jst) + st.nr
    return EliminationRefinement{T}(zeros(T, voff[end] - 1), zeros(T, ne), voff, zeros(nf), zeros(nf), zeros(n), zeros(n),
                                    zeros(n), zeros(reduced_nrows(jst) + st.nr))
end
# Which systems are refined: by the element type and the reduced system
elimination_refinement(J::SchurBlockJacobian{T}, ::DenseQRReduced) where T = T === Float64 ? nothing : EliminationRefinement{T}(J.structure)
elimination_refinement(::SchurBlockJacobian, red) = nothing

# Keep reflector k of eliminated table e (its vector below the diagonal, F[cb + k + 1 : cb + M], and its factor), before
# the elimination kernel zeroes it. Nothing for a system that is not refined.
@inline keep_reflector!(::Nothing, F, cb, M, e, k, so, tau) = nothing
@inline function keep_reflector!(rf::EliminationRefinement, F::Vector, cb::Int, M::Int, e::Int, k::Int, so::Int, tau)
    @inbounds begin
        vb = rf.voff[e] + (k - 1) * M - 1
        for i in k + 1:M
            rf.V[vb + i] = F[cb + i]
        end
        rf.tau[so + k - 1] = tau
    end
    return nothing
end

# f[rq + 1 : rq + M] = H_k f[…] for reflector k of eliminated table e: the operations the elimination kernel applies to
# the residual, in Float64 from the T reflector
@inline function reflect_table!(f::Vector{Float64}, rf::EliminationRefinement, e::Int, k::Int, rq::Int, M::Int, so::Int)
    @inbounds begin
        V = rf.V
        vb = rf.voff[e] + (k - 1) * M - 1
        s = f[rq + k]
        for i in k + 1:M
            s += V[vb + i] * f[rq + i]
        end
        s *= rf.tau[so + k - 1]
        f[rq + k] -= s
        for i in k + 1:M
            f[rq + i] -= s * V[vb + i]
        end
    end
    return f
end
# f = Qᵀf (forward) or Q f (reverse) over the rows of every eliminated table
function apply_table_reflectors!(f::Vector{Float64}, rf::EliminationRefinement, jst::SchurJacobianStructure, reverse::Bool)
    st = jst.schur
    foreach_range(st.neblocks, qr_nchunks(jst), qr_threaded(jst, min(st.neblocks, qr_nchunks(jst)))) do e
        @inbounds begin
            M = jst.table_nrows[e]
            de = Int(st.e_dims[e])
            rq = jst.residual_offsets[e] - 1
            so = st.boffsets_e[e]
            if reverse
                for k in de:-1:1
                    reflect_table!(f, rf, e, k, rq, M, so)
                end
            else
                for k in 1:de
                    reflect_table!(f, rf, e, k, rq, M, so)
                end
            end
        end
    end
    return f
end

# w = the reduced rows of f (every table's rows below its top, then the retained damping rows), and back
function gather_reduced!(w::Vector{Float64}, f::Vector{Float64}, jst::SchurJacobianStructure)
    foreach_table(jst) do t
        @inbounds begin
            top = jst.table_top[t]
            rb = jst.residual_offsets[t] + top
            wo = jst.reduced_offsets[t]
            for i in 0:jst.table_nrows[t] - top - 1
                w[wo + i] = f[rb + i]
            end
        end
    end
    mt, nf = reduced_nrows(jst), residual_length(jst)
    @inbounds for j in 1:jst.schur.nr
        w[mt + j] = f[nf + j]
    end
    return w
end
function scatter_reduced!(f::Vector{Float64}, w::Vector{Float64}, jst::SchurJacobianStructure)
    foreach_table(jst) do t
        @inbounds begin
            top = jst.table_top[t]
            rb = jst.residual_offsets[t] + top
            wo = jst.reduced_offsets[t]
            for i in 0:jst.table_nrows[t] - top - 1
                f[rb + i] = w[wo + i]
            end
        end
    end
    mt, nf = reduced_nrows(jst), residual_length(jst)
    @inbounds for j in 1:jst.schur.nr
        f[nf + j] = w[mt + j]
    end
    return f
end
# Set the top d_e rows of every eliminated table in f from v (over the eliminated variables, v[nr + …]), or to zero
function set_tops!(f::Vector{Float64}, v::Union{Nothing, Vector{Float64}}, jst::SchurJacobianStructure)
    st = jst.schur
    @inbounds for e in 1:st.neblocks
        rq = jst.residual_offsets[e] - 1
        eo = st.nr + st.boffsets_e[e] - 1
        for i in 1:Int(st.e_dims[e])
            f[rq + i] = v === nothing ? 0.0 : v[eo + i]
        end
    end
    return f
end

# The damping entry of A for variable j (√(λ·sc_j)·s_j), in Float64
@inline damping_row(lambda::Float64, sc::AbstractVector, cs::AbstractVector, j::Int) = @inbounds sqrt(lambda * sc[j]) * cs[j]

# f = c − ρ − A y, from the raw tables (x = S y in `xs`), r, and the damping
function residual_rows!(f::Vector{Float64}, rho::Vector{Float64}, J::SchurBlockJacobian, xs::Vector{Float64},
                        lambda::Float64, sc::AbstractVector, cs::AbstractVector, y::Vector{Float64})
    jst = J.structure
    st = jst.schur
    D, r = J.data, J.r
    foreach_table(jst) do t
        @inbounds begin
            M = jst.table_nrows[t]
            de = t <= st.neblocks ? Int(st.e_dims[t]) : 0
            me = M - de
            off = jst.raw_offsets[t]
            rq = jst.residual_offsets[t] - 1
            for i in 1:me
                f[rq + i] = r[rq + i] - rho[rq + i]
            end
            for u in jst.table_ptr[t]:jst.table_ptr[t + 1] - 1
                ro = st.boffsets_r[jst.entry_retained[u]]
                cb = off + (jst.entry_col[u] - 1) * M - 1
                for j in 1:Int(st.r_dims[jst.entry_retained[u]])
                    xj = xs[ro + j - 1]
                    c0 = cb + (j - 1) * M
                    for i in 1:me
                        f[rq + i] -= Float64(D[c0 + i]) * xj
                    end
                end
            end
            if de > 0
                C = jst.table_ncols[t] - 1
                eo = st.nr + st.boffsets_e[t] - 1
                for k in 1:de
                    xj = xs[eo + k]
                    c0 = off + (C - de + k - 1) * M - 1
                    for i in 1:me
                        f[rq + i] -= Float64(D[c0 + i]) * xj
                    end
                    f[rq + me + k] = -rho[rq + me + k] - damping_row(lambda, sc, cs, eo + k) * y[eo + k]
                end
            end
        end
    end
    nf = residual_length(jst)
    @inbounds for j in 1:st.nr
        f[nf + j] = -rho[nf + j] - damping_row(lambda, sc, cs, j) * y[j]
    end
    return f
end

# h = −Aᵀρ, from the raw tables, the scale and the damping
function gradient_rows!(h::Vector{Float64}, rho::Vector{Float64}, J::SchurBlockJacobian, lambda::Float64,
                        sc::AbstractVector, cs::AbstractVector)
    jst = J.structure
    st = jst.schur
    D = J.data
    nf = residual_length(jst)
    foreach_retained(jst) do rb
        @inbounds begin
            ro = st.boffsets_r[rb]
            dr = Int(st.r_dims[rb])
            for j in 0:dr - 1
                h[ro + j] = 0.0
            end
            for k in jst.retained_ptr[rb]:jst.retained_ptr[rb + 1] - 1
                u = jst.retained_entry[k]
                t = jst.entry_table[u]
                M = jst.table_nrows[t]
                me = M - (t <= st.neblocks ? Int(st.e_dims[t]) : 0)
                cb = jst.raw_offsets[t] + (jst.entry_col[u] - 1) * M
                rq = jst.residual_offsets[t]
                for j in 1:dr
                    h[ro + j - 1] += dot_run64(D, cb + (j - 1) * M, rho, rq, me)
                end
            end
            for j in ro:ro + dr - 1
                h[j] = -(cs[j] * h[j] + damping_row(lambda, sc, cs, j) * rho[nf + j])
            end
        end
    end
    foreach_range(st.neblocks, qr_nchunks(jst), qr_threaded(jst, min(st.neblocks, qr_nchunks(jst)))) do e
        @inbounds begin
            M = jst.table_nrows[e]
            de = Int(st.e_dims[e])
            me = M - de
            C = jst.table_ncols[e] - 1
            off = jst.raw_offsets[e]
            rq = jst.residual_offsets[e]
            eo = st.nr + st.boffsets_e[e] - 1
            for k in 1:de
                t = dot_run64(D, off + (C - de + k - 1) * M, rho, rq, me)
                h[eo + k] = -(cs[eo + k] * t + damping_row(lambda, sc, cs, eo + k) * rho[rq + me + k - 1])
            end
        end
    end
    return h
end

# h = R⁻ᵀh for R = [U B; 0 R̃]: h_e = U⁻ᵀh_e per eliminated block, then h_r = R̃⁻ᵀ(h_r − Bᵀh_e)
function solve_transpose_factor!(h::Vector{Float64}, ws::QRReductionWorkspace, Rt)
    jst = ws.structure
    st = jst.schur
    F = ws.factored
    nr = st.nr
    foreach_range(st.neblocks, qr_nchunks(jst), qr_threaded(jst, min(st.neblocks, qr_nchunks(jst)))) do e
        @inbounds begin
            M = jst.table_nrows[e]
            de = Int(st.e_dims[e])
            ub = jst.raw_offsets[e] + (jst.table_ncols[e] - 1 - de) * M - 1   # F[ub + (j - 1) * M + i] is U[i, j]
            eo = nr + st.boffsets_e[e] - 1
            for j in 1:de
                t = h[eo + j]
                for i in 1:j - 1
                    t -= Float64(F[ub + (j - 1) * M + i]) * h[eo + i]
                end
                h[eo + j] = t / Float64(F[ub + (j - 1) * M + j])
            end
        end
    end
    foreach_retained(jst) do rb
        @inbounds begin
            ro = st.boffsets_r[rb]
            for k in jst.retained_ptr[rb]:jst.retained_ptr[rb + 1] - 1
                u = jst.retained_entry[k]
                e = jst.entry_table[u]
                e <= st.neblocks || continue
                M = jst.table_nrows[e]
                cb = jst.raw_offsets[e] + (jst.entry_col[u] - 1) * M - 1
                eo = nr + st.boffsets_e[e] - 1
                for j in 1:Int(st.r_dims[rb]), i in 1:Int(st.e_dims[e])
                    h[ro + j - 1] -= Float64(F[cb + (j - 1) * M + i]) * h[eo + i]
                end
            end
        end
    end
    ldiv!(Rt, view(h, 1:nr))
    return h
end

# `steps` steps of refinement of the eliminated solve whose solution, in the conditioned variables, is (x_r, x_e) =
# (`ls.xr`, `ls.xe`), at the damping `lambda`, `sc` (the system's scaling) and the scale `cs`. `bjorck` takes Björck's
# start instead of the solve's (for the tests).
refine_elimination!(::Nothing, ls, cs, lambda, sc, steps, bjorck) = nothing
function refine_elimination!(rf::EliminationRefinement, ls, cs::AbstractVector, lambda::Float64, sc::AbstractVector,
                             steps::Integer, bjorck::Bool)
    ws = ls.ws
    J = ls.A.J
    jst = J.structure
    st = jst.schur
    nr = st.nr
    n = length(rf.h)
    Q, R = DampedLinearSolve.qrfactors(ls.reduced.sys)
    rho, f, h, dy, xs, w = rf.rho, rf.f, rf.h, rf.dy, rf.xs, rf.w
    xr, xe = ls.xr, ls.xe
    nf = residual_length(jst)
    # d = Qᵀc: the tables' part is the transformed residual of the elimination (ws.fres); then ρ₀ = Q (0, d₂)
    @inbounds for i in 1:nf
        f[i] = ws.fres[i]
    end
    @inbounds for j in 1:nr
        f[nf + j] = 0.0
    end
    gather_reduced!(w, f, jst)
    lmul!(Q', w)
    if bjorck
        @inbounds for i in 1:nr
            xr[i] = w[i]
        end
        ldiv!(R, xr)
        backsubstitute!(xe, xr, ws)
    end
    @inbounds for i in 1:nr
        w[i] = 0.0
    end
    lmul!(Q, w)
    scatter_reduced!(f, w, jst)
    set_tops!(f, nothing, jst)
    apply_table_reflectors!(f, rf, jst, true)
    copyto!(rho, f)
    for _ in 1:steps
        # y and x = S y
        @inbounds for i in 1:nr
            xs[i] = cs[i] * xr[i]
            dy[i] = xr[i]
        end
        @inbounds for i in 1:n - nr
            xs[nr + i] = cs[nr + i] * xe[i]
            dy[nr + i] = xe[i]
        end
        # f₁ = c − ρ − A y and f₂ = −Aᵀρ (in h), in Float64
        residual_rows!(f, rho, J, xs, lambda, sc, cs, dy)
        gradient_rows!(h, rho, J, lambda, sc, cs)
        solve_transpose_factor!(h, ws, R')                       # h = R⁻ᵀ f₂
        # d = Qᵀ f₁
        apply_table_reflectors!(f, rf, jst, false)
        gather_reduced!(w, f, jst)
        lmul!(Q', w)
        # δy = R⁻¹(d₁ − h): δy_r = R̃⁻¹(d̃₁ − h_r), then δy_e = U⁻¹(d₁_e − h_e − B δy_r); y += δy
        @inbounds for i in 1:nr
            w[i] -= h[i]
            dy[i] = w[i]
        end
        ldiv!(R, view(dy, 1:nr))
        @inbounds for e in 1:st.neblocks
            rq = jst.residual_offsets[e] - 1
            eo = nr + st.boffsets_e[e] - 1
            for i in 1:Int(st.e_dims[e])
                f[rq + i] -= h[eo + i]
            end
        end
        backsubstitute!(view(dy, nr + 1:n), view(dy, 1:nr), ws, f)
        @inbounds for i in 1:nr
            xr[i] += dy[i]
        end
        @inbounds for i in 1:n - nr
            xe[i] += dy[nr + i]
        end
        # δρ = Q (h, d₂); ρ += δρ
        @inbounds for i in 1:nr
            w[i] = h[i]
        end
        lmul!(Q, w)
        scatter_reduced!(f, w, jst)
        set_tops!(f, h, jst)
        apply_table_reflectors!(f, rf, jst, true)
        @inbounds for i in eachindex(rho)
            rho[i] += f[i]
        end
    end
    return nothing
end
