module ConditioningTests

# SP-B2: the eliminated systems are conditioned over every variable, retained and eliminated, before the elimination
# (src/conditioning.jl), on the Hessian side (the Schur systems) and the Jacobian side (QR elimination), with
# DampedLinearSolve's scale (decision 13, SP-D9's recomputation) and an internal switch.

using Test, Random, LinearAlgebra, SparseArrays, Static
using DampedLinearSolve, BlockLinearSystems, Polyester, Krylov
using DampedLinearSolve: Success, NotPositiveDefinite
include("fixtures/load.jl")
using Main.SchurFixtures, Main.QRFixtures
const BLS = BlockLinearSystems

const CHOLMOD32 = Float32 <: SparseArrays.CHOLMOD.VTypes
cd_allocs(ls, s) = (solve!(ls, s); @allocated solve!(ls, s))
dirdev(x, b) = norm(x / norm(x) - b / norm(b))

# ---- Hessian side -----------------------------------------------------------------------------------------------
cd_to32(H::SchurBlockHessian{Float64}) = SchurBlockHessian(H.structure,
    H.R isa Matrix ? Float32.(H.R) : BLS.SparseBlockLinearSystem{Float32}(Float32.(H.R.data), H.R.indices, H.R.indicestransposed,
                                                                         H.R.rowblocksizes, H.R.columnblocksizes, H.R.m, H.R.n),
    Float32.(H.E_data), Float32.(H.C_data); pattern = H.pattern)
function cd_hessian(kind, p, T)
    H = kind === :dense ? dense_schur_hessian(p) : sparse_schur_hessian(p)
    return T === Float32 ? cd_to32(H) : H
end
# The Schur system of H, conditioned (as makehessiansystem builds it) or not (the internal switch)
function cd_schur(H::SchurBlockHessian{T}, b; condition = true) where T
    n = length(b)
    ls = H.R isa Matrix ? BLS.DenseSchurLinearSystem(H, copy(b), zeros(n); condition) :
        BLS.SparseSchurLinearSystem(H, copy(b), zeros(n); pattern = H.pattern, cholmod = BLS.reduced_cholmod(T), condition)
    return ls
end
cd_hsolvers(kind, T) = kind === :dense || T === Float64 || CHOLMOD32 ? (:auto, :cholesky, :ldl) : (:ldl,)
# The problem with its variables rescaled by D: D H D and D b, so that the diagonal spans 1e-12 to 1e12
function cd_badly_scaled(p)
    n = length(p.b)
    D = exp10.(range(-6, 6; length = n))[randperm(Xoshiro(5), n)]
    return SchurTestProblem(Diagonal(D) * p.H * Diagonal(D), D .* p.b, p.r_dims, p.e_dims, p.couplings, p.rr_pairs), D
end

@testset "conditioning.jl: the unit scale of an unconditioned solve" begin
    u = BLS.UnitScale(3)
    @test size(u) == (3,) && u[2] === 1.0 && collect(u) == ones(3)
    @test BLS.unscale!([1.0, 2.0, 3.0], u) == [1.0, 2.0, 3.0]
end

@testset "conditioning.jl: the Schur systems" begin
    for (pname, p) in (("standard", standard_problem()), ("mixed dimensions", mixed_dims_problem())),
        kind in (:dense, :sparse), T in (Float64, Float32), s in cd_hsolvers(kind, T)
        @testset "$pname, $kind, $T, :$s" begin
            n = length(p.b)
            H = cd_hessian(kind, p, T)
            on, off = cd_schur(H, p.b), cd_schur(H, p.b; condition = false)
            @test on.cond.on && !off.cond.on
            md = max_diag(on)
            Rcopy, Ecopy, Ccopy = copy(H.R isa Matrix ? H.R : H.R.data), copy(H.E_data), copy(H.C_data)
            for (λ, sc) in ((0.0, nothing), (1e-3 * md, nothing), (0.3, abs.(diag(p.H))))
                dampen!(on, λ, sc); dampen!(off, λ, sc)
                @test solve!(on, s) == Success && solve!(off, s) == Success
                # Powers of two commute with every operation of the reduction and the reduced solve, so on and off
                # agree exactly in Float64; in Float32 to rounding (the reduced system's own DampedLinearSolve
                # conditioning, from the scaled reduced diagonal, may then differ only through the scale's 1e-12)
                T === Float64 ? (@test getx(on) == getx(off)) : (@test getx(on) ≈ getx(off) rtol = 16 * n * eps(Float32) * cond(p.H))
                ref = reference_solve(p, λ, isnothing(sc) ? ones(n) : sc)
                @test norm(getx(on) - ref) / norm(ref) <= 16 * n * eps(T) * cond(p.H + λ * (isnothing(sc) ? I : Diagonal(sc)))
            end
            # H as filled
            @test (H.R isa Matrix ? H.R : H.R.data) == Rcopy && H.E_data == Ecopy && H.C_data == Ccopy
            # The scale: computed once per fill, from the first solve's damping; recomputed (SP-D9's band) when the damping
            # has moved far, and the earlier damping's x is then the same again
            on = cd_schur(H, p.b)
            dampen!(on, 0.0); @test solve!(on, s) == Success; x0 = copy(getx(on)); s0 = copy(on.cond.s)
            @test !on.cond.stale && on.cond.nrescale == 0
            dampen!(on, 1e-2 * md); @test solve!(on, s) == Success
            @test on.cond.s == s0 && on.cond.nrescale == 0         # within the band: the same scale
            dampen!(on, 1e16 * md); @test solve!(on, s) == Success
            @test on.cond.nrescale == 1 && on.cond.s != s0
            dampen!(on, 0.0); @test solve!(on, s) == Success
            @test on.cond.nrescale == 2 && on.cond.s == s0 && getx(on) == x0
            # The small-step limit: as λ grows, x·λ tends to b in the original variables (Levenberg damping); with
            # uniform damping in the scaled variables it would tend to S²b
            dampen!(on, 1e12 * md); @test solve!(on, s) == Success
            @test dirdev(getx(on), p.b) <= (T === Float64 ? 1e-6 : 1e-3)
            # Zero allocation (the sparse :auto and :cholesky allocate inside CHOLMOD, rules.md 21)
            dampen!(on, 0.3)
            (kind === :dense || s === :ldl) && @test cd_allocs(on, static(s)) == 0
            # A copy_structure copy has its own scale; a sharex form shares it
            t = DampedLinearSolve.copy_structure(on)
            @test t.cond !== on.cond && t.cond.on && t.cond.stale
            @test DampedLinearSolve.sharex(t, on).cond === t.cond
        end
    end
    # A variable scale that spans 1e-6 to 1e6 (the diagonal, 1e-12 to 1e12): in Float32 the conditioned solve meets the
    # reference, in the variables' own scale, to the tolerance of the well-scaled problem
    p = standard_problem()
    q, D = cd_badly_scaled(p)
    n = length(q.b)
    for kind in (:dense, :sparse), s in cd_hsolvers(kind, Float32)
        @testset "badly scaled, $kind, :$s" begin
            ref = reference_solve(q, 0.0, ones(n))
            err(ls) = norm(D .* (getx(ls) - ref)) / norm(D .* ref)
            on = cd_schur(cd_hessian(kind, q, Float32), q.b)
            dampen!(on, 0.0)
            @test solve!(on, s) == Success
            @test err(on) <= 16 * n * eps(Float32) * cond(p.H)
            # Unconditioned, the Float32 elimination of the same matrix: observed to succeed with the same error (its
            # tests are relative to each block's damped diagonal, and the reduced system's DampedLinearSolve conditioning
            # or, sparse, CHOLMOD and LDLᵀ without pivoting are unaffected by the scale), so this matrix, whose entries
            # stay well inside Float32's range, does not need it: what is observed is asserted
            off = cd_schur(cd_hessian(kind, q, Float32), q.b; condition = false)
            dampen!(off, 0.0)
            @test solve!(off, s) == Success
            @test err(off) <= 16 * n * eps(Float32) * cond(p.H)
        end
    end
    # A Schur system and the same matrix assembled as one dense system give the same x, in Float32 to eps·cond
    for kind in (:dense, :sparse), s in cd_hsolvers(kind, Float32)
        H = cd_hessian(kind, p, Float32)
        ls = cd_schur(H, p.b)
        dense = makehessiansystem(Float32.(DampedLinearSolve.conformance_matrix(H)), s === :ldl ? :ldl : :cholesky)
        getb(dense) .= p.b
        for λ in (0.0, 0.3)
            dampen!(ls, λ); dampen!(dense, λ)
            @test solve!(ls, s) == Success && solve!(dense, s === :ldl ? :ldl : :cholesky) == Success
            @test getx(ls) ≈ getx(dense) rtol = 16 * length(p.b) * eps(Float32) * cond(p.H)
        end
    end
end

# ---- Jacobian side ----------------------------------------------------------------------------------------------
cd_jsolvers(::Type{Matrix{T}}) where T = (:auto, :qr, :cholesky, :ldl)
cd_jsolvers(::Type{SparseMatrixCSC{Float64, Int}}) = (:auto, :qr, :cholesky, :ldl)
cd_jsolvers(::Type{SparseMatrixCSC{Float32, Int}}) = CHOLMOD32 ? (:cholesky, :ldl) : (:ldl,)
function cd_jacobian(p, ::Type{T}, ::Type{R}) where {T, R}
    return SchurBlockJacobian{T, R}(T.(p.J.data), Float64.(p.J.r), p.J.structure)
end
function cd_qr(J, s; condition = true)
    ls = BLS.QRSchurLinearSystem(J, s; condition)
    gradient!(getb(ls), getA(ls)); markstale!(getA(ls))
    return ls
end
# The same problem with its columns scaled by D (from 1e-6 to 1e6): each table's columns, and the dense J
function cd_scale_columns(p, D)
    J = SchurBlockJacobian{Float64, Matrix{Float64}}(copy(p.J.data), copy(p.J.r), p.J.structure)
    jst = J.structure; st = jst.schur
    for t in 1:jst.ntables
        M, off = jst.table_nrows[t], jst.raw_offsets[t]
        C = jst.table_ncols[t] - 1
        for u in jst.table_ptr[t]:jst.table_ptr[t + 1] - 1
            r = jst.entry_retained[u]
            for k in 1:Int(st.r_dims[r])
                cb = off + (jst.entry_col[u] + k - 2) * M
                J.data[cb:cb + M - 1] .*= D[st.boffsets_r[r] + k - 1]
            end
        end
        if t <= st.neblocks
            de = Int(st.e_dims[t])
            for k in 1:de
                cb = off + (C - de + k - 1) * M
                J.data[cb:cb + M - 1] .*= D[st.nr + st.boffsets_e[t] + k - 1]
            end
        end
    end
    return J, p.Jf * Diagonal(D)
end
cd_jref(Jf, r, λ, sc) = (A = Jf' * Jf + λ * (isnothing(sc) ? I : Diagonal(sc)); (A \ (Jf' * r), cond(A)))

@testset "conditioning.jl: the QR-eliminated systems" begin
    for p in qr_problems(Float64)[[1, 3]], T in (Float64, Float32),
        R in (Matrix{T}, SparseMatrixCSC{T, Int}), s in cd_jsolvers(R)
        @testset "$(p.name), $R, :$s" begin
            J = cd_jacobian(p, T, R)
            n = size(p.Jf, 2)
            Jf = Float64.(T.(p.Jf))
            on, off = cd_qr(J, s), cd_qr(J, s; condition = false)
            @test getA(on).cond.on && !getA(off).cond.on
            md = max_diag(on)
            data, r = copy(J.data), copy(J.r)
            for (λ, sc) in ((0.0, nothing), (1e-3 * md, nothing), (0.3, vec(sum(abs2, Jf; dims = 1))))
                dampen!(on, λ, sc); dampen!(off, λ, sc)
                st = solve!(on, s)
                @test st in (Success, Indefinite) && solve!(off, s) == st
                # Householder QR commutes exactly with a power-of-two column scaling, so on and off agree exactly in
                # Float64 (the reduced solvers condition the reduced system again, to the same net scale)
                T === Float64 ? (@test getx(on) == getx(off)) : (@test getx(on) ≈ getx(off) rtol = 1e-3)
                ref, κ = cd_jref(Jf, Float64.(p.rf), λ, sc)
                @test norm(getx(on) - ref) / norm(ref) <= 16 * n * eps(T) * κ
            end
            @test J.data == data && J.r == r                 # J and r as filled
            # The scale, as on the Hessian side; markstale! invalidates it
            markstale!(getA(on)); @test getA(on).cond.stale
            dampen!(on, 0.0); @test solve!(on, s) == Success; x0 = copy(getx(on)); s0 = copy(getA(on).cond.s)
            dampen!(on, 1e16 * md); @test solve!(on, s) in (Success, Indefinite)
            @test getA(on).cond.nrescale == 1
            dampen!(on, 0.0); @test solve!(on, s) == Success
            @test getA(on).cond.s == s0 && getx(on) == x0
            # The small-step limit, in the original variables. In Float32 the elimination is a Float32 QR, which loses the
            # heavily damped step's direction (decision 15's documented limit; observed 0.095 on the bundle adjustment
            # at 1e12·max_diag): conditioning does not change it (off gives the same)
            dampen!(on, 1e12 * md); dampen!(off, 1e12 * md)
            @test solve!(on, s) == Success && solve!(off, s) == Success
            @test dirdev(getx(on), getb(on)) <= (T === Float64 ? 1e-6 : 0.1)
            T === Float32 && @test dirdev(getx(on), getb(on)) ≈ dirdev(getx(off), getb(off)) rtol = 1e-2
            dampen!(on, 0.3)
            R <: Matrix && @test cd_allocs(on, static(s)) == 0
            t = DampedLinearSolve.copy_structure(on)
            @test getA(t).cond !== getA(on).cond && getA(t).cond.on
        end
    end
    # LSMR is not conditioned (a Krylov solve: a diagonal scaling would be a preconditioner)
    p = qr_problems(Float64)[1]
    @test !getA(cd_qr(cd_jacobian(p, Float64, Matrix{Float64}), :lsmr)).cond.on
    # Badly scaled columns (1e-6 to 1e6): in Float32 the conditioned elimination meets the reference, in the variables'
    # own scale, to the tolerance of the well-scaled problem
    n = size(p.Jf, 2)
    D = exp10.(range(-6, 6; length = n))[randperm(Xoshiro(7), n)]
    J64, Jd = cd_scale_columns(p, D)
    for R in (Matrix{Float32}, SparseMatrixCSC{Float32, Int}), s in cd_jsolvers(R)
        @testset "badly scaled, $R, :$s" begin
            J = SchurBlockJacobian{Float32, R}(Float32.(J64.data), copy(J64.r), J64.structure)
            Jf = Float64.(Float32.(Jd))
            ref = (Jf' * Jf) \ (Jf' * Float64.(p.rf))
            on = cd_qr(J, s)
            dampen!(on, 0.0)
            @test solve!(on, s) in (Success, Indefinite)
            relerr_on = norm(D .* (getx(on) - ref)) / norm(D .* ref)
            κ = cond(Float64.(Float32.(p.Jf))' * Float64.(Float32.(p.Jf)))
            @test relerr_on <= 16 * n * eps(Float32) * κ
            # Unconditioned: observed to succeed with the same error, as on the Hessian side (Householder QR commutes with
            # the column scaling, and its rank test is relative to each column)
            off = cd_qr(J, s; condition = false)
            dampen!(off, 0.0)
            @test solve!(off, s) in (Success, Indefinite)
            @test norm(D .* (getx(off) - ref)) / norm(D .* ref) <= 16 * n * eps(Float32) * κ
        end
    end
end

end # module
