module QRRefinementTests

# SP-B7: the whole-system refinement of the Float32 QR-eliminated solve with the dense reduced QR (src/qr_refinement.jl)
using Test, LinearAlgebra, SparseArrays, Random, Static, Polyester
using BlockLinearSystems, DampedLinearSolve
const BLS = BlockLinearSystems
include("fixtures/load.jl")
using Main.QRFixtures

function qrr_setres!(J, r)
    row = 0
    for t in 1:J.structure.ntables
        v = BLS.table_residual_block(J, t); v .= r[row+1:row+length(v)]; row += length(v)
    end
    return J
end
# A system of element type T for the tables `data` (in the fixture's layout `jst`) and the residual r
function qrr_system(::Type{T}, jst, data, r, s; condition = true, RT = Matrix{T}) where T
    J = qrr_setres!(SchurBlockJacobian{T, RT}(jst), r)
    J.data .= T.(data)
    ls = BLS.QRSchurLinearSystem(J, s; condition)
    gradient!(getb(ls), getA(ls)); markstale!(getA(ls))
    return ls
end
# SP-B6's large residual: r orthogonal to the range of the narrowed J but for ‖Jᵀr‖ = 1e-6·‖J‖·‖r‖
function qrr_largeres(p, rng)
    Jd = Float64.(Float32.(qr_dense(p.J)))
    m, n = size(Jd)
    Q1 = Matrix(qr(Jd).Q)[:, 1:n]
    rp = randn(rng, m); rp -= Q1 * (Q1' * rp); rp /= norm(rp)
    x0 = randn(rng, n)
    return Jd, rp + Jd * x0 * (1e-6 * opnorm(Jd) / norm(Jd' * (Jd * x0)))
end
# The damped least-squares solution of Jd, in BigFloat
qrr_reference(Jd, r, λ) = setprecision(BigFloat, 256) do
    n = size(Jd, 2)
    Float64.(qr(vcat(BigFloat.(Jd), sqrt(BigFloat(λ)) * Matrix{BigFloat}(I, n, n))) \ vcat(BigFloat.(r), zeros(BigFloat, n)))
end
qrr_rel(a, b) = norm(a - b) / norm(b)
qrr_median(v) = sort(v)[cld(length(v), 2)]
qrr_errors(ls, xr, steps; bjorck = false) =
    (BLS.refined_solve!(ls, static(:qr), steps, bjorck) == Success ? qrr_rel(getx(ls), xr) : NaN)
function qrr_allocs(ls, s)
    solve!(ls, s)
    return @allocated solve!(ls, s)
end

@testset "qr_refinement.jl" begin
    probs = qr_problems(Float64)

    @testset "storage: Float32 with the dense reduced QR only" begin
        p = probs[1]
        jst = p.J.structure
        r = randn(Xoshiro(1), BLS.residual_length(jst))
        for s in (:auto, :qr)
            rf = qrr_system(Float32, jst, p.J.data, r, s).refine
            @test rf isa BLS.EliminationRefinement{Float32}
            @test length(rf.rho) == length(rf.f) == BLS.residual_length(jst) + jst.schur.nr
            @test length(rf.w) == BLS.reduced_nrows(jst) + jst.schur.nr && length(rf.h) == BLS.jacobian_ncols(p.J)
            @test length(rf.tau) == sum(Int, jst.schur.e_dims) && length(rf.V) == sum(e -> jst.table_nrows[e] * Int(jst.schur.e_dims[e]), 1:jst.schur.neblocks)
        end
        # Float64, the normal equations, the sparse storage and LSMR: none
        @test qrr_system(Float64, jst, p.J.data, r, :qr).refine === nothing
        for s in (:cholesky, :ldl)
            @test qrr_system(Float32, jst, p.J.data, r, s).refine === nothing
        end
        @test qrr_system(Float32, jst, p.J.data, r, :ldl; RT = SparseMatrixCSC{Float32, Int}).refine === nothing
        @test BLS.ELIMINATION_REFINEMENT_STEPS == 1
    end

    @testset "the tables' reflectors, kept: $(p.name)" for p in probs
        jst = p.J.structure
        r = randn(Xoshiro(2), BLS.residual_length(jst))
        ls = qrr_system(Float32, jst, p.J.data, r, :qr)
        dampen!(ls, 0.1 * max_diag(ls))
        @test solve!(ls, :qr) == Success
        rf = ls.refine
        # Qᵀ of the residual (its damping rows zero) is the elimination's transformed residual, exactly
        f = [copy(getA(ls).J.r); zeros(jst.schur.nr)]
        for e in 1:jst.schur.neblocks
            M, de = jst.table_nrows[e], Int(jst.schur.e_dims[e])
            f[jst.residual_offsets[e] .+ (M - de:M - 1)] .= 0
        end
        BLS.apply_table_reflectors!(f, rf, jst, false)
        @test f[1:BLS.residual_length(jst)] == ls.ws.fres
        # and Q undoes it
        v = randn(Xoshiro(3), length(f))
        @test qrr_rel(BLS.apply_table_reflectors!(BLS.apply_table_reflectors!(copy(v), rf, jst, false), rf, jst, true), v) <= 1e-6
        # the factored tables still end as [U; 0] (the reflectors are not kept there)
        for e in 1:jst.schur.neblocks
            M, de, C = jst.table_nrows[e], Int(jst.schur.e_dims[e]), jst.table_ncols[e] - 1
            T = reshape(ls.ws.factored[jst.raw_offsets[e]:jst.raw_offsets[e+1]-1], M, C)
            @test all(iszero, (T[i, C - de + k] for k in 1:de for i in k+1:M))
        end
    end

    # The error falls by about eps(Float32)·κ per step (Björck; DampedLinearSolve SP-D11), from the Float32
    # elimination's own error to the Float64 eliminated solve's
    @testset "large residual: $(p.name), λ = $λrel·max_diag" for (k, p) in enumerate(probs), λrel in (0.0, 1e-4)
        jst = p.J.structure
        e = map(1:4) do seed
            Jd, r = qrr_largeres(p, Xoshiro(100k + seed))
            ls = qrr_system(Float32, jst, Float32.(p.J.data), r, :qr)
            λ = λrel * max_diag(ls); dampen!(ls, λ)
            xr = qrr_reference(Jd, r, λ)
            l64 = qrr_system(Float64, jst, Float64.(Float32.(p.J.data)), r, :qr); dampen!(l64, λ)
            @test solve!(l64, :qr) == Success
            e2b = qrr_errors(ls, xr, 2; bjorck = true)
            (qrr_errors(ls, xr, 0), qrr_errors(ls, xr, 1), qrr_errors(ls, xr, 2), qrr_rel(getx(l64), xr), e2b,
             (solve!(ls, :qr); qrr_rel(getx(ls), xr)))
        end
        e0, e1, e2, e64, e2b, edef = (qrr_median(getindex.(e, i)) for i in 1:6)
        @test e1 <= 1e-4 * e0
        @test e2 <= 1e-7 * e0 && e2 <= max(10 * e64, 1e-9)
        @test e2b <= 1e-7 * e0          # Björck's start converges as well
        @test edef == e1                # solve! takes one step
    end

    # Ill-conditioned with a small residual: never worse than unrefined
    @testset "small residual, ill-conditioned: $(p.name), λ = $λrel·max_diag" for (k, p) in enumerate(probs), λrel in (0.0, 1e-4)
        jst = p.J.structure
        st = jst.schur
        for seed in 1:4
            rng = Xoshiro(200k + seed)
            D = Float64.(Float32.(p.J.data))
            # every retained block's second column in each table is its first plus δ·noise
            for t in 1:jst.ntables
                M = jst.table_nrows[t]; me = M - (t <= st.neblocks ? Int(st.e_dims[t]) : 0)
                for u in jst.table_ptr[t]:jst.table_ptr[t+1]-1
                    Int(st.r_dims[jst.entry_retained[u]]) >= 2 || continue
                    c1 = jst.raw_offsets[t] + (jst.entry_col[u] - 1) * M
                    D[c1 + M .+ (0:me-1)] .= D[c1 .+ (0:me-1)] .+ 1e-4 .* randn(rng, me)
                end
            end
            J32 = SchurBlockJacobian{Float32, Matrix{Float32}}(jst); J32.data .= Float32.(D)
            Jd = Float64.(qr_dense(J32)); m, n = size(Jd)
            b = Jd * randn(rng, n); r = b + 1e-7 * norm(b) / sqrt(m) * randn(rng, m)
            ls = qrr_system(Float32, jst, J32.data, r, :qr)
            λ = λrel * max_diag(ls); dampen!(ls, λ)
            xr = qrr_reference(Jd, r, λ)
            e0, e1, e2 = (qrr_errors(ls, xr, s) for s in 0:2)
            @test e1 <= e0 && e2 <= e0
        end
    end

    @testset "unconditioned (the internal switch off)" begin
        p = probs[2]
        Jd, r = qrr_largeres(p, Xoshiro(7))
        ls = qrr_system(Float32, p.J.structure, Float32.(p.J.data), r, :qr; condition = false)
        dampen!(ls, 0.0)
        xr = qrr_reference(Jd, r, 0.0)
        @test qrr_errors(ls, xr, 1) <= 1e-4 * qrr_errors(ls, xr, 0)
    end

    @testset "Float64 is not refined" begin
        p = probs[3]
        r = randn(Xoshiro(4), BLS.residual_length(p.J.structure))
        ls = qrr_system(Float64, p.J.structure, p.J.data, r, :qr)
        for λ in (0.0, 0.3)
            dampen!(ls, λ)
            x = map(s -> (BLS.refined_solve!(ls, static(:qr), s, false); copy(getx(ls))), 0:2)
            @test x[1] == x[2] == x[3]
        end
    end

    @testset "the contract, the trial system, and allocation" begin
        p = probs[1]
        jst = p.J.structure
        Jd, r = qrr_largeres(p, Xoshiro(5))
        for s in (:auto, :qr)
            ls = qrr_system(Float32, jst, Float32.(p.J.data), r, s)
            dampen!(ls, 0.0); @test solve!(ls, s) == Success; x0 = copy(getx(ls))
            dampen!(ls, 0.2, fill(0.5, length(x0))); @test solve!(ls, s) == Success; x1 = copy(getx(ls))
            dampen!(ls, 0.0); @test solve!(ls, s) == Success; @test getx(ls) == x0
            # the trial system shares the refinement's storage, and a solve of it changes nothing of the main one's
            tr = DampedLinearSolve.copy_structure(ls)
            getA(tr).J.data .= getA(ls).J.data .* 2; getA(tr).J.r .= getA(ls).J.r
            gradient!(getb(tr), getA(tr)); markstale!(getA(tr))
            @test tr.refine === ls.refine
            dampen!(tr, 0.0); @test solve!(tr, s) == Success
            @test getx(tr) ≈ x0 ./ 2 rtol = 1e-5
            dampen!(ls, 0.0); @test solve!(ls, s) == Success; @test getx(ls) == x0
            # Zero allocation, undamped and damped with a Marquardt scaling (the fixtures are below the threading threshold)
            dampen!(ls, 0.0); @test qrr_allocs(ls, static(s)) == 0
            dampen!(ls, 0.2, fill(0.5, length(x0))); @test qrr_allocs(ls, static(s)) == 0
        end
    end

    # Threaded as the elimination is: the same x on one thread and threaded (min_work = 0, two chunks)
    @testset "threaded" begin
        layout(minwork) = SchurJacobianStructure(fill(6, 8), [3, 3, 9, 12, 3, 3], [[1, 2], [2, 3], [4], [5, 6], [7], [8, 1]], fill(14, 6);
                                                 nchunks = 2, min_work = minwork)
        rng = Xoshiro(9)
        J0 = SchurBlockJacobian{Float32, Matrix{Float32}}(layout(0))
        data, res = randn(rng, length(J0.data)), randn(rng, length(J0.r))
        xs = map((typemax(Int), 0)) do mw
            ls = qrr_system(Float32, layout(mw), data, res, :qr)
            dampen!(ls, 0.1)
            @test solve!(ls, :qr) == Success
            copy(getx(ls))
        end
        @test xs[1] == xs[2]
    end
end

end # module
