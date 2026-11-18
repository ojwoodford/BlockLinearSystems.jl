module DampedLinearSolveQREliminationTests

# The QR-eliminated systems of a SchurBlockJacobian (src/qr_linear_systems.jl) through
# DampedLinearSolve's contract. Moved from SecondOrderOptimization's test/qrelimination.jl; the tests that run
# its iterators stay there.

using DampedLinearSolve, BlockLinearSystems, Krylov, SparseArrays, LinearAlgebra, Static, Random, Test
using Polyester   # BlockLinearSystems' kernels thread only with Polyester loaded
using DampedLinearSolve: dampen!, zero!, Success, NotPositiveDefinite, check_linear_system
using DampedLinearSolve: getA, getb, getx, solve!
const SOO = DampedLinearSolve   # the contract the systems implement
const QREXT = BlockLinearSystems   # the linear systems are the package's own (EL5)

# TP3: QR elimination on the Jacobian path (a SchurBlockJacobian prototype), against the normal path's Schur
# system and a dense least-squares solve of the same problem

const TP3_SOLVERS = Dict(Matrix{Float64} => (:auto, :qr, :cholesky, :ldl, :lsmr),
                         SparseMatrixCSC{Float64, Int} => (:auto, :qr, :cholesky, :ldl, :lsmr))

# Three retained blocks; six eliminated blocks, one above D_STATIC (9) and, with `under`, one with fewer rows
# than dimensions; a free group of rows coupled to every retained block
function tp3_structure(; under = false, nchunks = 1)
    r_dims, e_dims = [6, 6, 6], [3, 3, 3, 9, 2, 1]
    couplings = [[1, 2], [1, 2, 3], [2, 3], [1, 3], [1, 2], [3]]
    e_rows = [8, 10, 6, 12, under ? 1 : 4, 3]
    return SchurJacobianStructure(r_dims, e_dims, couplings, e_rows; free_rows = [20], free_retained = [[1, 2, 3]], nchunks)
end

# Where each Jacobian entry and residual of the dense problem lives in the tables: (table index, dense row, dense
# column) for J, (table index, dense row) for r
function tp3_layout(jst)
    st = jst.schur
    jidx, jrow, jcol, ridx, rrow = Int[], Int[], Int[], Int[], Int[]
    row0 = 0
    for t in 1:jst.ntables
        M, C, off = jst.table_nrows[t], jst.table_ncols[t], jst.table_offsets[t]
        m = M - (t <= st.neblocks ? jst.table_top[t] : 0)
        blocks = Tuple{Vector{Int}, Vector{Int}}[]
        for u in jst.table_ptr[t]:jst.table_ptr[t+1]-1
            r = jst.entry_retained[u]
            dr = Int(st.r_dims[r])
            push!(blocks, (collect(st.boffsets_r[r] .+ (0:dr-1)), collect(jst.entry_col[u] .+ (0:dr-1))))
        end
        if t <= st.neblocks
            de = Int(st.e_dims[t])
            push!(blocks, (collect(st.nr .+ st.boffsets_e[t] .+ (0:de-1)), collect((C - de):(C - 1))))
        end
        for i in 1:m
            for (cols, tcols) in blocks, (c, tc) in zip(cols, tcols)
                push!(jidx, off + (tc - 1) * M + i - 1); push!(jrow, row0 + i); push!(jcol, c)
            end
            push!(ridx, off + (C - 1) * M + i - 1); push!(rrow, row0 + i)
        end
        row0 += m
    end
    return (; jidx, jrow, jcol, ridx, rrow, m = row0, n = st.nr + sum(st.e_dims))
end

# The problem r(x) = V (x + 0.3 sin x) − y, with V random inside the pattern: its Jacobian keeps the pattern
struct TP3Problem{L <: NamedTuple}
    jst::SchurJacobianStructure
    lay::L
    V::Matrix{Float64}
    y::Vector{Float64}
    u::Vector{Float64}
    rv::Vector{Float64}
end
function TP3Problem(jst; seed = 1)
    rng = Xoshiro(seed)
    lay = tp3_layout(jst)
    V = zeros(lay.m, lay.n)
    for k in eachindex(lay.jidx)
        V[lay.jrow[k], lay.jcol[k]] = randn(rng)
    end
    return TP3Problem(jst, lay, V, randn(rng, lay.m), zeros(lay.n), zeros(lay.m))
end
function tp3_residual!(p::TP3Problem, x)
    @inbounds for j in eachindex(p.u)
        p.u[j] = x[j] + 0.3 * sin(x[j])
    end
    @inbounds for i in eachindex(p.rv)
        p.rv[i] = -p.y[i]
    end
    @inbounds for j in eachindex(p.u), i in eachindex(p.rv)
        p.rv[i] += p.V[i, j] * p.u[j]
    end
    return p.rv
end
function tp3_cost(p::TP3Problem, x)
    r = tp3_residual!(p, x)
    total = 0.0
    @inbounds for i in eachindex(r)
        total += r[i]^2
    end
    return 0.5 * total
end
# resjac! for the tables, and for a dense J
function tp3_tables!(p::TP3Problem, J::SchurBlockJacobian, x)
    r = tp3_residual!(p, x)
    lay, D = p.lay, J.data
    @inbounds for k in eachindex(lay.jidx)
        c = lay.jcol[k]
        D[lay.jidx[k]] = p.V[lay.jrow[k], c] * (1 + 0.3 * cos(x[c]))
    end
    @inbounds for k in eachindex(lay.ridx)
        D[lay.ridx[k]] = r[lay.rrow[k]]
    end
    return tp3_cost(p, x)
end
function tp3_dense!(p::TP3Problem, rr, J::Matrix, x)
    rr .= tp3_residual!(p, x)
    @inbounds for j in axes(J, 2), i in axes(J, 1)
        J[i, j] = p.V[i, j] * (1 + 0.3 * cos(x[j]))
    end
    return tp3_cost(p, x)
end

# A system linearized at x
function tp3_system(p, ::Type{RT}, s, x) where RT
    ls = QREXT.QRSchurLinearSystem(SchurBlockJacobian{Float64, RT}(p.jst), s)
    zero!(ls)
    TallFill((r, J, x) -> tp3_tables!(p, J, x))(getb(ls), getA(ls), x)
    return ls
end

# The normal path's dense Schur system for the same JᵀJ and Jᵀr
function tp3_schur(p, x)
    st = p.jst.schur
    Jd = zeros(p.lay.m, p.lay.n); rd = zeros(p.lay.m)
    tp3_dense!(p, rd, Jd, x)
    clen = st.coffsets[end] + Int(st.r_dims[st.entry_retained[end]]) * Int(st.e_dims[end]) - 1
    H = SchurBlockHessian(st, zeros(st.nr, st.nr), zeros(sum(abs2, st.e_dims)), zeros(clen))
    SOO.conformance_fill!(H, Jd' * Jd)
    ls = SOO.makehessiansystem(H, static(:cholesky))
    getb(ls) .= Jd' * rd
    return ls, Jd, rd
end

# A warm threaded run allocates nothing, except on Julia 1.10, where each threaded BlockLinearSystems kernel call
# may allocate a few KB per thread (rules.md 21; `threaded_allocs_ok` in fixtures/AllocFixtures.jl). A run makes many such calls, so
# on 1.10 the bound is per run: 48 KB per thread, a starting value above the largest measured (2026-11-15, 4
# threads, nchunks = 4: 140 640 bytes for :lsmr, 35 160 per thread). 1.11 and later, and every serial run: zero.
const TP3_110_RUN_BYTES_PER_THREAD = 48 * 1024
tp3_threaded_run_ok(bytes) = bytes == 0 ||
    (VERSION < v"1.11" && Threads.nthreads() > 1 && bytes <= TP3_110_RUN_BYTES_PER_THREAD * Threads.nthreads())

tp3_solve!(ls, s, lambda, scaling) = (dampen!(ls, lambda, scaling); solve!(ls, static(s)))
# Allocations of one warm :qr solve, behind a function barrier (as TP1's tp1_allocations)
# (a literal static(:qr): `static(s)` of a Symbol variable is a dynamic call, which allocates once)
tp6_solve!(ls, lambda, scaling) = (dampen!(ls, lambda, scaling); solve!(ls, static(:qr)))
tp6_allocations(ls, lambda, scaling) = (tp6_solve!(ls, lambda, scaling); Base.@allocations tp6_solve!(ls, lambda, scaling))
tp6_fill_bytes(red, ws) = (reduced_jacobian!(red.Jt, red.rt, ws); @allocated reduced_jacobian!(red.Jt, red.rt, ws))

# Fills a tall system as an optimizer's linearization does (test only)
struct TallFill{F}
    f::F
end
function (t::TallFill)(b::AbstractVector, A, x)
    cost = t.f(DampedLinearSolve.getr(A), DampedLinearSolve.getJ(A), x)
    DampedLinearSolve.gradient!(b, A)
    DampedLinearSolve.markstale!(A)
    return cost
end

# LP1: the TP3 problem with the retained columns of each block scaled by powers of ten from 1e-2 to 1e2, so that
# the reduced Jacobian is badly scaled within each (coupled) retained block
function tp3_scaled_problem(; seed = 1)
    p = TP3Problem(tp3_structure(); seed)
    st = p.jst.schur
    for r in 1:st.nrblocks
        dr = Int(st.r_dims[r])
        for k in 1:dr
            p.V[:, st.boffsets_r[r] + k - 1] .*= exp10(-2 + 4 * (k - 1) / (dr - 1))
        end
    end
    return p
end
# (a literal static(:lsmr): `static(s)` of a Symbol variable is a dynamic call, which allocates)
lp1_solve!(ls, lambda, scaling) = (dampen!(ls, lambda, scaling); solve!(ls, static(:lsmr)))
lp1_bytes(ls, lambda, scaling) = (lp1_solve!(ls, lambda, scaling); @allocated lp1_solve!(ls, lambda, scaling))

@testset "qrelimination.jl" begin
    @testset "Steps equal the Schur system's: $RT, under-determined block = $under" for RT in keys(TP3_SOLVERS), under in (false, true)
        p = TP3Problem(tp3_structure(; under))
        x = 0.1 .* randn(Xoshiro(2), p.lay.n)
        lsS, Jd, rd = tp3_schur(p, x)
        sc = 0.5 .+ rand(Xoshiro(3), p.lay.n)
        for s in TP3_SOLVERS[RT]
            ls = tp3_system(p, RT, s, x)
            @test getb(ls) ≈ Jd' * rd rtol = 1e-12
            @test SOO.quadform(getA(ls), x) ≈ norm(Jd * x)^2 rtol = 1e-12
            @test SOO.max_diag(getA(ls)) ≈ maximum(sum(abs2, Jd; dims = 1)) rtol = 1e-12
            for (lambda, scaling) in ((0.0, nothing), (0.3, nothing), (0.3, sc))
                a = tp3_solve!(ls, s, lambda, scaling)
                if under && lambda == 0
                    # an under-determined block without damping is singular, as on the Schur path
                    @test a == NotPositiveDefinite
                    @test tp3_solve!(lsS, :cholesky, lambda, scaling) == NotPositiveDefinite
                    continue
                end
                @test a == Success
                @test tp3_solve!(lsS, :cholesky, lambda, scaling) == Success
                xref = (Jd' * Jd + lambda * Diagonal(something(scaling, ones(p.lay.n)))) \ (Jd' * rd)
                tol = s === :lsmr ? 1e-6 : 1e-10
                @test getx(ls) ≈ getx(lsS) rtol = tol
                @test getx(ls) ≈ xref rtol = tol
            end
        end
    end

    @testset "Meets the linear-system interface: $RT" for RT in keys(TP3_SOLVERS)
        jst = tp3_structure()
        n = jst.schur.nr + sum(jst.schur.e_dims)
        @test isnothing(check_linear_system(s -> QREXT.QRSchurLinearSystem(SchurBlockJacobian{Float64, RT}(jst), s); n, linsolvers = TP3_SOLVERS[RT], rtol = 1e-6))
    end

    @testset "LSMR on a badly scaled reduced Jacobian equals QR (LP1): $RT" for RT in keys(TP3_SOLVERS)
        p = tp3_scaled_problem()
        x = 0.1 .* randn(Xoshiro(2), p.lay.n)
        lsq = tp3_system(p, RT, :qr, x)
        lsl = tp3_system(p, RT, :lsmr, x)
        d = DampedLinearSolve.getdiag!(zeros(p.lay.n), lsq)   # a Marquardt scaling
        for (lambda, scaling) in ((0.0, nothing), (1e-8, nothing), (1e-4, nothing), (1e-8, d), (1e-4, d))
            @test tp3_solve!(lsq, :qr, lambda, scaling) == Success
            @test tp3_solve!(lsl, :lsmr, lambda, scaling) == Success
            @test getx(lsl) ≈ getx(lsq) rtol = 1e-6
        end
        @test lp1_bytes(lsl, 1e-4, d) == 0
    end
end

end # module DampedLinearSolveQREliminationTests
