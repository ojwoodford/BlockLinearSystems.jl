using Test

# Starts a second Julia process: a cold start and package load, so it is in the `process` group.

# `using BlockLinearSystems` alone provides DampedLinearSolve's contract for an eliminated system: DampedLinearSolve is
# loaded with the package (a dependency, not an extension), and a system laid out, filled and solved through the public
# API gives the dense solution, on both paths. The Krylov extension loads with Krylov, and only then.
const FRESH_LOAD = raw"""
using BlockLinearSystems, LinearAlgebra
any(m -> nameof(m) === :DampedLinearSolve, values(Base.loaded_modules)) || error("DampedLinearSolve not loaded")
Base.get_extension(BlockLinearSystems, :KrylovExt) === nothing || error("KrylovExt loaded without Krylov")
import DampedLinearSolve as DLS   # already loaded: this only names it
# Blocks 1 and 3 eliminated (3 each), 2 and 4 retained (6 each); one row block per observation, and a prior on block 2
bs, el = [3, 6, 3, 6], [true, false, true, false]
rbs, rows = [[1, 2], [1, 4], [2, 3], [3, 4], [2]], [2, 2, 2, 2, 6]
jst = SchurJacobianStructure(bs, rbs, rows; eliminated=el)
st = SchurStructure(jst)
lay = invperm(blockorder(jst))                 # layout number of each of the caller's blocks
off = computestartindices(blocksizes(jst)); n = sum(blocksizes(jst))
J = SchurBlockJacobian{Float64, Matrix{Float64}}(jst)
Jd, rd = zeros(sum(rows), n), zeros(sum(rows))
r0 = 0
for (k, rb) in enumerate(rbs)
    m = rows[k]
    for b in rb
        j = lay[b]; d = bs[b]
        X = [sin(i + 7k + 3j + 11c) for i in 1:m, c in 1:d]
        block(J, k, blockentry(jst, k, j), m, d) .= X
        Jd[r0 .+ (1:m), off[j] .+ (0:d-1)] .= X
    end
    residual_block(J, k) .= cos.(r0 .+ (1:m))
    rd[r0 .+ (1:m)] .= cos.(r0 .+ (1:m))
    global r0 += m
end
lambda = 0.1
x_ref = (Jd' * Jd + lambda * I) \ (Jd' * rd)
ls = DLS.makejacobiansystem(J, n, :auto)
DLS.getb(ls) .= Jd' * rd
DLS.dampen!(ls, lambda, ones(n))
DLS.solve!(ls, :auto) == DLS.Success || error("Jacobian path: solve failed")
norm(DLS.getx(ls) - x_ref) <= 1e-10 * norm(x_ref) || error("Jacobian path: wrong solution")
# The Hessian path, on the same problem: H = JᵀJ added block by block
H = SchurBlockHessian{Float64, Matrix{Float64}}(st)
A = Jd' * Jd
rg(j) = off[j] .+ (0:blocksizes(st)[j]-1)
for i in 1:nblocks(st), j in 1:i
    if i != j
        (i > nretained(st) && j > nretained(st)) && continue   # two eliminated blocks: no storage
        iszero(A[rg(i), rg(j)]) && continue                    # no row block touches both
    end
    addblock!(H, i, j, A, rg(i), rg(j))
end
lh = DLS.makehessiansystem(H, :auto)
DLS.getb(lh) .= Jd' * rd
DLS.dampen!(lh, lambda, ones(n))
DLS.solve!(lh, :auto) == DLS.Success || error("Hessian path: solve failed")
norm(DLS.getx(lh) - x_ref) <= 1e-10 * norm(x_ref) || error("Hessian path: wrong solution")
@eval using Krylov
Base.get_extension(BlockLinearSystems, :KrylovExt) === nothing && error("KrylovExt not loaded with Krylov")
print("ok")
"""

@testset "using BlockLinearSystems alone solves an eliminated system; Krylov's extension loads with Krylov (EL5)" begin
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $FRESH_LOAD`
    @test readchomp(cmd) == "ok"
end
