using Test

# Starts a second Julia process: a cold start and package load, so it has a test group of its own (`process`)

# In a fresh session on 2 threads: without Polyester the kernels run serially (and give the serial result); once it is
# loaded, the very first threaded call of the session works and gives the same result.
const FRESH_SESSION = raw"""
using BlockLinearSystems, Random
using BlockLinearSystems: QRReductionWorkspace, reduce_qr!   # internal: the QR kernels
const BLS = BlockLinearSystems
rng = Random.Xoshiro(3)
layout(minwork) = SchurJacobianStructure(fill(6, 8), [3, 3, 9, 12, 3, 3], [[1, 2], [2, 3], [4], [5, 6], [7], [8, 1]], fill(14, 6);
                                         nchunks=2, min_work=minwork)
J0 = SchurBlockJacobian{Float64, Matrix{Float64}}(layout(0))
data, res = randn(rng, length(J0.data)), randn(rng, length(J0.r))
sc = ones(sum(Int, layout(0).schur.e_dims))
function factored(minwork)
    J = SchurBlockJacobian{Float64, Matrix{Float64}}(copy(data), copy(res), layout(minwork))
    ws = QRReductionWorkspace(J)
    reduce_qr!(ws, J, sc, 0.1) || error("reduce_qr! failed")
    return copy(ws.factored)
end
Base.get_extension(BLS, :PolyesterExt) === nothing || error("PolyesterExt loaded without Polyester")
serial = factored(typemax(Int))
factored(0) == serial || error("without Polyester: differs from serial")
@eval using Polyester
Base.get_extension(BLS, :PolyesterExt) === nothing && error("PolyesterExt not loaded")
factored(0) == serial || error("first threaded call: differs from serial")
print("ok")
"""

@testset "Threading needs Polyester; the first threaded call of a session works (R2)" begin
    cmd = `$(Base.julia_cmd()) --startup-file=no --threads=2 --project=$(Base.active_project()) -e $FRESH_SESSION`
    @test readchomp(cmd) == "ok"
end
