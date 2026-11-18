using Test, LinearAlgebra, BlockLinearSystems
using BlockLinearSystems: SchurReductionWorkspace   # internal: the kernels, withdrawn from the exports by EL6
using Polyester   # BlockLinearSystems' kernels thread only with Polyester loaded
include("fixtures/load.jl")
using Main.SchurFixtures

# The thread-dependent testset of schur_problems.jl: with the threshold at zero, on more than one thread the Schur
# kernels take their threaded paths

@testset "Column chunks tile the retained blocks without overlap (M3)" begin
    p = standard_problem()
    st = schur_structure(p)
    for K in (1, 2, 4, 9)
        ws = SchurReductionWorkspace{Float64}(st, K)
        @test length(ws.col_chunks) == K
        @test reduce(vcat, collect.(ws.col_chunks)) == collect(1:st.nrblocks)
        # the retained-block-major index covers every coupling exactly once, in ascending eliminated block order
        @test sort(ws.retained_entry) == collect(1:length(st.entry_retained))
        for r in 1:st.nrblocks
            es = ws.retained_entry[ws.retained_ptr[r]:ws.retained_ptr[r+1]-1]
            @test all(st.entry_retained[es] .== r) && issorted(ws.entry_eliminated[es])
        end
    end
    # Forcing the threaded paths gives the same result
    old = BlockLinearSystems.SCHUR_THREAD_MIN_WORK[]
    try
        BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = 0
        x_ref = reference_solve(p, 0.4, ones(length(p.b)))
        for make in (dense_schur_hessian, sparse_schur_hessian)
            x, ok = schur_solve(make(p; nchunks=3), p.b, ones(length(p.b)), 0.4; nchunks=3)
            @test ok && norm(x - x_ref) <= 1e-10 * norm(x_ref)
        end
    finally
        BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = old
    end
end
