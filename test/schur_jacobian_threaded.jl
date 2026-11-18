using Test, StaticArrays, Static, BlockLinearSystems
using BlockLinearSystems: eliminated_block, coupling_block   # internal: withdrawn from the exports by EL4
using BlockLinearSystems: table_residual_block   # internal: the table-indexed residual view
include("fixtures/load.jl")
using Main.QRFixtures

# The thread-dependent testset of schur_jacobian.jl: tables filled from different threads

@testset "Filling different tables from different threads is race-free (E1)" begin
    p = qr_problems(Float64)[1]
    J = p.J
    jst = J.structure
    function fill_table!(J, t)
        jst = J.structure
        m = t <= jst.schur.neblocks ? jst.e_rows[t] : jst.free_rows[t - jst.schur.neblocks]
        for row in 1:2:m - 1                        # two rows per residual, as a caller would write them
            for u in jst.table_ptr[t]:jst.table_ptr[t+1]-1
                coupling_block(J, u, row, static(2), static(6)) .= SMatrix{2, 6}(ntuple(i -> sin(i + 7row + 13u), 12))
            end
            t <= jst.schur.neblocks && (eliminated_block(J, t, row, static(2), static(3)) .= SMatrix{2, 3}(ntuple(i -> cos(i + row + t), 6)))
            table_residual_block(J, t, row, static(2)) .= SVector(row + t, row - t)
        end
    end
    serial = copy_structure(J)
    for t in 1:jst.ntables
        fill_table!(serial, t)
    end
    threaded = copy_structure(J)
    Threads.@threads for t in 1:jst.ntables
        fill_table!(threaded, t)
    end
    @test threaded.data == serial.data
    @test count(!iszero, serial.data) > 0.8 * sum(jst.e_rows[e] * jst.table_ncols[e] for e in 1:jst.schur.neblocks)
end
