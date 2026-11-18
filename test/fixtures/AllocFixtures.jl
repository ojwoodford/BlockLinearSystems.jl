# Allocation helpers: the bound on a threaded call's allocation, warm measurement, the threading threshold, and the
# kernel wrappers that measure inside a function. Loaded once per session into Main by fixtures/load.jl; no testsets.
module AllocFixtures

using BlockLinearSystems, LinearAlgebra
using BlockLinearSystems: reset_schur_dense!, reset_schur_sparse!, reduce_schur_dense!, reduce_schur_sparse!, backsubstitute!, reduce_qr!, reduced_mul!, reduced_tmul!, reduced_residual!, reduced_jacobian!, reduced_normal_dense!, reduced_normal_sparse!, reduced_diagonal_blocks!, jtr!, jtj_diag!, jtj_quadform   # internal: the kernels, withdrawn from the exports by EL6

export threaded_allocs_ok, allocs_dot, warm_allocs, with_min_work, alloc_backsubstitute_qr, alloc_backsubstitute_schur, alloc_jtj_diag, alloc_jtj_quadform, alloc_jtr, alloc_reduce_qr, alloc_reduce_schur_dense, alloc_reduce_schur_sparse, alloc_reduced_diagonal_blocks, alloc_reduced_jacobian, alloc_reduced_mul, alloc_reduced_normal_dense, alloc_reduced_normal_sparse, alloc_reduced_residual, alloc_reduced_tmul, alloc_reset_schur_dense, alloc_reset_schur_sparse

# The kernels allocate nothing, serial or threaded, except a threaded call on Julia 1.10 (accepted 2026-11-15:
# rebuilding a nested closure inside the threaded loop allocates there, about 300 bytes per chunk), which is held to a
# bound per thread (3 072 bytes, rules.md 21, raised from 2 048 on 2026-11-16) so that a regression still shows.
# `threaded`: whether the call measured took the threaded path.
threaded_allocs_ok(bytes, threaded=Threads.nthreads() > 1) =
    bytes == 0 || (VERSION < v"1.11" && threaded && bytes <= 3072 * Threads.nthreads())

# Measured inside functions: @allocated at a testset's top level counts spurious boxing on Julia 1.10
alloc_reduce_qr(ws, J, sc, lambda) = @allocated reduce_qr!(ws, J, sc, lambda)
alloc_jtr(g, src) = @allocated jtr!(g, src)
alloc_jtj_diag(d, src) = @allocated jtj_diag!(d, src)
alloc_jtj_quadform(src, v) = @allocated jtj_quadform(src, v)
alloc_reduced_mul(y, ws, v) = @allocated reduced_mul!(y, ws, v)
alloc_reduced_tmul(g, ws, w) = @allocated reduced_tmul!(g, ws, w)
alloc_reduced_residual(rt, ws) = @allocated reduced_residual!(rt, ws)
alloc_reduced_jacobian(Jt, rt, ws) = @allocated reduced_jacobian!(Jt, rt, ws)
alloc_reduced_normal_dense(S, g, ws) = @allocated reduced_normal_dense!(S, g, ws)
alloc_reduced_normal_sparse(S, g, ws, pat) = @allocated reduced_normal_sparse!(S, g, ws, pat)
alloc_reduced_diagonal_blocks(D, ws) = @allocated reduced_diagonal_blocks!(D, ws)
alloc_backsubstitute_qr(x_e, x_r, ws) = @allocated backsubstitute!(x_e, x_r, ws)
alloc_reset_schur_dense(S, R, sc_r, lambda, ws) = @allocated reset_schur_dense!(S, R, sc_r, lambda, ws)
alloc_reset_schur_sparse(Sv, Rv, map, di, sc_r, lambda, ws) = @allocated reset_schur_sparse!(Sv, Rv, map, di, sc_r, lambda, ws)
alloc_reduce_schur_dense(S, g, b_r, b_e, sc_e, lambda, st, H, ws) =
    @allocated reduce_schur_dense!(S, g, b_r, b_e, sc_e, lambda, st, H.E_data, H.C_data, ws)
alloc_reduce_schur_sparse(S, g, b_r, b_e, sc_e, lambda, st, H, pat, ws) =
    @allocated reduce_schur_sparse!(S, g, b_r, b_e, sc_e, lambda, st, H.E_data, H.C_data,
                                    pat.block_colptr, pat.block_rowval, pat.block_row_offsets, pat.col_heights, ws)
alloc_backsubstitute_schur(x_e, x_r, st, H, ws) = @allocated backsubstitute!(x_e, x_r, st, H.C_data, ws)

# Warm the call, then the least of three measurements
warm_allocs(f, args...) = (f(args...); minimum(f(args...) for _ in 1:3))

with_min_work(f, minwork) = begin
    old = BlockLinearSystems.SCHUR_THREAD_MIN_WORK[]
    try
        BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = minwork
        f()
    finally
        BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = old
    end
end

# Measured inside a function: @allocated at a testset's top level counts spurious boxing on Julia 1.10
allocs_dot(x, H, y) = @allocated dot(x, H, y)

end # module AllocFixtures
