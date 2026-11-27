# Precompile workload (improvements.md item 3): the Schur-eliminated Hessian systems (dense and sparse reduced) and the
# QR-eliminated Jacobian systems, laid out, filled through the public API and solved once, on small fixed data, in Float64.
# Plain functions, because the workload is run twice: here, for a session without Polyester, and by PolyesterExt, whose
# threaded method of `batch_foreach` replaces the serial one in everything compiled here.
const PRECOMPILE_SIZES = [6, 6, 3, 3, 3]
const PRECOMPILE_ROWBLOCKS = [[3, 1, 2], [4, 1, 2], [5, 2]]
const PRECOMPILE_ELIMINATED = [false, false, true, true, true]
function precompile_hessian(::Type{R}) where R
    st = SchurStructure(PRECOMPILE_SIZES, PRECOMPILE_ROWBLOCKS; eliminated = PRECOMPILE_ELIMINATED)
    sizes = blocksizes(st); off = computestartindices(sizes); n = sum(sizes)
    lay = invperm(blockorder(st))
    rg(p) = off[p]:off[p] + sizes[p] - 1
    B = Matrix{Float64}(4.0 * I, n, n)
    for rb in PRECOMPILE_ROWBLOCKS, r in rb[2:end]
        B[rg(lay[r]), rg(lay[rb[1]])] .= 0.1; B[rg(lay[rb[1]]), rg(lay[r])] .= 0.1
    end
    H = SchurBlockHessian{Float64, R}(st)
    foreach(p -> addblock!(H, p, p, B, rg(p), rg(p)), 1:nblocks(st))
    for rb in PRECOMPILE_ROWBLOCKS, r in rb[2:end]
        addblock!(H, lay[r], lay[rb[1]], B, rg(lay[r]), rg(lay[rb[1]]))
    end
    precompile_solve(makehessiansystem(H, :auto), n)
    return nothing
end
function precompile_jacobian(::Type{R}) where R
    rows = [4, 4, 3]
    jst = SchurJacobianStructure(PRECOMPILE_SIZES, PRECOMPILE_ROWBLOCKS, rows; eliminated = PRECOMPILE_ELIMINATED)
    J = SchurBlockJacobian{Float64, R}(jst)
    sizes, lay = blocksizes(jst), invperm(blockorder(jst))
    for (k, rb) in enumerate(PRECOMPILE_ROWBLOCKS)
        for b in rb
            X = block(J, k, blockentry(jst, k, lay[b]), rows[k], sizes[lay[b]])
            X .= [sin(0.3 * i + 0.7 * j + k + b) + (i == j ? 2.0 : 0.0) for i in axes(X, 1), j in axes(X, 2)]
        end
        residual_block(J, k) .= 0.5
    end
    n = sum(sizes)
    for s in (:auto, :cholesky)
        ls = DampedLinearSolve.makejacobiansystem(J, n, s)
        DampedLinearSolve.gradient!(getb(ls), getA(ls))
        markstale!(getA(ls))
        precompile_solve(ls, n, s)
    end
    return nothing
end
# The calls an optimizer makes on a system during one iteration and its trial
function precompile_solve(ls, n, s = :auto)
    getb(ls) .= 1.0
    trial = DampedLinearSolve.copy_structure(ls)
    for sys in (ls, sharex(trial, ls))
        dampen!(sys, 0.1)
        solve!(sys, s)
        dampen!(sys, 0.1, ones(n))
        solve!(sys, static(s))
        max_diag(sys); DampedLinearSolve.dampingfloor(sys)
        DampedLinearSolve.quadratic_model_change(sys, getx(sys))
        DampedLinearSolve.zero!(sys)
    end
    return nothing
end
function precompile_workload()
    precompile_hessian(Matrix{Float64})
    precompile_hessian(SparseBlockLinearSystem{Float64})
    precompile_jacobian(Matrix{Float64})
    precompile_jacobian(SparseMatrixCSC{Float64, Int})
    return nothing
end

@compile_workload precompile_workload()
