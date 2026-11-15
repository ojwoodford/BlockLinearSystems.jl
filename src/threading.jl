# Threading helpers shared by the Schur and QR-elimination kernels.

# The threaded loop; PolyesterExt adds its method.
function batch_foreach end

# Threading helper. Polyester (the same mechanism NLLSModel.jl uses for assembly, so the two never compete
# for threads) has low per-call overhead, but loops still run serially unless more than one thread is
# available *and* there is enough work to be worth splitting.
const SCHUR_THREAD_MIN_WORK = Ref(4096)

@inline use_threads(nitems::Integer, work::Integer=nitems) = nitems > 1 && work >= SCHUR_THREAD_MIN_WORK[] && Threads.nthreads() > 1

@inline function foreach_index(f::F, n::Int, threaded::Bool) where F
    if threaded
        batch_foreach(f, n)
    else
        for i in 1:n
            f(i)
        end
    end
    return nothing
end
