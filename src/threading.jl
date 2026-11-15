# Threading helpers shared by the Schur and QR-elimination kernels.
#
# Threading needs Polyester (the mechanism NLLSModel.jl uses for assembly, so the two never compete for threads), a weak
# dependency: PolyesterExt adds the threaded method of `batch_foreach` and the methods of `threadargs` /
# `fromthreadargs`. Without Polyester loaded every kernel runs serially. With it, a loop is threaded only when more
# than one thread is available *and* there is enough work to be worth splitting.

# Declared here, methods in PolyesterExt (documented in docstrings.jl)
function threadargs end
function fromthreadargs end

# batch_foreach(f, n)
# `f(i)` for `i in 1:n`, threaded. PolyesterExt adds the threaded method, for `n::Int` (what every caller passes);
# without it this one runs the loop serially.
function batch_foreach(f::F, n::Integer) where F
    for i in 1:n
        f(i)
    end
    return nothing
end

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
