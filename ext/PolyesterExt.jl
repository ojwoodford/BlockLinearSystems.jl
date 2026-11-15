module PolyesterExt

import Polyester
import BlockLinearSystems: batch_foreach

# Kept out of line so that the serial path does not pay for the closure Polyester builds
@noinline function batch_foreach(f::F, n::Int) where F
    Polyester.@batch for i in 1:n
        f(i)
    end
    return nothing
end

# Arguments of a `@batch` loop, passed by address. Polyester hands the threads an isbits value as it is, but boxes any
# other captured value, one allocation per loop, even a mutable object or an array of mutable objects. `threadargs`
# rewrites a value of type `T` as an isbits one, the flat tuple of its leaves: walking through its immutable fields,
# each isbits field stays as it is, and each object held by reference (a mutable object, or the box in a field of
# abstract type) becomes its address. `fromthreadargs` rebuilds the value, of type `T`, inside the loop. The caller
# keeps the original alive with `GC.@preserve` for the duration of the loop.
# Both are generated as one expression for the whole type, and the leaves are a flat tuple, because inference widens
# deeply nested types (a recursion through the fields, or nested wrappers), and the widened code allocates. A field of
# an isbits union type (`Union{Nothing, Int}`) is stored inline, not boxed, so it is passed as it is (it allocates).
# `fromthreadargs` is not forced inline: inlined into the loop body, it took the first call of the threaded Schur
# linearization from 0.9 s to 18 s of compilation.
struct ThreadArgs{T, L<:Tuple}
    leaves::L
end
byvalue(T) = isbitstype(T) || Base.isbitsunion(T)
byaddress(T) = !byvalue(T) && (!isconcretetype(T) || ismutabletype(T))
function threadargs_leaves!(leaves, types, T, ex)
    if byvalue(T)
        push!(leaves, ex); push!(types, T)
    elseif byaddress(T)
        push!(leaves, :(ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), $ex))); push!(types, Ptr{Cvoid})
    else
        foreach(i -> threadargs_leaves!(leaves, types, fieldtype(T, i), :(getfield($ex, $i))), 1:fieldcount(T))
    end
    return leaves, types
end
function fromthreadargs_expr(T, ex, next::Ref{Int})
    if byvalue(T) || byaddress(T)
        leaf = :(getfield(getfield($ex, :leaves), $(next[] += 1)))
        return byvalue(T) ? leaf : :(unsafe_pointer_to_objref($leaf)::$T)
    end
    fields = [fromthreadargs_expr(fieldtype(T, i), ex, next) for i in 1:fieldcount(T)]
    return T <: Tuple ? :($(Expr(:tuple, fields...))::$T) : Expr(:new, T, fields...)
end
@generated function threadargs(x)
    leaves, types = threadargs_leaves!([], [], x, :x)
    return Expr(:block, Expr(:meta, :inline), :(ThreadArgs{$x, Tuple{$(types...)}}($(Expr(:tuple, leaves...)))))
end
@generated fromthreadargs(a::ThreadArgs{T}) where T = fromthreadargs_expr(T, :a, Ref(0))

end # module PolyesterExt
