# Helpers shared by the linear systems of a SchurBlockHessian and a SchurBlockJacobian (DampedLinearSolve's contract)

# A solver is named by a Symbol or a StaticSymbol
linsolver_symbol(s::Symbol) = s
linsolver_symbol(s::StaticSymbol) = dynamic(s)

# Indefiniteness of an LDLᵀ factorization from its pivots: a pivot below -rtol times the largest pivot magnitude
# (the test DampedLinearSolve applies to its own LDLᵀ). rtol is about the precision of the factorization: 1e-6 for
# Float64, eps(Float32)^(3/8) ≈ 2.5e-3 for Float32 (the rule of single-precision.md; a starting value).
indefinite_rtol(::Type{Float64}) = 1e-6
indefinite_rtol(::Type{T}) where T = Float64(eps(T))^(3 / 8)
function is_indefinite(F::LDLFactorizations.LDLFactorization, rtol::Float64=indefinite_rtol(eltype(F.d)))
    maxabs = 0.0
    minpivot = Inf
    @inbounds for d in F.d
        maxabs = max(maxabs, abs(d))
        minpivot = min(minpivot, d)
    end
    return minpivot < -rtol * maxabs
end
