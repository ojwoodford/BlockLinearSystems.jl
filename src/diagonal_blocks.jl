# Kernels on a flat array of square diagonal blocks: block b, of dimension d_b, is stored column-major at the
# offset that the blocks before it take (as `reduced_diagonal_blocks!` writes them). Plain loops, single-threaded,
# allocation free: the blocks are small (rules.md 23).

# factor_diagonal_blocks!(D, dims) -> Bool: the Cholesky factor L of each block, in place in its lower triangle
function factor_diagonal_blocks!(D::AbstractVector{T}, dims) where T
    o = 0
    @inbounds for db in dims
        d = Int(db)
        for j in 1:d
            s = D[o + (j - 1) * d + j]
            for k in 1:j-1
                l = D[o + (k - 1) * d + j]
                s -= l * l
            end
            (s > zero(T) && isfinite(s)) || return false
            ljj = sqrt(s)
            D[o + (j - 1) * d + j] = ljj
            for i in j+1:d
                t = D[o + (j - 1) * d + i]
                for k in 1:j-1
                    t -= D[o + (k - 1) * d + i] * D[o + (k - 1) * d + j]
                end
                D[o + (j - 1) * d + i] = t / ljj
            end
        end
        o += d * d
    end
    return true
end

# solve_diagonal_blocks!(y, D, dims, x) -> y: y = (L Lᵀ)⁻¹ x, block by block, with D as factored above; y may be x
function solve_diagonal_blocks!(y::AbstractVector, D::AbstractVector, dims, x::AbstractVector)
    o = 0
    p = 0
    @inbounds for db in dims
        d = Int(db)
        if y !== x
            for i in 1:d
                y[p + i] = x[p + i]
            end
        end
        for j in 1:d
            yj = y[p + j] / D[o + (j - 1) * d + j]
            y[p + j] = yj
            for i in j+1:d
                y[p + i] -= D[o + (j - 1) * d + i] * yj
            end
        end
        for j in d:-1:1
            t = y[p + j]
            for i in j+1:d
                t -= D[o + (j - 1) * d + i] * y[p + i]
            end
            y[p + j] = t / D[o + (j - 1) * d + j]
        end
        o += d * d
        p += d
    end
    return y
end
