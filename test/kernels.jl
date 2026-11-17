using BlockLinearSystems: inv_sym_block   # internal names

@testset "inv_sym_block in Float32 and Float64 (M4)" begin
    H2 = SMatrix{2, 2, Float32, 4}(1, 1, 1, 1 + 1f-6)
    @test !inv_sym_block(H2, 0f0, SVector{2, Float32}(1, 1)).success
    @test inv_sym_block(SMatrix{2, 2, Float64, 4}(1, 1, 1, 1 + 1e-6), 0.0, SVector{2, Float64}(1, 1)).success
end

# The closed forms for sizes 1-3 accept exactly what the Cholesky path (sizes >= 4) accepts (AB1)
@testset "inv_sym_block closed forms agree with the Cholesky path (AB1)" begin
    # The generic (Cholesky) method, called directly at sizes 1-3
    generic = Base.tuple_type_tail(which(inv_sym_block, Tuple{SMatrix{4, 4, Float64, 16}, Float64, SVector{4, Float64}}).sig)
    chol_inv(H, lambda, scale) = invoke(inv_sym_block, generic, H, lambda, scale)
    sm(A) = SMatrix{size(A)..., Float64}(A)
    nodamp(D) = (0.0, zeros(SVector{D, Float64}))

    # Not positive definite, yet with a positive determinant (or a positive 1x1 entry test that is too loose)
    for A in ([-2.0 0; 0 -2], [-2.0 0 0; 0 -2 0; 0 0 2], [1.0 2 0; 2 1 0; 0 0 -1],
              fill(-1.0, 1, 1), -Matrix(1.0I, 2, 2), -Matrix(1.0I, 3, 3))
        D = size(A, 1)
        @test !inv_sym_block(sm(A), nodamp(D)...).success
        @test !chol_inv(sm(A), nodamp(D)...).success
    end

    rng = Random.MersenneTwister(1)
    for D in 1:3, T in (Float64, Float32), kind in (:pd, :indefinite, :nd, :nearsingular), damped in (false, true)
        agree = true; close = true
        for _ in 1:50
            Q = qr(randn(rng, D, D)).Q * Matrix(I, D, D)
            ev = kind === :pd ? rand(rng, D) .+ 0.1 :
                 kind === :nd ? -(rand(rng, D) .+ 0.1) :
                 kind === :indefinite ? (v = randn(rng, D); v[1] = abs(v[1]) + 0.1; v[end] = -abs(v[end]) - 0.1; v) :
                 (v = rand(rng, D) .+ 0.1; v[1] = rand(rng, (1e-3, 1e-8, 1e-16, 0.0, -1e-16)); v)
            A = Q * Diagonal(ev) * Q'
            H = SMatrix{D, D, T}((A + A') / 2)
            lambda = damped ? T(rand(rng, (1e-6, 1e-2, 1.0))) : zero(T)
            scale = damped ? SVector{D, T}(abs.(diag(H))) : zeros(SVector{D, T})
            closed = inv_sym_block(H, lambda, scale)
            chol = chol_inv(H, lambda, scale)
            agree &= closed.success == chol.success
            if closed.success && chol.success
                close &= isapprox(closed.invH, chol.invH; rtol = sqrt(eps(T)) * 1e3)
            end
        end
        @test agree
        @test close
    end
end
