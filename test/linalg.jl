@testset "dot(v, H, v) equals the full quadratic form (B6)" begin
    for p in (standard_problem(), coupled_groups_problem())
        n = length(p.b)
        dx = randn(Random.Xoshiro(5), n)
        expected = dot(dx, p.H * dx)
        for kind in (:dense, :sparse)
            H = kind === :dense ? dense_schur_hessian(p) : sparse_schur_hessian(p)
            @test dot(dx, H, dx) ≈ expected rtol = 1e-12
            # Type stable (a captured `Type` was once widened to DataType on Julia 1.10, which made the
            # result `Any` and boxed it on every call) and allocation free, for both kinds of R
            @test @inferred(dot(dx, H, dx)) isa Float64
            dx32 = Float32.(dx)
            @test @inferred(dot(dx32, H, dx32)) isa Float64
            allocs_dot(dx, H, dx)   # compile
            @test allocs_dot(dx, H, dx) == 0
        end
    end
end

@testset "dot(v, H, v): any real element type, and eliminated blocks of every size class (M13)" begin
    for p in (standard_problem(), mixed_dims_problem())
        n = length(p.b)
        dx = randn(Random.Xoshiro(8), n)
        expected = dot(dx, p.H * dx)
        for make in (dense_schur_hessian, sparse_schur_hessian)
            H = make(p)
            @test dot(dx, H, dx) ≈ expected rtol = 1e-12
            vdx = view(dx, 1:n)
            @test dot(vdx, H, vdx) ≈ expected rtol = 1e-12                    # AbstractVector
            dx32 = Float32.(dx)
            @test dot(dx32, H, dx32) ≈ dot(dx32, p.H * dx32) rtol = 1e-4
            @test dot(dx32, H, dx32) isa Float64                              # promoted element type
        end
    end
end

@testset "dot(x, H, y) is the bilinear form xᵀHy, for any x and y (D1)" begin
    rng = Random.Xoshiro(31)
    # mixed_dims_problem has eliminated blocks on both sides of D_STATIC (static and run-time kernels);
    # coupled_groups_problem has retained-retained blocks with no eliminated variable
    @test any(>(BlockLinearSystems.D_STATIC), mixed_dims_problem().e_dims)
    @test any(<=(BlockLinearSystems.D_STATIC), mixed_dims_problem().e_dims)
    for p in (standard_problem(), coupled_groups_problem(), mixed_dims_problem())
        n = length(p.b)
        for make in (dense_schur_hessian, sparse_schur_hessian)
            H = make(p)
            x, y = randn(rng, n), randn(rng, n)
            tol = 1e-12 * opnorm(p.H) * norm(x) * norm(y)
            # Against the dense form, for independent x and y
            @test abs(dot(x, H, y) - x' * p.H * y) <= tol
            # Symmetric
            @test abs(dot(x, H, y) - dot(y, H, x)) <= tol
            # The fast path is taken for x === y, and not for an equal copy; both give the quadratic form
            xc = copy(x)
            @test dot(x, H, x) === BlockLinearSystems.schur_quadform(H, x)
            @test dot(x, H, xc) === BlockLinearSystems.schur_bilinear(H, x, xc)
            @test dot(x, H, x) ≈ dot(x, H, xc) rtol = 1e-12
            @test dot(x, H, x) ≈ dot(x, p.H * x) rtol = 1e-12
            # Mixed element types promote
            x32, yi = Float32.(x), rand(rng, -3:3, n)
            @test @inferred(dot(x32, H, yi)) isa Float64
            @test abs(dot(x32, H, yi) - Float64.(x32)' * p.H * yi) <= tol * norm(yi) / norm(y)
            @test @inferred(dot(x32, H, x32)) isa Float64
            # Views
            xbig = [0.0; x; 0.0]
            @test dot(view(xbig, 2:n+1), H, y) ≈ dot(x, H, y) rtol = 1e-14
            @test dot(view(xbig, 2:n+1), H, view(xbig, 2:n+1)) ≈ dot(x, H, x) rtol = 1e-14
            # Type stable and allocation free
            @test @inferred(dot(x, H, y)) isa Float64
            allocs_dot(x, H, y)   # compile
            @test allocs_dot(x, H, y) == 0
            # Wrong lengths are rejected; complex vectors are not supported
            @test_throws DimensionMismatch dot(x, H, randn(rng, n + 1))
            @test_throws DimensionMismatch dot(randn(rng, n - 1), H, y)
            @test_throws DimensionMismatch dot(x[1:n-1], H, x[1:n-1])
            @test_throws MethodError dot(ComplexF64.(x), H, y)
        end
    end
    # A Float32 H with Float64 vectors promotes too
    p = mixed_dims_problem()
    n = length(p.b)
    H = sparse_schur_hessian(p)
    R32 = SparseBlockLinearSystem{Float32}(Float32.(H.R.data), H.R.indices, H.R.indicestransposed, H.R.rowblocksizes, H.R.columnblocksizes, H.R.m, H.R.n)
    H32 = SchurBlockHessian(H.structure, R32, Float32.(H.E_data), Float32.(H.C_data))
    Hd32 = SchurBlockHessian(H.structure, Float32.(dense_schur_hessian(p).R), Float32.(H.E_data), Float32.(H.C_data))
    x, y = randn(rng, n), randn(rng, n)
    # The quadratic form accumulates in the promoted type (AB4): type stable, and equal to rounding to the Float64
    # result of the same (Float32-valued) matrix
    R64 = SparseBlockLinearSystem{Float64}(Float64.(R32.data), H.R.indices, H.R.indicestransposed, H.R.rowblocksizes, H.R.columnblocksizes, H.R.m, H.R.n)
    H64 = SchurBlockHessian(H.structure, R64, Float64.(H32.E_data), Float64.(H32.C_data))
    Hd64 = SchurBlockHessian(H.structure, Float64.(Hd32.R), Float64.(H32.E_data), Float64.(H32.C_data))
    for (Hk, Hk64) in ((H32, H64), (Hd32, Hd64))
        @test @inferred(dot(x, Hk, y)) isa Float64
        @test dot(x, Hk, y) ≈ dot(x, H, y) rtol = 1e-5
        @test dot(x, Hk, x) ≈ dot(x, H, x) rtol = 1e-5
        @test @inferred(BlockLinearSystems.schur_quadform(Hk, x)) isa Float64
        @test BlockLinearSystems.schur_quadform(Hk, x) ≈ BlockLinearSystems.schur_quadform(Hk64, x) rtol = 1e-13
        # No accumulator changes type inside the loop (the result above is inferred even when one does, since the
        # union is split, so the slot types are what show it)
        ci = first(code_typed(BlockLinearSystems.schur_quadform, (typeof(Hk), Vector{Float64}); optimize=false))[1]
        @test !any(t -> t isa Union && Float32 <: t, ci.slottypes)
    end
    @test !isdefined(BlockLinearSystems, :fast_bAb)
end
