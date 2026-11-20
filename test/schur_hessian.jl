using Test, BlockLinearSystems, Random, SparseArrays
include(joinpath(@__DIR__, "fixtures", "load.jl"))
using Main.SchurFixtures

# Storage from a structure (EL2): SchurBlockHessian{T, RType}(structure; pattern) and SchurBlockHessian(R, like)

same_storage_sizes(H, ref) = size(H.R) == size(ref.R) && length(H.E_data) == length(ref.E_data) && length(H.C_data) == length(ref.C_data)
allzero(H) = all(iszero, H.R isa Matrix ? H.R : H.R.data) && all(iszero, H.E_data) && all(iszero, H.C_data)

# Measured inside a function: @allocated at a testset's top level counts spurious boxing on Julia 1.10
alloc_like(R, like) = (SchurBlockHessian(R, like); @allocated SchurBlockHessian(R, like))

@testset "Storage from a structure (EL2)" begin
    problems = (("standard", standard_problem(1)), ("mixed dimensions", mixed_dims_problem()), ("gauge", gauge_problem()))
    @testset "$name, $T" for (name, p) in problems, T in (Float64, Float32)
        st = schur_structure(p)
        # Dense retained part: the sizes of the hand-built system, all zero, no pattern unless given
        ref = dense_schur_hessian(p)
        H = SchurBlockHessian{T, Matrix{T}}(st)
        @test H isa SchurBlockHessian{T, Matrix{T}}
        @test same_storage_sizes(H, ref) && allzero(H)
        @test H.structure === st && H.pattern === nothing
        pat = merge(build_schur_pattern(st, T), (S_factor = :attached,))
        @test SchurBlockHessian{T, Matrix{T}}(st; pattern=pat).pattern === pat
        # Sparse retained part: the block pattern of the reduced system, kept in `pattern`
        sref = sparse_schur_hessian(p)
        Hs = SchurBlockHessian{T, SparseBlockLinearSystem{T}}(st)
        @test Hs isa SchurBlockHessian{T, SparseBlockLinearSystem{T}}
        @test same_storage_sizes(Hs, sref) && allzero(Hs)
        @test Hs.R.indicestransposed.colptr == sref.R.indicestransposed.colptr && Hs.R.indicestransposed.rowval == sref.R.indicestransposed.rowval
        @test Hs.R.rowblocksizes == st.r_dims && Hs.R.columnblocksizes == st.r_dims
        @test Hs.pattern !== nothing && Hs.pattern.block_pattern == build_schur_pattern(st, T).block_pattern
        # ... and on a given pattern, which it keeps, entries attached by the caller included
        Hp = SchurBlockHessian{T, SparseBlockLinearSystem{T}}(st; pattern=pat)
        @test Hp.pattern === pat && Hp.pattern.S_factor === :attached
        @test Hp.R.indicestransposed == Hs.R.indicestransposed
    end

    @testset "A structure with retained pairs of its own" begin
        # Retained blocks 1 and 2 share a row block but no eliminated block: the sparse retained part stores the pair
        st = SchurStructure([6, 6, 3, 3], [[1, 3], [2, 4], [1, 2]]; eliminated=[false, false, true, true])
        Hs = SchurBlockHessian{Float64, SparseBlockLinearSystem{Float64}}(st)
        @test BlockLinearSystems.validblock(Hs.R, 2, 1)
        @test length(Hs.R.data) == 3 * 36 && length(Hs.E_data) == 18 && length(Hs.C_data) == 36
    end

    @testset "An unsupported retained storage" begin
        st = schur_structure(standard_problem(1))
        msg = "Retained storage SparseMatrixCSC{Float64, Int64} is not supported; use Matrix{Float64} or SparseBlockLinearSystem{Float64}"
        @test_throws ArgumentError(msg) SchurBlockHessian{Float64, SparseMatrixCSC{Float64, Int}}(st)
        @test_throws ArgumentError SchurBlockHessian{Float64, Matrix{Float32}}(st)
        @test_throws ArgumentError SchurBlockHessian{Float64, DenseBlockLinearSystem{Float64}}(st)
    end

    @testset "SchurBlockHessian(R, like): $kind" for kind in (:dense, :sparse)
        st = schur_structure(standard_problem(1))
        like = kind === :dense ? SchurBlockHessian{Float64, Matrix{Float64}}(st) :
                                 SchurBlockHessian{Float64, SparseBlockLinearSystem{Float64}}(st)
        R = kind === :dense ? zeros(size(like.R)) : copy_structure(like.R)
        H = SchurBlockHessian(R, like)
        @test H isa typeof(like)
        @test H.R === R && H.E_data === like.E_data && H.C_data === like.C_data
        @test H.structure === like.structure && H.pattern === like.pattern
        # It writes into R, not into like.R
        Rdata = kind === :dense ? R : R.data
        fill!(Rdata, 1.0)
        @test all(iszero, kind === :dense ? like.R : like.R.data)
        @test alloc_like(R, like) == 0
        # A sparse R of the same layout that does not share like's index arrays is accepted
        if kind === :sparse
            R2 = SparseBlockLinearSystem{Float64}(zeros(length(R.data)), copy(R.indices), copy(R.indicestransposed),
                                                  R.rowblocksizes, R.columnblocksizes, R.m, R.n)
            @test SchurBlockHessian(R2, like).R === R2
        end
        # A wrong R
        @test_throws DimensionMismatch SchurBlockHessian(kind === :dense ? zeros(3, 3) : copy_structure(SchurBlockHessian{Float64, SparseBlockLinearSystem{Float64}}(schur_structure(mixed_dims_problem())).R), like)
        @test_throws DimensionMismatch SchurBlockHessian(kind === :dense ? copy_structure(SchurBlockHessian{Float64, SparseBlockLinearSystem{Float64}}(st).R) : zeros(size(like.R)), like)
        @test_throws DimensionMismatch SchurBlockHessian(kind === :dense ? zeros(Float32, size(like.R)) : zeros(3, 3), like)
    end
end
