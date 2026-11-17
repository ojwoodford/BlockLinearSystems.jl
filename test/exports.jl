using Test, BlockLinearSystems, LinearAlgebra

# The public API rule: a name is either public (exported, with a docstring and an entry in the API reference)
# or internal (not exported, not in the API reference). Adding or dropping an export must change this list.
const PUBLIC_API = sort([
    :BlockRLE, :DenseBlockLinearSystem, :QRReductionWorkspace, :SchurBlockJacobian,
    :SchurBlockHessian, :SchurJacobianStructure, :SchurReductionWorkspace, :SchurStructure,
    :SparseBlockLinearSystem, :TransposedPattern,
    :backsubstitute!, :block, :block_sparse_nnz, :blockrange, :blocksizes, :build_r_to_s_map,
    :build_reduced_jacobian_pattern, :build_schur_pattern, :copy_structure, :coupling_block, :coupling_entry, :computestartindices, :eliminated_block, :find_diag_inds,
    :factor_diagonal_blocks!, :free_block_pairs, :jtj_diag!, :jtj_quadform, :jtr!, :nblocks, :reduce_qr!, :reduce_schur_dense!,
    :reduce_schur_sparse!, :reduced_diagonal_blocks!, :reduced_jacobian!, :reduced_mul!,
    :reduced_normal_dense!, :reduced_normal_sparse!, :reduced_nrows, :reduced_residual!, :reduced_tmul!,
    :reset_schur_dense!, :reset_schur_sparse!, :residual_block, :solve_diagonal_blocks!, :retained_block, :retained_block_pairs,
    :schur_jacobian_bytes, :sort_schur_blocks, :sparse_with_map, :symmetrifyfull, :symmetrifyfull!,
    :fromthreadargs, :threadargs, :trial_workspace, :uniformscaling!, :zero!,
])

@testset "Public API: exactly the documented names are exported (X1)" begin
    exported = sort(filter(!=(:BlockLinearSystems), names(BlockLinearSystems)))
    @test exported == PUBLIC_API
    meta = Base.Docs.meta(BlockLinearSystems)
    for name in PUBLIC_API
        @test haskey(meta, Base.Docs.Binding(BlockLinearSystems, name))
    end
    # ... and each is listed in the API reference, which lists nothing else unqualified
    api = joinpath(@__DIR__, "..", "docs", "src", "api.md")
    if isfile(api)
        listed = Symbol[]
        for block in eachmatch(r"```@docs\n(.*?)```"s, read(api, String)), line in split(block.captures[1], '\n')
            entry = strip(line)
            isempty(entry) || push!(listed, Symbol(entry))
        end
        # ... apart from methods added to other packages' functions, listed qualified (rules.md 7)
        qualified = filter(e -> occursin('.', string(e)), listed)
        @test qualified == [Symbol("LinearAlgebra.dot(::AbstractVector{<:Real}, ::SchurBlockHessian{<:Real}, ::AbstractVector{<:Real})")]
        @test sort(unique(setdiff(listed, qualified))) == PUBLIC_API
    end
    # Internal names stay defined (qualified access), and the old alias is gone
    for name in (:copyto!, :inv_sym_block, :expand_block_perm, :validblock, :symmetrifysparse, :blockdata,
                 :eachstoredblock, :nruns, :runlength, :runblocksize, :matrixdimension, :runblockrange,
                 :runmatrixrange, :eachrun, :BlockOffsets, :makesparseindices)
        @test isdefined(BlockLinearSystems, name) && !(name in exported)
    end
    @test !isdefined(BlockLinearSystems, :SchurBlockStructure)
    # fast_bAb is gone (D1): dot(v, H, v) for a SchurBlockHessian, jtj_quadform for the QR tables
    @test !isdefined(BlockLinearSystems, :fast_bAb)
    @test hasmethod(LinearAlgebra.dot, Tuple{Vector{Float64}, SchurBlockHessian{Float64, Matrix{Float64}}, Vector{Float64}})
end
