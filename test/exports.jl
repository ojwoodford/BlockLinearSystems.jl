using Test, BlockLinearSystems

# The public API rule: a name is either public (exported, with a docstring and an entry in the API reference)
# or internal (not exported, not in the API reference). Adding or dropping an export must change this list.
const PUBLIC_API = sort([
    :BlockOffsets, :BlockRLE, :DenseBlockLinearSystem, :QRReductionWorkspace, :SchurBlockJacobian,
    :SchurBlockMatrix, :SchurJacobianStructure, :SchurReductionWorkspace, :SchurStructure,
    :SparseBlockLinearSystem, :TransposedPattern,
    :backsubstitute!, :block, :block_sparse_nnz, :blockrange, :blocksizes, :build_r_to_s_map,
    :build_schur_pattern, :copy_structure, :coupling_block, :coupling_entry, :eliminated_block, :fast_bAb,
    :free_block_pairs, :jtj_diag!, :jtr!, :makesparseindices, :nblocks, :reduce_qr!, :reduce_schur_dense!,
    :reduce_schur_sparse!, :reduced_diagonal_blocks!, :reduced_jacobian!, :reduced_mul!,
    :reduced_normal_dense!, :reduced_normal_sparse!, :reduced_nrows, :reduced_residual!, :reduced_tmul!,
    :reset_schur_dense!, :reset_schur_sparse!, :residual_block, :retained_block, :retained_block_pairs,
    :schur_jacobian_bytes, :sort_schur_blocks, :sparse_with_map, :symmetrifyfull, :symmetrifyfull!,
    :trial_workspace, :uniformscaling!, :zero!,
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
        @test sort(unique(listed)) == PUBLIC_API
    end
    # Internal names stay defined (qualified access), and the old alias is gone
    for name in (:copyto!, :inv_sym_block, :expand_block_perm, :validblock, :symmetrifysparse, :blockdata,
                 :eachstoredblock, :nruns, :runlength, :runblocksize, :matrixdimension, :runblockrange,
                 :runmatrixrange, :eachrun, :sort_blocks_hierarchical)
        @test isdefined(BlockLinearSystems, name) && !(name in exported)
    end
    @test !isdefined(BlockLinearSystems, :SchurBlockStructure)
end
