using Test, LinearAlgebra, SparseArrays, StaticArrays, Static, BlockLinearSystems

@testset "Schur-Complement Linear Solver: Pure-Data Kernels & Equivalence" begin
    # --------------------------------------------------------------------------
    # 1. Setup synthetic bipartite problem with retained blocks of several sizes and eliminated blocks of mixed sizes
    # --------------------------------------------------------------------------
    c_dims = UInt8[6, 3, 6, 9]       # 4 retained blocks of dimension 6, 3, 6 and 9
    ncblocks = length(c_dims)
    nc = Int(sum(c_dims))             # 24 scalars

    boffsets_c = Vector{Int}(undef, ncblocks)
    boffsets_c[1] = 1
    for i in 2:ncblocks
        boffsets_c[i] = boffsets_c[i-1] + Int(c_dims[i-1])
    end

    # 10 eliminated blocks: 8 of dimension 3, 2 of dimension 1
    p_dims = UInt8[3, 3, 3, 3, 3, 3, 3, 3, 1, 1]
    npblocks = length(p_dims)
    np = Int(sum(p_dims))             # 26 scalars

    boffsets_p = Vector{Int}(undef, npblocks)
    boffsets_p[1] = 1
    for i in 2:npblocks
        boffsets_p[i] = boffsets_p[i-1] + Int(p_dims[i-1])
    end

    # Flat C_data offsets
    ppoffsets = Vector{Int}(undef, npblocks)
    ppoffsets[1] = 1
    for i in 2:npblocks
        ppoffsets[i] = ppoffsets[i-1] + Int(p_dims[i-1])^2
    end
    total_c_data = ppoffsets[end] + Int(p_dims[end])^2 - 1

    # Coupling graph: each eliminated block is coupled to 2-4 retained blocks
    # Ensure retained block indices are strictly sorted ascending per eliminated block!
    couplings = [
        [1, 2],          # l=1
        [1, 3],          # l=2
        [2, 3, 4],       # l=3
        [1, 4],          # l=4
        [1, 2, 3],       # l=5
        [2, 4],          # l=6
        [3, 4],          # l=7
        [1, 2, 3, 4],    # l=8
        [1, 3],          # l=9 (1D)
        [2, 4]           # l=10 (1D)
    ]

    cpptr = Vector{Int}(undef, npblocks + 1)
    cpptr[1] = 1
    for l in 1:npblocks
        cpptr[l+1] = cpptr[l] + length(couplings[l])
    end
    num_couplings = cpptr[end] - 1

    cpblock = Vector{Int}(undef, num_couplings)
    cpoffsets = Vector{Int}(undef, num_couplings)
    curr_e_off = 1
    idx = 1
    for l in 1:npblocks
        dp = Int(p_dims[l])
        for c in couplings[l]
            cpblock[idx] = c
            cpoffsets[idx] = curr_e_off
            curr_e_off += Int(c_dims[c]) * dp
            idx += 1
        end
    end
    total_e_data = curr_e_off - 1

    # Max coupled dimension across eliminated blocks
    max_coupled_dim = maximum(sum(Int(c_dims[c]) for c in couplings[l]) for l in 1:npblocks)

    # Groups: 3D points (1:8), 1D depths (9:10)
    p_groups = [1:8, 9:10]
    p_sizes = [3, 1]

    # Pre-computed workload chunks for K=1 and K=4
    chunks_K1 = [[1:8], [9:10]]
    chunks_K4 = [
        [1:2, 3:4, 5:6, 7:8],
        [9:9, 10:10, 11:10, 11:10] # Empty ranges for threads 3 and 4 in 1D group
    ]

    structure_K1 = BlockLinearSystems.SchurStructure(
        nc, ncblocks, npblocks,
        c_dims, p_dims, boffsets_c, boffsets_p,
        ppoffsets, cpptr, cpblock, cpoffsets,
        max_coupled_dim, p_groups, p_sizes, chunks_K1
    )

    structure_K4 = BlockLinearSystems.SchurStructure(
        nc, ncblocks, npblocks,
        c_dims, p_dims, boffsets_c, boffsets_p,
        ppoffsets, cpptr, cpblock, cpoffsets,
        max_coupled_dim, p_groups, p_sizes, chunks_K4
    )

    # --------------------------------------------------------------------------
    # 2. Populate Numerical Data (B, C, E, gradients, damping)
    # --------------------------------------------------------------------------
    # Retained Hessian B: block-structured matching block_pattern (SPD)
    B = zeros(nc, nc)
    for c in 1:ncblocks
        dc = Int(c_dims[c])
        scol = boffsets_c[c]
        Xc = randn(dc, dc)
        B[scol:scol+dc-1, scol:scol+dc-1] = Xc' * Xc + 20.0 * I
    end
    # Add some off-diagonal retained-retained couplings
    for c2 in 1:ncblocks, c1 in 1:c2-1
        if (c1, c2) in [(1, 2), (2, 3), (3, 4)]
            dc1 = Int(c_dims[c1])
            dc2 = Int(c_dims[c2])
            srow = boffsets_c[c1]
            scol = boffsets_c[c2]
            X12 = 0.5 * randn(dc1, dc2)
            B[srow:srow+dc1-1, scol:scol+dc2-1] = X12
            B[scol:scol+dc2-1, srow:srow+dc1-1] = X12'
        end
    end

    # Eliminated Hessian C (block-diagonal, SPD)
    C_data = zeros(total_c_data)
    for l in 1:npblocks
        dp = Int(p_dims[l])
        po = ppoffsets[l]
        Xc = randn(dp, dp)
        Cl = Xc' * Xc + 3.0 * I
        for j in 1:dp, i in 1:dp
            C_data[po + (j-1)*dp + i - 1] = Cl[i, j]
        end
    end

    # Coupling matrix E (stored column-major as E^T in E_data)
    E_data = randn(total_e_data)

    # Damping & scaling
    lambda = 0.5
    scaling_c = 1.0 .+ rand(nc)
    scaling_p = 1.0 .+ rand(np)

    # Gradients
    b_c = randn(nc)
    b_p = randn(np)

    # --------------------------------------------------------------------------
    # 3. Assemble Full Monolithic Normal Equations & Ground Truth Solve
    # --------------------------------------------------------------------------
    H_full = zeros(nc + np, nc + np)
    # B block + lambda * diag(scaling_c)
    H_full[1:nc, 1:nc] .= B
    for i in 1:nc
        H_full[i, i] += lambda * scaling_c[i]
    end

    # C block + lambda * diag(scaling_p)
    for l in 1:npblocks
        dp = Int(p_dims[l])
        po = ppoffsets[l]
        p_off = boffsets_p[l]
        for col in 1:dp, row in 1:dp
            H_full[nc + p_off + row - 1, nc + p_off + col - 1] = C_data[po + (col-1)*dp + row - 1]
        end
        for i in 1:dp
            H_full[nc + p_off + i - 1, nc + p_off + i - 1] += lambda * scaling_p[p_off + i - 1]
        end
    end

    # E block
    for l in 1:npblocks
        dp = Int(p_dims[l])
        p_off = boffsets_p[l]
        e_start = cpptr[l]
        e_end = cpptr[l+1] - 1
        for e in e_start:e_end
            c = cpblock[e]
            dc = Int(c_dims[c])
            co = boffsets_c[c]
            eo = cpoffsets[e]
            for col in 1:dp, row in 1:dc
                # E_data stores E^T column-major: row `row` of E_cl is contiguous length-dp
                e_val = E_data[eo + (row - 1)*dp + col - 1]
                H_full[co + row - 1, nc + p_off + col - 1] = e_val
                H_full[nc + p_off + col - 1, co + row - 1] = e_val
            end
        end
    end

    g_full = [b_c; b_p]

    # Ground truth solve
    dx_full = H_full \ g_full
    dx_full_c = dx_full[1:nc]
    dx_full_p = dx_full[nc+1:end]

    # --------------------------------------------------------------------------
    # 4. Dense Schur Reduction & Equivalence Test (K = 1)
    # --------------------------------------------------------------------------
    ws_K1 = BlockLinearSystems.SchurReductionWorkspace{Float64}(structure_K1, 1; is_dense=true)
    S_dense = zeros(nc, nc)
    g_S_dense = zeros(nc)

    BlockLinearSystems.reset_schur_dense!(S_dense, B, scaling_c, lambda, ws_K1)
    success = BlockLinearSystems.reduce_schur_dense!(
        S_dense, g_S_dense, b_c, b_p, scaling_p, lambda,
        structure_K1, C_data, E_data, ws_K1
    )
    @test success == true
    @test ws_K1.failed_block[] == 0

    # S_dense is upper-triangular: symmetrize to solve directly
    S_sym = Symmetric(S_dense, :U)
    dx_c_dense = S_sym \ g_S_dense

    # Verify reduced step matches ground truth
    @test norm(dx_c_dense - dx_full_c) / norm(dx_full_c) < 1e-12

    # Back-substitute eliminable variables
    dx_p_dense = zeros(np)
    BlockLinearSystems.backsubstitute!(
        dx_p_dense, dx_c_dense, structure_K1, E_data,
        ws_K1.C_inv_data, ws_K1.z_data
    )

    # Verify back-substituted point step matches ground truth
    @test norm(dx_p_dense - dx_full_p) / norm(dx_full_p) < 1e-12

    # Full solution equivalence: ||Delta x_schur - Delta x_full|| / ||Delta x_full|| < 1e-12
    dx_schur_dense = [dx_c_dense; dx_p_dense]
    @test norm(dx_schur_dense - dx_full) / norm(dx_full) < 1e-12

    # --------------------------------------------------------------------------
    # 5. Multi-Chunk Parallel Reduction & Thread-Invariance Test (K = 4)
    # --------------------------------------------------------------------------
    ws_K4 = BlockLinearSystems.SchurReductionWorkspace{Float64}(structure_K4, 4; is_dense=true)
    S_dense_K4 = zeros(nc, nc)
    g_S_dense_K4 = zeros(nc)

    BlockLinearSystems.reset_schur_dense!(S_dense_K4, B, scaling_c, lambda, ws_K4)
    success_K4 = BlockLinearSystems.reduce_schur_dense!(
        S_dense_K4, g_S_dense_K4, b_c, b_p, scaling_p, lambda,
        structure_K4, C_data, E_data, ws_K4
    )
    @test success_K4 == true
    @test ws_K4.failed_block[] == 0

    # S and g_S must match K=1 to floating-point tolerance
    @test norm(triu(S_dense_K4) - triu(S_dense)) / norm(triu(S_dense)) < 1e-14
    @test norm(g_S_dense_K4 - g_S_dense) / norm(g_S_dense) < 1e-14

    dx_c_K4 = Symmetric(S_dense_K4, :U) \ g_S_dense_K4
    dx_p_K4 = zeros(np)
    BlockLinearSystems.backsubstitute!(
        dx_p_K4, dx_c_K4, structure_K4, E_data,
        ws_K4.C_inv_data, ws_K4.z_data
    )
    @test norm([dx_c_K4; dx_p_K4] - dx_full) / norm(dx_full) < 1e-12

    # --------------------------------------------------------------------------
    # 6. Sparse Reduced-System Pattern & Sparse Schur Reduction Test
    # --------------------------------------------------------------------------
    # Build the retained-block pattern of the reduced system (symmetric, upper triangle)
    block_pattern = spzeros(Bool, ncblocks, ncblocks)
    for l in 1:npblocks
        cams = couplings[l]
        for idx2 in 1:length(cams)
            c2 = cams[idx2]
            for idx1 in 1:idx2
                c1 = cams[idx1]
                block_pattern[c1, c2] = true
            end
        end
    end
    # Ensure block diagonal is present
    for c in 1:ncblocks
        block_pattern[c, c] = true
    end

    # CSC block-level indexing
    block_colptr = block_pattern.colptr
    block_rowval = block_pattern.rowval
    nnz_blocks = nnz(block_pattern)

    # Compute row offsets and total scalar column heights for S
    col_heights = zeros(Int, ncblocks)
    block_row_offsets = zeros(Int, nnz_blocks)
    for c2 in 1:ncblocks
        h = 0
        for idx in block_colptr[c2]:(block_colptr[c2+1]-1)
            c1 = block_rowval[idx]
            block_row_offsets[idx] = h
            h += Int(c_dims[c1])
        end
        col_heights[c2] = h
    end

    # Build scalar SparseMatrixCSC S directly from block pattern
    I_list = Int[]
    J_list = Int[]
    for c2 in 1:ncblocks
        for idx in block_colptr[c2]:(block_colptr[c2+1]-1)
            c1 = block_rowval[idx]
            for c_col in 1:c_dims[c2], c_row in 1:c_dims[c1]
                push!(I_list, boffsets_c[c1] + c_row - 1)
                push!(J_list, boffsets_c[c2] + c_col - 1)
            end
        end
    end

    S_sparse = sparse(I_list, J_list, zeros(length(I_list)), nc, nc)
    nnz_S = nnz(S_sparse)

    B_sparse = sparse(triu(B))
    b_to_s_map = Int[]
    for j in 1:nc
        for k in B_sparse.colptr[j]:(B_sparse.colptr[j+1]-1)
            i = B_sparse.rowval[k]
            s_k = 0
            for idx in S_sparse.colptr[j]:(S_sparse.colptr[j+1]-1)
                if S_sparse.rowval[idx] == i
                    s_k = idx
                    break
                end
            end
            @assert s_k != 0
            push!(b_to_s_map, s_k)
        end
    end

    diag_inds = zeros(Int, nc)
    for j in 1:nc
        for idx in S_sparse.colptr[j]:(S_sparse.colptr[j+1]-1)
            if S_sparse.rowval[idx] == j
                diag_inds[j] = idx
                break
            end
        end
        @assert diag_inds[j] != 0
    end

    # Sparse workspace (K = 1)
    ws_sparse_K1 = BlockLinearSystems.SchurReductionWorkspace{Float64}(
        structure_K1, 1; is_dense=false, nnz_S=nnz_S
    )

    BlockLinearSystems.reset_schur_sparse!(
        S_sparse.nzval, B_sparse.nzval, b_to_s_map, diag_inds, scaling_c, lambda, ws_sparse_K1
    )

    g_S_sparse = zeros(nc)
    success_sparse = BlockLinearSystems.reduce_schur_sparse!(
        S_sparse, g_S_sparse, b_c, b_p, scaling_p, lambda,
        structure_K1, C_data, E_data,
        block_colptr, block_rowval, block_row_offsets, col_heights,
        ws_sparse_K1
    )
    @test success_sparse == true
    @test ws_sparse_K1.failed_block[] == 0

    # Verify sparse S matches dense S on all upper-triangular nonzeros
    S_sparse_dense = Matrix(S_sparse)
    for j in 1:nc, i in 1:j
        if S_sparse_dense[i, j] != 0
            @test isapprox(S_sparse_dense[i, j], S_dense[i, j]; atol=1e-12)
        end
    end
    @test norm(g_S_sparse - g_S_dense) < 1e-12

    # Solve via sparse Cholesky
    F_sparse = cholesky(Symmetric(S_sparse, :U))
    dx_c_sparse = F_sparse \ g_S_sparse
    @test norm(dx_c_sparse - dx_full_c) / norm(dx_full_c) < 1e-12

    # Sparse reduction with K = 4 chunks
    ws_sparse_K4 = BlockLinearSystems.SchurReductionWorkspace{Float64}(
        structure_K4, 4; is_dense=false, nnz_S=nnz_S
    )
    S_sparse_K4 = copy(S_sparse)
    BlockLinearSystems.reset_schur_sparse!(
        S_sparse_K4.nzval, B_sparse.nzval, b_to_s_map, diag_inds, scaling_c, lambda, ws_sparse_K4
    )
    g_S_sparse_K4 = zeros(nc)
    success_sparse_K4 = BlockLinearSystems.reduce_schur_sparse!(
        S_sparse_K4, g_S_sparse_K4, b_c, b_p, scaling_p, lambda,
        structure_K4, C_data, E_data,
        block_colptr, block_rowval, block_row_offsets, col_heights,
        ws_sparse_K4
    )
    @test success_sparse_K4 == true
    @test ws_sparse_K4.failed_block[] == 0
    # Chunks are summed in a different order than for K=1, so compare to rounding error
    @test norm(S_sparse_K4.nzval - S_sparse.nzval) < 1e-13 * norm(S_sparse.nzval)
    @test norm(g_S_sparse_K4 - g_S_sparse) < 1e-13 * norm(g_S_sparse)

    # --------------------------------------------------------------------------
    # 7. Exact Rank-Deficient Degenerate Eliminated Block Test
    # --------------------------------------------------------------------------
    # Create rank-deficient block for eliminated block 1: outer product of 1D vector (rank 1 at lambda = 0)
    C_data_degen = copy(C_data)
    v_rank1 = randn(3)
    C_rank1 = v_rank1 * v_rank1' # rank 1 in 3x3 => det = 0
    po1 = ppoffsets[1]
    for j in 1:3, i in 1:3
        C_data_degen[po1 + (j-1)*3 + i - 1] = C_rank1[i, j]
    end

    ws_degen = BlockLinearSystems.SchurReductionWorkspace{Float64}(structure_K1, 1; is_dense=true)
    S_degen = zeros(nc, nc)
    g_S_degen = zeros(nc)
    BlockLinearSystems.reset_schur_dense!(S_degen, B, scaling_c, 0.0, ws_degen)

    # Reduction with lambda = 0 must fail on eliminated block 1
    success_degen = BlockLinearSystems.reduce_schur_dense!(
        S_degen, g_S_degen, b_c, b_p, scaling_p, 0.0,
        structure_K1, C_data_degen, E_data, ws_degen
    )
    @test success_degen == false
    @test ws_degen.failed_block[] == 1

    # --------------------------------------------------------------------------
    # 8. Zero-Allocation Unit Tests on Inner Group Kernels
    # --------------------------------------------------------------------------
    # Phase 1 (eliminated block factorization) and phase 2 (column accumulation) kernels
    call_phase1() = BlockLinearSystems.factor_eliminated_group!(
        ws_K1.C_inv_data, ws_K1.z_data, ws_K1.W_data, ws_K1.W_scratch[1], ws_K1.failed_block,
        structure_K1, C_data, E_data, b_p, scaling_p, lambda, 1:8, static(3))
    call_phase1()
    @test (@allocated call_phase1()) == 0

    call_dense_cols() = BlockLinearSystems.accumulate_dense_columns!(
        S_dense, g_S_dense, structure_K1, E_data, ws_K1.W_data, ws_K1.z_data,
        ws_K1.retained_ptr, ws_K1.retained_entry, ws_K1.entry_eliminated, 1:ncblocks)
    call_dense_cols()
    @test (@allocated call_dense_cols()) == 0

    call_phase1_sparse() = BlockLinearSystems.factor_eliminated_group!(
        ws_sparse_K1.C_inv_data, ws_sparse_K1.z_data, ws_sparse_K1.W_data, ws_sparse_K1.W_scratch[1], ws_sparse_K1.failed_block,
        structure_K1, C_data, E_data, b_p, scaling_p, lambda, 1:8, static(3))
    call_phase1_sparse()
    @test (@allocated call_phase1_sparse()) == 0

    call_sparse_cols() = BlockLinearSystems.accumulate_sparse_columns!(
        S_sparse.nzval, g_S_sparse, structure_K1, E_data, ws_sparse_K1.W_data, ws_sparse_K1.z_data,
        ws_sparse_K1.retained_ptr, ws_sparse_K1.retained_entry, ws_sparse_K1.entry_eliminated,
        S_sparse.colptr, block_colptr, block_rowval, block_row_offsets, col_heights, 1:ncblocks)
    call_sparse_cols()
    @test (@allocated call_sparse_cols()) == 0

    # --------------------------------------------------------------------------
    # 9. Two-Pass Ordering & Permutation Expansion Tests
    # --------------------------------------------------------------------------
    # Test expand_block_perm
    b_perm = [2, 1, 3]
    b_sizes = [3, 2, 4]
    s_perm = expand_block_perm(b_perm, b_sizes)
    @test s_perm == [4, 5, 1, 2, 3, 6, 7, 8, 9]

    # Test sort_schur_blocks
    is_elim = Bool[false, true, false, true, true]
    sizes = [6, 3, 3, 1, 3]
    obs = [[], [2, 1], [], [1], [3, 2]]
    # Retained: 1 (dim 6), 3 (dim 3) -> ordered [1, 3]
    # Eliminable: 2 (dim 3, c_min 1), 5 (dim 3, c_min 2), 4 (dim 1, c_min 1)
    # Dim 3: 2 has c_min 1, 5 has c_min 2 -> [2, 5]
    # Dim 1: 4 has c_min 1 -> [4]
    # Total perm: [1, 3, 2, 5, 4]
    schur_perm = sort_schur_blocks(is_elim, sizes, obs)
    @test schur_perm == [1, 3, 2, 5, 4]

    # --------------------------------------------------------------------------
    # 10. Symbolic Pattern & B-to-S Map Generator Tests
    # --------------------------------------------------------------------------
    pattern = build_schur_pattern(structure_K1)
    @test pattern.block_colptr == block_colptr
    @test pattern.block_rowval == block_rowval
    @test pattern.block_row_offsets == block_row_offsets
    @test pattern.col_heights == col_heights
    @test pattern.diag_inds == diag_inds
    @test nnz(pattern.S_sparse) == nnz_S

    b_map = build_b_to_s_map(B_sparse, pattern.S_sparse)
    @test b_map == b_to_s_map

    # --------------------------------------------------------------------------
    # 11. SchurBlockMatrix & fast_bAb Equivalence Test
    # --------------------------------------------------------------------------
    H_dense_block = SchurBlockMatrix(B, C_data, E_data, structure_K1)
    # Undamped H_full
    H_full_undamped = copy(H_full)
    for i in 1:nc
        H_full_undamped[i, i] -= lambda * scaling_c[i]
    end
    for l in 1:npblocks
        dp = Int(p_dims[l])
        p_off = boffsets_p[l]
        for i in 1:dp
            H_full_undamped[nc + p_off + i - 1, nc + p_off + i - 1] -= lambda * scaling_p[p_off + i - 1]
        end
    end

    dx_test = randn(nc + np)
    quad_exact = dot(dx_test, H_full_undamped * dx_test)
    quad_fast = fast_bAb(H_dense_block, dx_test)
    @test isapprox(quad_fast, quad_exact; atol=1e-10)
end
