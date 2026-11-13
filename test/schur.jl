using Test, LinearAlgebra, SparseArrays, StaticArrays, Static, BlockLinearSystems

@testset "Schur-Complement Linear Solver: Pure-Data Kernels & Equivalence" begin
    # --------------------------------------------------------------------------
    # 1. Setup synthetic bipartite problem with retained blocks of several sizes and eliminated blocks of mixed sizes
    # --------------------------------------------------------------------------
    r_dims = UInt8[6, 3, 6, 9]       # 4 retained blocks of dimension 6, 3, 6 and 9
    nrblocks = length(r_dims)
    nr = Int(sum(r_dims))             # 24 scalars

    boffsets_r = Vector{Int}(undef, nrblocks)
    boffsets_r[1] = 1
    for i in 2:nrblocks
        boffsets_r[i] = boffsets_r[i-1] + Int(r_dims[i-1])
    end

    # 10 eliminated blocks: 8 of dimension 3, 2 of dimension 1
    e_dims = UInt8[3, 3, 3, 3, 3, 3, 3, 3, 1, 1]
    neblocks = length(e_dims)
    ne = Int(sum(e_dims))             # 26 scalars

    boffsets_e = Vector{Int}(undef, neblocks)
    boffsets_e[1] = 1
    for i in 2:neblocks
        boffsets_e[i] = boffsets_e[i-1] + Int(e_dims[i-1])
    end

    # Flat E_data offsets
    eoffsets = Vector{Int}(undef, neblocks)
    eoffsets[1] = 1
    for i in 2:neblocks
        eoffsets[i] = eoffsets[i-1] + Int(e_dims[i-1])^2
    end
    total_e_data = eoffsets[end] + Int(e_dims[end])^2 - 1

    # Coupling graph: each eliminated block is coupled to 2-4 retained blocks
    # Ensure retained block indices are strictly sorted ascending per eliminated block!
    couplings = [
        [1, 2],          # e=1
        [1, 3],          # e=2
        [2, 3, 4],       # e=3
        [1, 4],          # e=4
        [1, 2, 3],       # e=5
        [2, 4],          # e=6
        [3, 4],          # e=7
        [1, 2, 3, 4],    # e=8
        [1, 3],          # e=9 (1D)
        [2, 4]           # e=10 (1D)
    ]

    eliminated_ptr = Vector{Int}(undef, neblocks + 1)
    eliminated_ptr[1] = 1
    for e in 1:neblocks
        eliminated_ptr[e+1] = eliminated_ptr[e] + length(couplings[e])
    end
    num_couplings = eliminated_ptr[end] - 1

    entry_retained = Vector{Int}(undef, num_couplings)
    coffsets = Vector{Int}(undef, num_couplings)
    curr_c_off = 1
    idx = 1
    for e in 1:neblocks
        de = Int(e_dims[e])
        for r in couplings[e]
            entry_retained[idx] = r
            coffsets[idx] = curr_c_off
            curr_c_off += Int(r_dims[r]) * de
            idx += 1
        end
    end
    total_c_data = curr_c_off - 1

    # Max coupled dimension across eliminated blocks
    max_coupled_dim = maximum(sum(Int(r_dims[r]) for r in couplings[e]) for e in 1:neblocks)

    # Groups: eliminated blocks of dimension 3 (1:8) and 1 (9:10)
    e_groups = [1:8, 9:10]
    e_sizes = [3, 1]

    # Pre-computed workload chunks for K=1 and K=4
    chunks_K1 = [[1:8], [9:10]]
    chunks_K4 = [
        [1:2, 3:4, 5:6, 7:8],
        [9:9, 10:10, 11:10, 11:10] # Empty ranges for threads 3 and 4 in 1D group
    ]

    structure_K1 = BlockLinearSystems.SchurStructure(
        nr, nrblocks, neblocks,
        r_dims, e_dims, boffsets_r, boffsets_e,
        eoffsets, eliminated_ptr, entry_retained, coffsets,
        max_coupled_dim, e_groups, e_sizes, chunks_K1
    )

    structure_K4 = BlockLinearSystems.SchurStructure(
        nr, nrblocks, neblocks,
        r_dims, e_dims, boffsets_r, boffsets_e,
        eoffsets, eliminated_ptr, entry_retained, coffsets,
        max_coupled_dim, e_groups, e_sizes, chunks_K4
    )

    # --------------------------------------------------------------------------
    # 2. Populate Numerical Data (R, C, E, gradients, damping)
    # --------------------------------------------------------------------------
    # Retained Hessian R: block-structured matching block_pattern (SPD)
    R = zeros(nr, nr)
    for r in 1:nrblocks
        dr = Int(r_dims[r])
        scol = boffsets_r[r]
        Xc = randn(dr, dr)
        R[scol:scol+dr-1, scol:scol+dr-1] = Xc' * Xc + 20.0 * I
    end
    # Add some off-diagonal retained-retained couplings
    for r2 in 1:nrblocks, r1 in 1:r2-1
        if (r1, r2) in [(1, 2), (2, 3), (3, 4)]
            dr1 = Int(r_dims[r1])
            dr2 = Int(r_dims[r2])
            srow = boffsets_r[r1]
            scol = boffsets_r[r2]
            X12 = 0.5 * randn(dr1, dr2)
            R[srow:srow+dr1-1, scol:scol+dr2-1] = X12
            R[scol:scol+dr2-1, srow:srow+dr1-1] = X12'
        end
    end

    # Eliminated Hessian C (block-diagonal, SPD)
    E_data = zeros(total_e_data)
    for e in 1:neblocks
        de = Int(e_dims[e])
        eo = eoffsets[e]
        Xc = randn(de, de)
        Cl = Xc' * Xc + 3.0 * I
        for j in 1:de, i in 1:de
            E_data[eo + (j-1)*de + i - 1] = Cl[i, j]
        end
    end

    # Coupling blocks C_er (each stored column-major in C_data)
    C_data = randn(total_c_data)

    # Damping & scaling
    lambda = 0.5
    scaling_r = 1.0 .+ rand(nr)
    scaling_e = 1.0 .+ rand(ne)

    # Gradients
    b_r = randn(nr)
    b_e = randn(ne)

    # --------------------------------------------------------------------------
    # 3. Assemble Full Monolithic Normal Equations & Ground Truth Solve
    # --------------------------------------------------------------------------
    H_full = zeros(nr + ne, nr + ne)
    # R block + lambda * diag(scaling_r)
    H_full[1:nr, 1:nr] .= R
    for i in 1:nr
        H_full[i, i] += lambda * scaling_r[i]
    end

    # C block + lambda * diag(scaling_e)
    for e in 1:neblocks
        de = Int(e_dims[e])
        eo = eoffsets[e]
        e_off = boffsets_e[e]
        for col in 1:de, row in 1:de
            H_full[nr + e_off + row - 1, nr + e_off + col - 1] = E_data[eo + (col-1)*de + row - 1]
        end
        for i in 1:de
            H_full[nr + e_off + i - 1, nr + e_off + i - 1] += lambda * scaling_e[e_off + i - 1]
        end
    end

    # E block
    for e in 1:neblocks
        de = Int(e_dims[e])
        e_off = boffsets_e[e]
        c_start = eliminated_ptr[e]
        c_end = eliminated_ptr[e+1] - 1
        for c in c_start:c_end
            r = entry_retained[c]
            dr = Int(r_dims[r])
            ro = boffsets_r[r]
            co = coffsets[c]
            for col in 1:de, row in 1:dr
                # C_er is column-major (de x dr): column `row` (a retained index) is contiguous, length de
                c_val = C_data[co + (row - 1)*de + col - 1]
                H_full[ro + row - 1, nr + e_off + col - 1] = c_val
                H_full[nr + e_off + col - 1, ro + row - 1] = c_val
            end
        end
    end

    g_full = [b_r; b_e]

    # Ground truth solve
    dx_full = H_full \ g_full
    dx_full_r = dx_full[1:nr]
    dx_full_e = dx_full[nr+1:end]

    # --------------------------------------------------------------------------
    # 4. Dense Schur Reduction & Equivalence Test (K = 1)
    # --------------------------------------------------------------------------
    ws_K1 = BlockLinearSystems.SchurReductionWorkspace{Float64}(structure_K1, 1; is_dense=true)
    S_dense = zeros(nr, nr)
    g_S_dense = zeros(nr)

    BlockLinearSystems.reset_schur_dense!(S_dense, R, scaling_r, lambda, ws_K1)
    success = BlockLinearSystems.reduce_schur_dense!(
        S_dense, g_S_dense, b_r, b_e, scaling_e, lambda,
        structure_K1, E_data, C_data, ws_K1
    )
    @test success == true
    @test ws_K1.failed_block[] == 0

    # S_dense is upper-triangular: symmetrize to solve directly
    S_sym = Symmetric(S_dense, :U)
    dx_r_dense = S_sym \ g_S_dense

    # Verify reduced step matches ground truth
    @test norm(dx_r_dense - dx_full_r) / norm(dx_full_r) < 1e-12

    # Back-substitute eliminated variables
    dx_e_dense = zeros(ne)
    BlockLinearSystems.backsubstitute!(
        dx_e_dense, dx_r_dense, structure_K1, C_data,
        ws_K1.E_inv_data, ws_K1.z_data
    )

    # Verify back-substituted eliminated step matches ground truth
    @test norm(dx_e_dense - dx_full_e) / norm(dx_full_e) < 1e-12

    # Full solution equivalence: ||Delta x_schur - Delta x_full|| / ||Delta x_full|| < 1e-12
    dx_schur_dense = [dx_r_dense; dx_e_dense]
    @test norm(dx_schur_dense - dx_full) / norm(dx_full) < 1e-12

    # --------------------------------------------------------------------------
    # 5. Multi-Chunk Parallel Reduction & Thread-Invariance Test (K = 4)
    # --------------------------------------------------------------------------
    ws_K4 = BlockLinearSystems.SchurReductionWorkspace{Float64}(structure_K4, 4; is_dense=true)
    S_dense_K4 = zeros(nr, nr)
    g_S_dense_K4 = zeros(nr)

    BlockLinearSystems.reset_schur_dense!(S_dense_K4, R, scaling_r, lambda, ws_K4)
    success_K4 = BlockLinearSystems.reduce_schur_dense!(
        S_dense_K4, g_S_dense_K4, b_r, b_e, scaling_e, lambda,
        structure_K4, E_data, C_data, ws_K4
    )
    @test success_K4 == true
    @test ws_K4.failed_block[] == 0

    # S and g_S must match K=1 to floating-point tolerance
    @test norm(triu(S_dense_K4) - triu(S_dense)) / norm(triu(S_dense)) < 1e-14
    @test norm(g_S_dense_K4 - g_S_dense) / norm(g_S_dense) < 1e-14

    dx_r_K4 = Symmetric(S_dense_K4, :U) \ g_S_dense_K4
    dx_e_K4 = zeros(ne)
    BlockLinearSystems.backsubstitute!(
        dx_e_K4, dx_r_K4, structure_K4, C_data,
        ws_K4.E_inv_data, ws_K4.z_data
    )
    @test norm([dx_r_K4; dx_e_K4] - dx_full) / norm(dx_full) < 1e-12

    # --------------------------------------------------------------------------
    # 6. Sparse Reduced-System Pattern & Sparse Schur Reduction Test
    # --------------------------------------------------------------------------
    # Build the retained-block pattern of the reduced system (symmetric, upper triangle)
    block_pattern = spzeros(Bool, nrblocks, nrblocks)
    for e in 1:neblocks
        coupled = couplings[e]
        for idx2 in 1:length(coupled)
            r2 = coupled[idx2]
            for idx1 in 1:idx2
                r1 = coupled[idx1]
                block_pattern[r1, r2] = true
            end
        end
    end
    # Ensure block diagonal is present
    for r in 1:nrblocks
        block_pattern[r, r] = true
    end

    # CSC block-level indexing
    block_colptr = block_pattern.colptr
    block_rowval = block_pattern.rowval
    nnz_blocks = nnz(block_pattern)

    # Compute row offsets and total scalar column heights for S
    col_heights = zeros(Int, nrblocks)
    block_row_offsets = zeros(Int, nnz_blocks)
    for r2 in 1:nrblocks
        h = 0
        for idx in block_colptr[r2]:(block_colptr[r2+1]-1)
            r1 = block_rowval[idx]
            block_row_offsets[idx] = h
            h += Int(r_dims[r1])
        end
        col_heights[r2] = h
    end

    # Build scalar SparseMatrixCSC S directly from block pattern
    I_list = Int[]
    J_list = Int[]
    for r2 in 1:nrblocks
        for idx in block_colptr[r2]:(block_colptr[r2+1]-1)
            r1 = block_rowval[idx]
            for r_col in 1:r_dims[r2], r_row in 1:r_dims[r1]
                push!(I_list, boffsets_r[r1] + r_row - 1)
                push!(J_list, boffsets_r[r2] + r_col - 1)
            end
        end
    end

    S_sparse = sparse(I_list, J_list, zeros(length(I_list)), nr, nr)
    nnz_S = nnz(S_sparse)

    R_sparse = sparse(triu(R))
    r_to_s_map = Int[]
    for j in 1:nr
        for k in R_sparse.colptr[j]:(R_sparse.colptr[j+1]-1)
            i = R_sparse.rowval[k]
            s_k = 0
            for idx in S_sparse.colptr[j]:(S_sparse.colptr[j+1]-1)
                if S_sparse.rowval[idx] == i
                    s_k = idx
                    break
                end
            end
            @assert s_k != 0
            push!(r_to_s_map, s_k)
        end
    end

    diag_inds = zeros(Int, nr)
    for j in 1:nr
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
        S_sparse.nzval, R_sparse.nzval, r_to_s_map, diag_inds, scaling_r, lambda, ws_sparse_K1
    )

    g_S_sparse = zeros(nr)
    success_sparse = BlockLinearSystems.reduce_schur_sparse!(
        S_sparse, g_S_sparse, b_r, b_e, scaling_e, lambda,
        structure_K1, E_data, C_data,
        block_colptr, block_rowval, block_row_offsets, col_heights,
        ws_sparse_K1
    )
    @test success_sparse == true
    @test ws_sparse_K1.failed_block[] == 0

    # Verify sparse S matches dense S on all upper-triangular nonzeros
    S_sparse_dense = Matrix(S_sparse)
    for j in 1:nr, i in 1:j
        if S_sparse_dense[i, j] != 0
            @test isapprox(S_sparse_dense[i, j], S_dense[i, j]; atol=1e-12)
        end
    end
    @test norm(g_S_sparse - g_S_dense) < 1e-12

    # Solve via sparse Cholesky
    F_sparse = cholesky(Symmetric(S_sparse, :U))
    dx_r_sparse = F_sparse \ g_S_sparse
    @test norm(dx_r_sparse - dx_full_r) / norm(dx_full_r) < 1e-12

    # Sparse reduction with K = 4 chunks
    ws_sparse_K4 = BlockLinearSystems.SchurReductionWorkspace{Float64}(
        structure_K4, 4; is_dense=false, nnz_S=nnz_S
    )
    S_sparse_K4 = copy(S_sparse)
    BlockLinearSystems.reset_schur_sparse!(
        S_sparse_K4.nzval, R_sparse.nzval, r_to_s_map, diag_inds, scaling_r, lambda, ws_sparse_K4
    )
    g_S_sparse_K4 = zeros(nr)
    success_sparse_K4 = BlockLinearSystems.reduce_schur_sparse!(
        S_sparse_K4, g_S_sparse_K4, b_r, b_e, scaling_e, lambda,
        structure_K4, E_data, C_data,
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
    E_data_degen = copy(E_data)
    v_rank1 = randn(3)
    E_rank1 = v_rank1 * v_rank1' # rank 1 in 3x3 => det = 0
    eo1 = eoffsets[1]
    for j in 1:3, i in 1:3
        E_data_degen[eo1 + (j-1)*3 + i - 1] = E_rank1[i, j]
    end

    ws_degen = BlockLinearSystems.SchurReductionWorkspace{Float64}(structure_K1, 1; is_dense=true)
    S_degen = zeros(nr, nr)
    g_S_degen = zeros(nr)
    BlockLinearSystems.reset_schur_dense!(S_degen, R, scaling_r, 0.0, ws_degen)

    # Reduction with lambda = 0 must fail on eliminated block 1
    success_degen = BlockLinearSystems.reduce_schur_dense!(
        S_degen, g_S_degen, b_r, b_e, scaling_e, 0.0,
        structure_K1, E_data_degen, C_data, ws_degen
    )
    @test success_degen == false
    @test ws_degen.failed_block[] == 1

    # --------------------------------------------------------------------------
    # 8. Zero-Allocation Unit Tests on Inner Group Kernels
    # --------------------------------------------------------------------------
    # Phase 1 (eliminated block factorization) and phase 2 (column accumulation) kernels
    call_phase1() = BlockLinearSystems.factor_eliminated_group!(
        ws_K1.E_inv_data, ws_K1.z_data, ws_K1.W_data, ws_K1.W_scratch[1], ws_K1.failed_block,
        structure_K1, E_data, C_data, b_e, scaling_e, lambda, 1:8, static(3))
    call_phase1()
    @test (@allocated call_phase1()) == 0

    call_dense_cols() = BlockLinearSystems.accumulate_dense_columns!(
        S_dense, g_S_dense, structure_K1, C_data, ws_K1.W_data, ws_K1.z_data,
        ws_K1.retained_ptr, ws_K1.retained_entry, ws_K1.entry_eliminated, 1:nrblocks)
    call_dense_cols()
    @test (@allocated call_dense_cols()) == 0

    call_phase1_sparse() = BlockLinearSystems.factor_eliminated_group!(
        ws_sparse_K1.E_inv_data, ws_sparse_K1.z_data, ws_sparse_K1.W_data, ws_sparse_K1.W_scratch[1], ws_sparse_K1.failed_block,
        structure_K1, E_data, C_data, b_e, scaling_e, lambda, 1:8, static(3))
    call_phase1_sparse()
    @test (@allocated call_phase1_sparse()) == 0

    call_sparse_cols() = BlockLinearSystems.accumulate_sparse_columns!(
        S_sparse.nzval, g_S_sparse, structure_K1, C_data, ws_sparse_K1.W_data, ws_sparse_K1.z_data,
        ws_sparse_K1.retained_ptr, ws_sparse_K1.retained_entry, ws_sparse_K1.entry_eliminated,
        S_sparse.colptr, block_colptr, block_rowval, block_row_offsets, col_heights, 1:nrblocks)
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
    # Eliminable: 2 (dim 3, r_min 1), 5 (dim 3, r_min 2), 4 (dim 1, r_min 1)
    # Dim 3: 2 has r_min 1, 5 has r_min 2 -> [2, 5]
    # Dim 1: 4 has r_min 1 -> [4]
    # Total perm: [1, 3, 2, 5, 4]
    schur_perm = sort_schur_blocks(is_elim, sizes, obs)
    @test schur_perm == [1, 3, 2, 5, 4]

    # --------------------------------------------------------------------------
    # 10. Symbolic Pattern & R-to-S Map Generator Tests
    # --------------------------------------------------------------------------
    pattern = build_schur_pattern(structure_K1)
    @test pattern.block_colptr == block_colptr
    @test pattern.block_rowval == block_rowval
    @test pattern.block_row_offsets == block_row_offsets
    @test pattern.col_heights == col_heights
    @test pattern.diag_inds == diag_inds
    @test nnz(pattern.S_sparse) == nnz_S

    b_map = build_r_to_s_map(R_sparse, pattern.S_sparse)
    @test b_map == r_to_s_map

    # --------------------------------------------------------------------------
    # 11. SchurBlockMatrix & fast_bAb Equivalence Test
    # --------------------------------------------------------------------------
    H_dense_block = SchurBlockMatrix(R, E_data, C_data, structure_K1)
    # Undamped H_full
    H_full_undamped = copy(H_full)
    for i in 1:nr
        H_full_undamped[i, i] -= lambda * scaling_r[i]
    end
    for e in 1:neblocks
        de = Int(e_dims[e])
        e_off = boffsets_e[e]
        for i in 1:de
            H_full_undamped[nr + e_off + i - 1, nr + e_off + i - 1] -= lambda * scaling_e[e_off + i - 1]
        end
    end

    dx_test = randn(nr + ne)
    quad_exact = dot(dx_test, H_full_undamped * dx_test)
    quad_fast = fast_bAb(H_dense_block, dx_test)
    @test isapprox(quad_fast, quad_exact; atol=1e-10)
end
