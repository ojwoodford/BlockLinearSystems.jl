using BenchmarkTools
using BlockLinearSystems
using SparseArrays
using Static
using StaticArrays

const SUITE = BenchmarkGroup()

# 1. Setup bundle-adjustment-like problem: 20 cameras (dim 6), 100 landmarks (dim 3)
ncams = 20
npts = 100
cam_dim = 6
pt_dim = 3
nblocks = ncams + npts
blocksizes = vcat(fill(cam_dim, ncams), fill(pt_dim, npts))

# Lower-triangular sparsity:
# cam-cam diagonal blocks, pt-pt diagonal blocks, and sparse cam-pt measurement edges
edges = Tuple{Int, Int}[]
# Cam-cam
for c in 1:ncams
    push!(edges, (c, c))
end
# Pt-pt
for p in 1:npts
    push!(edges, (ncams + p, ncams + p))
end
# Random observations: each landmark seen by 5 cameras
for p in 1:npts
    cams_seen = ((p * 3) .% ncams) .+ 1
    for c in 1:5
        cam_idx = ((cams_seen + c) % ncams) + 1
        push!(edges, (ncams + p, cam_idx))
    end
end
unique!(edges)

bsm = SparseBlockLinearSystem{Float64}(edges, blocksizes)
csc_mat, scatter_map = sparse_with_map(bsm; symmetric=true)

SUITE["sparse"] = BenchmarkGroup()

# Benchmark block accumulation (100 observation updates)
H_cam_pt = @SMatrix rand(3, 6)
SUITE["sparse"]["block_accumulation"] = @benchmarkable begin
    for (r, c) in $edges
        if r > $ncams && c <= $ncams
            block($bsm, r, c, Val(3), Val(6)) .+= $H_cam_pt
        end
    end
end

# Benchmark zero-allocation scatter into SparseMatrixCSC
SUITE["sparse"]["copyto_scatter"] = @benchmarkable copyto!($csc_mat, $bsm, $scatter_map)

# Benchmark full rebuilding of SparseMatrixCSC (naive approach)
SUITE["sparse"]["naive_rebuild_sparse"] = @benchmarkable symmetrifysparse($bsm)

# 2. Dense Linear System Benchmark
bdm = DenseBlockLinearSystem{Float64}(fill(6, 10))
H_dense_blk = @SMatrix rand(6, 6)
SUITE["dense"] = BenchmarkGroup()
SUITE["dense"]["block_accumulation"] = @benchmarkable begin
    for i in 1:10
        block($bdm, i, i, Val(6), Val(6)) .+= $H_dense_blk
    end
end
SUITE["dense"]["symmetrifyfull!"] = @benchmarkable symmetrifyfull!($bdm)

# 3. Schur Submatrix Extraction
H_cams = Matrix{Float64}(undef, ncams * cam_dim, ncams * cam_dim)
SUITE["schur"] = BenchmarkGroup()
SUITE["schur"]["submatrix_extraction"] = @benchmarkable symmetrifyfull!($H_cams, $bsm, 1:$ncams)

if abspath(PROGRAM_FILE) == @__FILE__
    results = run(SUITE, verbose=true)
    println("\n=== Benchmark Results ===")
    display(results)
end
