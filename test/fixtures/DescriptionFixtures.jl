# Descriptions of the fixture problems (EL1) and today's route from a description to a structure, as NLLSModel does it.
# Loaded once per session into Main by fixtures/load.jl; no testsets.
module DescriptionFixtures

using Random, BlockLinearSystems
using Main.SchurFixtures, Main.QRFixtures

export Description, caller_numbering, description, old_partition, old_hessian, old_jacobian, same_fields, NEW_FIELDS

const BLS = BlockLinearSystems

# A description in the caller's numbering
struct Description
    blocksizes::Vector{Int}
    eliminated::Vector{Bool}
    rowblocks::Vector{Vector{Int}}
    rowblocksizes::Vector{Int}
end

# Caller numbers for a fixture's partitioned form (retained 1:nr, then eliminated): a random permutation `q`, so that the
# caller's order is not the layout's
function caller_numbering(rng, r_dims, e_dims)
    nr, ne = length(r_dims), length(e_dims)
    q = randperm(rng, nr + ne)
    bs = zeros(Int, nr + ne); el = falses(nr + ne)
    for i in 1:nr; bs[q[i]] = r_dims[i]; end
    for e in 1:ne; bs[q[nr+e]] = e_dims[e]; el[q[nr+e]] = true; end
    return q, bs, collect(Bool, el)
end

# A SchurTestProblem as a description: for each eliminated block a row block of its own (a prior), and its couplings as one
# row block (even blocks) or one row block per coupling (odd blocks); each retained-retained pair as a row block
function description(p::SchurTestProblem, rng)
    nr = length(p.r_dims)
    q, bs, el = caller_numbering(rng, p.r_dims, p.e_dims)
    rbs = Vector{Int}[]
    for (e, c) in enumerate(p.couplings)
        push!(rbs, [q[nr+e]])
        if iseven(e)
            push!(rbs, shuffle(rng, [q[nr+e]; q[c]]))
        else
            foreach(r -> push!(rbs, shuffle(rng, [q[r], q[nr+e]])), c)
        end
    end
    foreach(((r1, r2),) -> push!(rbs, [q[r1], q[r2]]), p.rr_pairs)
    shuffle!(rng, rbs)
    return Description(bs, el, rbs, rand(rng, 1:3, length(rbs)))
end

# A QRTestProblem as a description: the rows of each eliminated block's table as a row block of all its couplings and,
# when it has more than one row, a one-row block of the eliminated block alone; each free group as one row block; and
# with `empty`, a row block touching no column block
function description(p::QRTestProblem, rng; empty::Bool=false)
    jst = p.J.structure
    nr, ne = length(p.r_dims), length(p.e_dims)
    q, bs, el = caller_numbering(rng, p.r_dims, p.e_dims)
    rbs = Vector{Int}[]; rows = Int[]
    for e in 1:ne
        m = jst.e_rows[e]
        push!(rbs, shuffle(rng, [q[nr+e]; q[p.couplings[e]]])); push!(rows, m > 1 ? m - 1 : m)
        m > 1 && (push!(rbs, [q[nr+e]]); push!(rows, 1))
    end
    for g in eachindex(jst.free_rows)
        t = ne + g
        push!(rbs, shuffle(rng, q[jst.entry_retained[jst.table_ptr[t]:jst.table_ptr[t+1]-1]])); push!(rows, jst.free_rows[g])
    end
    empty && (push!(rbs, Int[]); push!(rows, 2))
    perm = randperm(rng, length(rbs))
    return Description(bs, el, rbs[perm], rows[perm])
end

# Today's route, as NLLSModel does it (src/problem.jl partition_schur_problem, src/tall_system.jl tall_schur_map): coupling
# lists in the retained blocks' own numbering, sort_schur_blocks, the remapping, the positional constructors
function old_partition(d::Description)
    n = length(d.blocksizes)
    nret = count(!, d.eliminated)
    retained_map = Dict(v => i for (i, v) in enumerate(findall(!, d.eliminated)))
    coupled = [Int[] for _ in 1:n]
    pairs = Set{Tuple{Int, Int}}()
    for rb in d.rowblocks
        es = [v for v in rb if d.eliminated[v]]
        length(es) == 1 && foreach(v -> d.eliminated[v] || push!(coupled[es[1]], retained_map[v]), rb)
        rin = [retained_map[v] for v in rb if !d.eliminated[v]]
        for a in 2:length(rin), b in 1:a-1
            push!(pairs, minmax(rin[a], rin[b]))
        end
    end
    coupled = unique.(coupled)
    perm = BLS.sort_schur_blocks(d.eliminated, d.blocksizes, coupled)
    old_to_new = Dict(retained_map[perm[p]] => p for p in 1:nret)
    couplings = [sort([old_to_new[r] for r in coupled[perm[p]]]) for p in nret+1:n]
    extra = sort!([minmax(old_to_new[a], old_to_new[b]) for (a, b) in pairs])
    return (; perm, r_dims=d.blocksizes[perm[1:nret]], e_dims=d.blocksizes[perm[nret+1:n]], couplings, extra, nret)
end

function old_hessian(d::Description; nchunks=1)
    p = old_partition(d)
    return BLS.SchurStructure(p.r_dims, p.e_dims, p.couplings; nchunks), p
end

function old_jacobian(d::Description; nchunks=1)
    p = old_partition(d)
    nret, ne = p.nret, length(p.e_dims)
    role = zeros(Int, length(d.blocksizes))
    foreach(((q, v),) -> role[v] = q <= nret ? q : nret - q, enumerate(p.perm))
    K = length(d.rowblocks)
    table = zeros(Int, K); first = zeros(Int, K); e_rows = zeros(Int, ne)
    group = Dict{Vector{Int}, Int}(); free_retained = Vector{Int}[]; free_rows = Int[]
    for (k, rb) in enumerate(d.rowblocks)
        m = d.rowblocksizes[k]
        e = 0; rs = Int[]
        for v in rb
            role[v] < 0 && (e = -role[v]); role[v] > 0 && push!(rs, role[v])
        end
        if e > 0
            table[k] = e; first[k] = e_rows[e] + 1; e_rows[e] += m
        else
            sort!(rs)
            g = get(group, rs, 0)
            g == 0 && (push!(free_retained, rs); push!(free_rows, 0); g = group[rs] = length(free_rows))
            table[k] = ne + g; first[k] = free_rows[g] + 1; free_rows[g] += m
        end
    end
    st = BLS.SchurStructure(p.r_dims, p.e_dims, p.couplings; nchunks)
    return BLS.SchurJacobianStructure(st, e_rows; free_rows, free_retained, nchunks), p, table, first
end

# Field-for-field equality, skipping the fields only the description route fills
function same_fields(a, b; skip=())
    for f in fieldnames(typeof(a))
        f in skip && continue
        x, y = getfield(a, f), getfield(b, f)
        ok = x isa SchurStructure ? same_fields(x, y; skip) : isequal(x, y)
        ok || (@warn "field $f differs"; return false)
    end
    return true
end
const NEW_FIELDS = (:order, :retained_pairs, :row_table, :row_first, :row_nrows, :row_ptr, :row_blocks)


end # module DescriptionFixtures
