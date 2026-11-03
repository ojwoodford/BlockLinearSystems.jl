using Documenter
using BlockMatrices

DocMeta.setdocmeta!(BlockMatrices, :DocTestSetup, :(using BlockMatrices, SparseArrays, Static, StaticArrays); recursive=true)

makedocs(
    sitename = "BlockMatrices.jl",
    modules = [BlockMatrices],
    remotes = nothing,
    warnonly = false,
    format = Documenter.HTML(
        prettyurls = false,
        sidebar_sitename = true,
    ),
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md",
        "Tutorials & Guide" => [
            "Block Matrix Assembly & Indexing" => "tutorials/block_assembly.md",
            "Symmetrification, CSC Scatter & Schur Systems" => "tutorials/symmetrification_and_schur.md",
        ],
    ]
)
