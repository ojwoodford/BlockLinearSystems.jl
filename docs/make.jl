using Documenter
using BlockLinearSystems

DocMeta.setdocmeta!(BlockLinearSystems, :DocTestSetup, :(using BlockLinearSystems, SparseArrays, Static, StaticArrays); recursive=true)

makedocs(
    sitename = "BlockLinearSystems.jl",
    modules = [BlockLinearSystems],
    remotes = nothing,
    warnonly = false,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        sidebar_sitename = true,
    ),
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md",
        "Tutorials & Guide" => [
            "Block Linear System Assembly & Indexing" => "tutorials/block_assembly.md",
            "Symmetrification & CSC Scatter" => "tutorials/symmetrification_and_schur.md",
            "Schur Complement Reduction" => "tutorials/schur_reduction.md",
        ],
    ]
)

deploydocs(
    repo = "github.com/ojwoodford/BlockLinearSystems.jl.git",
    devbranch = "main",
    push_preview = true,
)
