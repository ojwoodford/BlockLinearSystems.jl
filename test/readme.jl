# Every ```julia block of the README, run in order in one module (later examples may use earlier ones' variables),
# so that the README's examples are kept working (AB7)
@testset "README examples run" begin
    readme = read(joinpath(pkgdir(BlockLinearSystems), "README.md"), String)
    blocks = [m.captures[1] for m in eachmatch(r"```julia\n(.*?)```"s, readme)]
    @test length(blocks) == 5
    mod = Module()
    for (k, code) in enumerate(blocks)
        @test (include_string(mod, code, "README.md block $k"); true)
    end
    # The two reduction examples solve their systems
    @test mod.ws isa QRReductionWorkspace
    @test all(isfinite, mod.x_r) && all(isfinite, mod.x_e)
end
