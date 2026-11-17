# Loads the test fixtures into Main, each once per session, so that including a test file again (or a test file in
# its own module) does not make new types. A test file includes this and then does `using Main.<fixture module>`.
for name in (:SchurFixtures, :QRFixtures, :AllocFixtures)   # QRFixtures uses SchurFixtures
    isdefined(Main, name) || Base.include(Main, joinpath(@__DIR__, "$name.jl"))
end
