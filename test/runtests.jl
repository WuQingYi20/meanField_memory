using Test

# Add project root to load path
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using DualMemoryABM

@testset "DualMemoryABM" begin
    include("test_deterministic.jl")
    include("test_statistical.jl")
end
