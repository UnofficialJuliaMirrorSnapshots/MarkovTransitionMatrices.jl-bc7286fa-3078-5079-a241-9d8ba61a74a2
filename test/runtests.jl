using MarkovTransitionMatrices
using Test
using BenchmarkTools
using SharedArrays
using Distributed
using LinearAlgebra

using Distributions

isapprox_oneval(x::AbstractArray) = all( x .≈ x[1] )

include("FarmerTodaBmark.jl")

# setup for test
testvec = -3.0:0.5:0.0
ss = collect(Base.product(testvec, testvec))
n = length(testvec)
nn = length(ss)
mvd = MvNormal(zeros(2), Matrix{Float64}(I,2,2))

# test the 1-d version
fullp1 = Matrix(markov_transition((s) -> 0., (s) -> 1., 1e-8, testvec))
ratio1 = fullp1 ./ pdf.(Normal(0,1), testvec')
@test isapprox_oneval(ratio1)

# test the 2-d version
fullp2 = Matrix(markov_transition((s) -> zeros(2), (s) -> Matrix{Float64}(I,2,2), 1e-8, testvec, testvec))
ratio2 = reshape(fullp2[1,:], n, n) ./  [pdf(mvd, [s...]) for s in ss]
@test isapprox_oneval(ratio2)

# test the markov-switching version
μswitch(r::Real,s::NTuple{N,T}) where {N,T<:Real} = r==1 ? zeros(2) : [s...]
Σswitch(r::Real,s::NTuple{N,T}) where {N,T<:Real} = r==1 ? Matrix{Float64}(I,2,2)   : eps(1.0)*Matrix(I,2,2)
πswitch = Matrix{Float64}(I, 2, 2)
fullp3 = Matrix(markovswitching_transition(μswitch, Σswitch, πswitch, 0.0, testvec, testvec))

@test all(fullp3[1:nn, 1:nn] .== fullp2)
@test all(fullp3[nn+1:2*nn, 1:nn] .== 0.0)
@test all(fullp3[1:nn, nn+1:2*nn] .== 0.0)
@test all(fullp3[nn+1:2*nn, nn+1:2*nn] .== Matrix(I,nn,nn))

# ---------------------------------------------------------------

println("Testing moment-matching")

s = -3.0:0.25:3.0
ss = -3.0:0.125:3.0, -3.0:0.125:3.0

pids = addprocs()
@everywhere using MarkovTransitionMatrices

# simulate random walks w/ moment matching
p_P_match_1, p_mom, p_er = markov_transition_moment_matching_parallel((s) -> 0.0   , (s) -> 1.0   , 1e-8, s)
s_P_match_1, s_mom, s_er = markov_transition_moment_matching_serial(  (s) -> 0.0   , (s) -> 1.0   , 1e-8, s)
@test true == all(s_P_match_1 .== p_P_match_1) == all(s_mom .== p_mom) == all(s_er[isfinite.(s_er)] .== p_er[isfinite.(p_er)])
println("Single-var serial/parallel OK")

# p_P_match_2, p_mom, p_er = markov_transition_moment_matching_parallel((s) -> [s...], (s) -> [1.0 0.0; 0.0 1.0], 1e-8, ss...)
p_P_match_2, p_mom, p_er = markov_transition_moment_matching_parallel((s) -> [s...], (s) -> Matrix{Float64}(LinearAlgebra.I,2,2), 1e-8, ss...)
s_P_match_2, s_mom, s_er = markov_transition_moment_matching_serial(  (s) -> [s...], (s) -> Matrix{Float64}(LinearAlgebra.I,2,2), 1e-8, ss...)
@test true == all(s_P_match_2 .≈ p_P_match_2) == all(s_mom .≈ p_mom) == all(s_er[isfinite.(s_er)] .== p_er[isfinite.(p_er)])
println("Multi-var serial/parallel OK")

# share of states that get moment-matching
share = sum(s_mom .> 0) / length(s_mom) * 100
@show "$share percent of states get moment-matching"
@test share .> 0.5

# without matching
P_nomatch_2 = markov_transition((s) -> [s...], (s) -> Matrix{Float64}(I,2,2), 1e-8, ss...)
number_diff = sum(s_P_match_2 .!= P_nomatch_2)
sd_diff = sum((s_P_match_2 .- P_nomatch_2).^2)/prod(size(s_P_match_2))
abs_diff = sum(abs.(s_P_match_2 .- P_nomatch_2))/prod(size(s_P_match_2))
sharediff = number_diff / prod(size(s_P_match_2))
nm = norm(vec(s_P_match_2 .- P_nomatch_2), Inf)
@show "max difference between matching & no-matching is $nm"
@show "$sharediff percent of probabilities change"


# benchmarking
println("Benchmarking parallel...")
@show @benchmark markov_transition_moment_matching_parallel(  (s) -> [s...], (s) -> Matrix{Float64}(LinearAlgebra.I,2,2), 1e-8, ss...)
println("Benchmarking serial...")
@show @benchmark markov_transition_moment_matching_serial(    (s) -> [s...], (s) -> Matrix{Float64}(I,2,2), 1e-8, ss...)

rmprocs(pids)

# using Plots
# gr()
#
# heatmap(s, s, reshape(P_match_2[10,:]  , length.(ss)...))
# heatmap(s, s, reshape(P_nomatch_2[10,:], length.(ss)...))
# heatmap(s, s, reshape(P_match_2[20,:] .- P_nomatch_2[20,:], length.(ss)...))
# heatmap(s, s, reshape(mom.>0, length.(ss)...))
# histogram(vec(clamp.(P_match_2 .- P_nomatch_2, -0.05, 0.05)) )
