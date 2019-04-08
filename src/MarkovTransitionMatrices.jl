__precompile__()

module MarkovTransitionMatrices

using GenGlobal
using Distributions
using Optim
using StatsFuns
using SharedArrays
using Base.Iterators
using Distributed
using LinearAlgebra
using SparseArrays

export markov_transition, markovswitching_transition, markov_transition_moment_matching_parallel, markov_transition_moment_matching_serial

myDist(μ::Real, σ::Real) = Normal(μ, σ)
myDist(μ::Vector, Σ::Matrix) = MvNormal(μ, Σ)
mypdf(dist::UnivariateDistribution, s2) = pdf(dist, s2...)
mypdf(dist::MultivariateDistribution, s2) = pdf(dist, [s2...])

whichP(P::AbstractMatrix, ::Type{Val{true}})  = sparse(P)
whichP(P::SharedMatrix,   ::Type{Val{false}}) = sdata(P)
whichP(P::Matrix,         ::Type{Val{false}}) = P

function sparsify!(P::AbstractMatrix{T}, minp::Real) where {T<:AbstractFloat}
  P ./= sum(P, dims=2)
  P .*= (P .> minp)
  P ./= sum(P, dims=2)
  makesparse = minp > 0.0
  return whichP(P, Val{makesparse})
end



include("simple_no_matching.jl")
include("moment_matching.jl")
include("transition_from_data.jl")
# include("Farmer_Toda_VAR.jl")
include("normal_moments.jl")
include("new-moment-matching.jl")


# module end
end
