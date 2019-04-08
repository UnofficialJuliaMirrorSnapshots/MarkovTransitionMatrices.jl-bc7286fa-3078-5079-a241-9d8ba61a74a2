export make_grid_min_max, BMtransition_from_data

function make_grid_min_max(logx::Vector; n::Int=31, scal::Real=5.0)
    a = minimum(logx)-log(scal)
    b = maximum(logx)+log(scal)
    step = (b-a)/(n-1)
    return a : step : b
end

markov_transition_match_option(μ::Function, Σ::Function, ::Type{Val{true}}, minp::Real,  grid::AbstractVector)     = markov_transition_moment_matching_serial(  μ, Σ, minp, grid)
markov_transition_match_option(μ::Function, Σ::Function, ::Type{Val{true}}, minp::Real,  grids::AbstractVector...) = markov_transition_moment_matching_parallel(μ, Σ, minp, grids...)
markov_transition_match_option(μ::Function, Σ::Function, ::Type{Val{false}}, minp::Real, grids::AbstractVector...) = markov_transition(μ, Σ, minp, grids...)

function BMtransition_from_data(logxy::Array{T,2}, matching::Type; zero_cov::Bool=true, minp::Real=1e-9, kwargs...) where {T<:Real}
    0.0 <= minp < 1.0 || throw(DomainError())
    grids = ((make_grid_min_max(logxy[:,j]; kwargs...) for j in 1:size(logxy, 2))...,)
    Σ = cov(diff(logxy,1))
    zero_cov  &&  Σ .= Diagonal(Σ)
    transition = markov_transition_match_option((xy::NTuple) -> [xy...], (xy::NTuple) -> Σ, matching, minp, grids...)
    return (grids, Σ, transition, )
end

function BMtransition_from_data(logx::Array{T,1}, matching::Type; minp::Real=1e-9, kwargs...) where {T<:Real}
    0.0 <= minp < 1.0 || throw(DomainError())
    grid = make_grid_min_max(logx; kwargs...)
    σ = sqrt(var(diff(logx)))
    transition = markov_transition_match_option( (x) -> x[1], (x) -> σ, matching, minp, grid)
    return (grid, σ^2., transition, )
end
