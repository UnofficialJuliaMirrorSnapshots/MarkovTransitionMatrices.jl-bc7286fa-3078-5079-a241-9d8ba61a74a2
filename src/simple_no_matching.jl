"""
    markov_transition(μ::Function, Σ::Function, minp::AbstractFloat, statevectors::AbstractVector...)

Returns the sparse, Markov Transition Matrix (that is, `P[i,j] = Pr(i|j)`) where all
elements are `> minp`. The state-space is defined as the Cartesian Product of the `statevectors`.
The functions `μ` and `Σ` should take a tuple from `Base.product(statevectors...)` and return EITHER
  - the mean and **STANDARD DEVIATION** of `Distributions.Normal`, or
  - the mean and **VARIANCE MATRIX** of `Distributions.MvNormal`
"""
function markov_transition(μ::Function, Σ::Function, minp::AbstractFloat, statevectors::AbstractVector{T}...) where {T<:AbstractFloat}

  0. <= minp < 1. || throw(DomainError())

  state_prod = Base.product(statevectors...)

  P = zeros(T, length(state_prod), length(state_prod))

  for (j, s2) in enumerate(state_prod)
    for (i, s1) in enumerate(state_prod)
      dist = myDist(μ(s1), Σ(s1))
      P[i,j] = mypdf(dist, s2)
    end
  end

  return sparsify!(P, minp)
end


"""
    markovswitching_transition(μ::Function, Σ::Function, π::Matrix{Float64}, minp::AbstractFloat, statevectors::AbstractVector...)

Returns a sparse, Markov Transition Matrix (that is, `P[i,j] = Pr(i|j)`) where all
elements are `> minp`. The state space is defined as the Cartesian Product of the
`statevectors` with `1:size(π,1)`. The functions `μ(::Int, ::Tuple)` and `Σ(::Int, ::Tuple)`
should take the index of a regime `1:size(π,1)` and a tuple from `Base.product(statevectors...)`
and return EITHER
  - the mean and **STANDARD DEVIATION** of `Distributions.Normal`, or
  - the mean and **VARIANCE MATRIX** of `Distributions.MvNormal`
The markov-switching matrix is NOT transposed and equals `π[i,j] = Pr(j|i)`

"""
function markovswitching_transition(μ::Function, Σ::Function, π::Matrix{Float64}, minp::AbstractFloat, statevectors::AbstractVector{T}...) where {T<:AbstractFloat}

  k = size(π,1)
  k == size(π,2)   || throw(DimensionMismatch())
  all(sum(π, dims=2) .≈ 1.0) || throw(error("each row of π must sum to 1"))


  n = prod(map(length, statevectors))
  regimes = Base.OneTo(k)
  P = zeros(T, k*n, k*n)

  for r1 in regimes
    for r2 in regimes
      P[(r1-1)*n+1:r1*n,  (r2-1)*n+1:r2*n] .= π[r1, r2] .* markov_transition( (s) -> μ(r1, s), (s) -> Σ(r1, s), minp, statevectors... )
    end
  end

  return whichP(P, Val{minp > 0.0})
end
