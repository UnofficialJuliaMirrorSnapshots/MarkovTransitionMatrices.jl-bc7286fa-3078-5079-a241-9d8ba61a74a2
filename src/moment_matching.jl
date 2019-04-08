
function lower_triangular!(x::AbstractMatrix{T}, ltvec::AbstractVector{T}) where {T}
  n,m = size(x)
  n == m || throw(DimensionMismatch("must be square"))
  length(ltvec) == n*(n+1)/2
  k = 0
  @inbounds for j = 1:n
    for i = j:n
      k += 1
      ltvec[k] = x[i,j]
    end
  end
  return ltvec
end

function lower_triangular(x::AbstractMatrix{T}) where {T}
  n = size(x, 1)
  ltvec = Vector{T}(undef, Int(n*(n+1)/2))
  get_lower_triangular(x, ltvec)
end

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------

function f(λ::Vector{T}, q0::Vector{T}, ΔT::Matrix{T}) where {T<:AbstractFloat}
  (L,J) = size(ΔT)
  J == length(q0) || throw(DimensionMismatch())
  L == length(λ) || throw(DimensionMismatch())
  val = zero(T)
  @inbounds for j = 1:J
    val += q0[j] * exp(dot(λ, @view(ΔT[:,j])))
  end
  return val
end

function g!(λ::Vector{T}, grad::Vector{T}, q0::Vector{T}, ΔT::Matrix{T}) where {T<:AbstractFloat}
  (L,J) = size(ΔT)
  J == length(q0) || throw(DimensionMismatch())
  L == length(λ) == length(grad) || throw(DimensionMismatch())
  grad .= 0.0
  @inbounds for j = 1:J
    x = q0[j] * exp(dot(λ, @view(ΔT[:,j])))
    grad .+= x .* @view(ΔT[:,j])
  end
end

function fg!(λ::Vector{T}, grad::Vector{T}, q0::Vector{T}, ΔT::Matrix{T}) where {T<:AbstractFloat}
  (L,J) = size(ΔT)
  J == length(q0) || throw(DimensionMismatch())
  L == length(λ) == length(grad) || throw(DimensionMismatch())
  grad .= zero(T)
  val = zero(T)
  @inbounds for j = 1:J
    x = q0[j] * exp(dot(λ, @view(ΔT[:,j])))
    grad .+= x .* @view(ΔT[:,j])
    val += x
  end
  return val
end


function h!(λ::Vector{T}, hess::Matrix{T}, q0::Vector{T}, ΔT::Matrix{T}) where {T<:AbstractFloat}
  (L,J) = size(ΔT)
  J == length(q0) || throw(DimensionMismatch())
  L == length(λ) || throw(DimensionMismatch())
  (L,L,) == size(hess) || throw(DimensionMismatch())
  hess .= zero(T)
  @inbounds for j = 1:J
    x = q0[j] * exp(dot(λ, @view(ΔT[:,j])))
    vw = @view(ΔT[:,j])
    BLAS.gemm!('N', 'T', x, vw, vw, one(T), hess)
  end
end

# --------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------


# Multivariate version
function momentdiff!(sprod::Base.Iterators.ProductIterator, theory_mean::Vector{T}, theory_var::Matrix{T}, ΔT::Matrix) where {T<:AbstractFloat}
  J = length(sprod)
  nμ = length(theory_mean)
  (nμ, nμ) == size(theory_var) || throw(error(DimensionMismatch()))
  (nμ + nμ*(nμ+1)/2, J) == size(ΔT) || throw(error(DimensionMismatch()))

  dev = zeros(T, ndims(sprod))
  outerprod = zeros(T, ndims(sprod), ndims(sprod))

  for (j,s) in enumerate(sprod)
    dev .= [s...] .- theory_mean
    LinearAlgebra.BLAS.gemm!('N', 'T', one(T), dev, dev, zero(T), outerprod)
    outerprod .-= theory_var
    ΔT[1:nμ, j] .= dev
    lower_triangular!(outerprod, @view(ΔT[nμ+1:end, j]))
  end
  return nothing
end


# Univariate version
function momentdiff!(sprod::Base.Iterators.ProductIterator, theory_mean::T, theory_sd::T, ΔT::Matrix) where {T<:AbstractFloat}
  J = length(sprod)
  ndims(sprod) == 1 || throw(DimensionMismatch())
  (2, J) == size(ΔT) || throw(DimensionMismatch())
  theory_var = theory_sd^2

  for (j,s) in enumerate(sprod)
    dev = s[1] - theory_mean
    outerprod = dev^2 - theory_var
    ΔT[:,j] .= [dev, outerprod]
  end
  return nothing
end


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------

function matchmoment!(i::Integer, s1::NTuple{N,T}, q0::Vector{T}, ΔT::Array{T}, grad::Vector{T}, approxErr::AbstractMatrix{T}, P::AbstractMatrix{T}, moments_matched::AbstractVector{Int}, μ::Function, Σ::Function, state_prod::Base.Iterators.ProductIterator) where {N,T}
  L, J = size(ΔT)

  mean0 = μ(s1)
  var0 =  Σ(s1)

  dist = myDist(mean0, var0)
  q0 .= zero(T)

  # fill in initial approximation
  for (j, s2) in enumerate(state_prod)
    q0[j] = mypdf(dist, s2)
  end

  momentdiff!(state_prod, mean0, var0, ΔT)

  f_cl(  λ::Vector{T})                  where {T} = f(  λ,       q0, ΔT)
  g_cl!( grad::Vector{T}, λ::Vector{T}) where {T} = g!( λ, grad, q0, ΔT)
  fg_cl!(grad::Vector{T}, λ::Vector{T}) where {T} = fg!(λ, grad, q0, ΔT)
  h_cl!( hess::Matrix{T}, λ::Vector{T}) where {T} = h!( λ, hess, q0, ΔT)
  td = TwiceDifferentiable(f_cl, g_cl!, fg_cl!, h_cl!, ones(T,L))

  res = Optim.optimize(td, ones(T,L), Optim.Options(time_limit=15))
  λ = Optim.minimizer(res)
  J_candidate = Optim.minimum(res)
  g_cl!(grad, λ)
  approxErr[:, i] .= grad ./ J_candidate

  # if we like the results, update and break
  if ( norm(@view(approxErr[:, i]), Inf) < 1e-4 ) & all(isfinite.(grad)) & all(isfinite.(λ)) & (J_candidate > 0.0)
    for k in 1:length(q0)
      P[i,k] = q0[k] * exp( dot(λ, @view(ΔT[:,k])) ) / J_candidate
    end
    moments_matched[i] = L
  else
    P[i,:] .= q0
  end
end

@GenGlobal g_q0 g_ΔT g_grad g_stateprod g_μ g_Σ g_approxErr g_P g_moments_matched

function matchmoment!(i::Integer, s1::NTuple{N,T}) where {N,T}
  global g_q0, g_ΔT, g_grad, g_stateprod, g_μ, g_Σ, g_approxErr, g_P, g_moments_matched
  matchmoment!(i, s1, g_q0::Vector{T}, g_ΔT::Array{T}, g_grad::Vector{T}, g_approxErr::SharedMatrix{T}, g_P::SharedMatrix{T}, g_moments_matched::SharedVector{Int}, g_μ::Function, g_Σ::Function, g_stateprod)
end

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------

function markov_transition_moment_matching_parallel(μ::Function, Σ::Function, minp::AbstractFloat, statevectors::AbstractVector{T}...) where {T<:AbstractFloat}

  0.0 <= minp < 1.0 || throw(DomainError())

  state_prod = Base.product(statevectors...)

  # dimensions
  nd = ndims(state_prod)     # num basis vectors in state space
  J = length(state_prod)     # size of state space
  L = Int(nd + nd*(nd+1)/2)  # moments to match (mean + var)

  P               = SharedMatrix{T}(   (J, J,) ) # , init = S -> S[localindices(S)] = zero(T))
  approxErr       = SharedMatrix{T}(   (L,J,)  ) # ,  init = S -> S[localindices(S)] = typemax(T))
  moments_matched = SharedVector{Int}( (J,)    ) # ,    init = S -> S[localindices(S)] = zero(Int))

  fill!(P, zero(T))
  fill!(approxErr, typemax(T))
  fill!(moments_matched, zero(Int))

  @eval @everywhere begin
    set_g_q0(  zeros($T, $J) )
    set_g_ΔT(  zeros($T, $L, $J) )
    set_g_grad(zeros($T, $L) )
    set_g_stateprod($state_prod)
    set_g_μ($μ)
    set_g_Σ($Σ)
    set_g_approxErr($approxErr)
    set_g_P($P)
    set_g_moments_matched($moments_matched)
  end

  @sync @distributed for is1 in collect(enumerate(state_prod))
    matchmoment!(is1...)
  end

  return sparsify!(P, minp), sdata(moments_matched), sdata(approxErr)
end







function markov_transition_moment_matching_serial(μ::Function, Σ::Function, minp::AbstractFloat, statevectors::AbstractVector{T}...) where {T<:AbstractFloat}

  0. <= minp < 1. || throw(DomainError())

  state_prod = Base.product(statevectors...)

  # dimensions
  nd = ndims(state_prod)     # num basis vectors in state space
  J = length(state_prod)     # size of state space
  L = Int(nd + nd*(nd+1)/2)  # moments to match (mean + var)

  # temp variables
  q0 = zeros(T, J)
  ΔT = zeros(T, L, J)
  grad = zeros(T, L)

  # output variables
  P               = zeros(T, J, J)
  approxErr       = fill(typemax(T), L, J)
  moments_matched = zeros(Int, J)

  # Need to parallelize this?
  for (i, s1) in enumerate(state_prod)
    matchmoment!(i, s1, q0, ΔT, grad, approxErr, P, moments_matched, μ, Σ, state_prod)
  end

  return sparsify!(P, minp), moments_matched, approxErr
end
