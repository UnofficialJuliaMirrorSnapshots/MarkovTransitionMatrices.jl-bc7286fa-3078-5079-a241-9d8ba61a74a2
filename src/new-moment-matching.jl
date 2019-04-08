export discreteApprox!, discreteApprox, discreteNormalApprox, discreteNormalApprox!

# ----------------------- objective functions for max entropy calcs --------------------------

function expΔTx!(tmpvec::Vector, ΔT::AbstractMatrix, x::AbstractVector)
  mul!(tmpvec, ΔT, x)
  tmpvec .= exp.(tmpvec)
end

# objective
function entropyObjective_f!(tmpvec::Vector, x::Vector, q::Vector, ΔT::AbstractMatrix)
  expΔTx!(tmpvec, ΔT, x)
  return dot(q, tmpvec)
end

# gradient
function entropyObjective_g!(grad::Vector, tmpvec::Vector, x::Vector, q::Vector, ΔT::AbstractMatrix)
  n,L = size(ΔT)
  expΔTx!(tmpvec, ΔT, x)
  tmpvec .*= q
  for l = 1:L
    grad[l] = dot(tmpvec, @view(ΔT[:,l]))
  end
end

function entropyObjective_fg!(grad::Vector, tmpvec::Vector, x::Vector, q::Vector, ΔT::AbstractMatrix)
  entropyObjective_g!(grad, tmpvec, x, q, ΔT)
  return sum(tmpvec)
end

function entropyObjective_h!(hess::Matrix{T}, tmpvec::Vector, x::Vector, q::Vector, ΔT::AbstractMatrix) where {T}
  n,L = size(ΔT)
  expΔTx!(tmpvec, ΔT, x)
  tmpvec .*= q
  fill!(hess, zero(T))
  for k = 1:L
    for l = 1:L
      hess[l,k] = sum(@view(ΔT[:,l]) .* tmpvec .* @view(ΔT[:,k]))
    end
  end
end

# ----------------------- wrappers --------------------------


function discreteApprox!(p::AbstractVector, λfinal::AbstractVector, err::AbstractVector, tmp::Vector, q0::Vector, ΔT::AbstractMatrix{T}) where {T}

  l = length(λfinal)
  n = length(q0)
  (n,l) == size(ΔT)  || throw(DimensionMismatch())
  n == length(p)     || throw(DimensionMismatch())
  l == length(err)   || throw(DimensionMismatch())
  n == length(tmp)   || throw(DimensionMismatch())


  # test that initial value is finite
  λ0 = zeros(T,l)
  grad = zeros(T,l)
  f0 = entropyObjective_fg!(grad, tmp, λ0, q0, ΔT)
  !isfinite(f0)         && return Inf
  !all(isfinite.(grad)) && return Inf

  tdf = TwiceDifferentiable(
    (x::Vector)               -> entropyObjective_f!(       tmp, x, q0, ΔT),
    (grad::Vector, x::Vector) -> entropyObjective_g!( grad, tmp, x, q0, ΔT),
    (grad::Vector, x::Vector) -> entropyObjective_fg!(grad, tmp, x, q0, ΔT),
    (hess::Matrix, x::Vector) -> entropyObjective_h!( hess, tmp, x, q0, ΔT),
    λ0
  )

  opt = Optim.optimize(tdf, λ0, Newton())
  λ1 = opt.minimizer
  J = opt.minimum

  # update gradient
  entropyObjective_g!(grad, tmp, λ1, q0, ΔT)

  if norm(grad, Inf) < 1e-9 && all(isfinite.(grad)) && all(isfinite.(λ1)) && 0.0 < J < Inf && maximum(abs.(grad ./ J)) < 1e-5
    λfinal .= λ1
    expΔTx!(tmp, ΔT, λ1)
    p .= q0 .* tmp ./ J
    err .= grad ./ J
    return J
  else
    return Inf
  end

end


function ΔTmat!(ΔT::Matrix, dev::AbstractVector, Tbar::Vector)
  n,L = size(ΔT)
  length(dev) == n || throw(DimensionMismatch())
  length(Tbar) == L || throw(DimensionMismatch())

  ΔT[:,1] .= dev
  for j = 2:L
    ΔT[:,j] .= ΔT[:,j-1] .* dev
  end
  for j = 1:L
    ΔT[:,j] .-= Tbar[j]
  end
end



function discreteApprox!(P::AbstractMatrix, y::AbstractVector{T}, S::Union{AbstractVector, Base.Iterators.ProductIterator}, zval::Function, pdffun::Function, scaled_moments::Vector, scale_factor::Real, maxMoments::Integer, κ::Real) where {T<:Real}

  nS = length(S)
  n = length(y)
  0 < maxMoments < n || throw(error("Must use 1 to $n-1 moments or fewer"))
  (nS,n) == size(P)  || throw(DimensionMismatch())

  # Initialize elements that will be returned
  Λ          = zeros(T, nS, maxMoments)
  JN         = zeros(T, nS)
  approxErr  = zeros(T, nS, maxMoments)
  numMoments = zeros(Int, nS)

  # preallocate these, which will be updated each iteration
  ΔT  = Array{T}(undef,n, maxMoments)
  z   = Array{T}(undef,n)
  q   = Array{T}(undef,n)
  tmp = Array{T}(undef,n)

  for (i,st) in enumerate(S)
    z .= zval.(y, st)
    q .= max.(pdffun.(z), κ)
    z ./= scale_factor
    ΔTmat!(ΔT, z, scaled_moments)
    updated = false
    for l in maxMoments:-1:2
      J = discreteApprox!(@view(P[i,:]), @view(Λ[i,1:l]), @view(approxErr[i,1:l]), tmp, q, @view(ΔT[:,1:l]))
      if isfinite(J)
        JN[i], numMoments[i] = (J, l)
        updated = true
        break
      end
    end
    if !updated
      sumq = sum(q)
      P[i,:] .= q ./ sumq
    end

  end

  return JN, Λ, numMoments, approxErr
end

# ----------------------- wrappers --------------------------

function discreteApprox(y::AbstractVector{T}, S::Union{AbstractVector, Base.Iterators.ProductIterator}, zval::Function, pdffun::Function, scaled_moments::Vector, scale_factor::Real, maxMoments::Integer, κ::Real) where {T<:Real}
  n = length(y)
  nS = length(S)
  P = Array{T}(nS,n)
  return (P, discreteApprox!(P, y, S, zval, pdffun, scaled_moments, scale_factor, maxMoments, κ)...)
end



function discreteNormalApprox!(P::AbstractMatrix, y::AbstractVector, S::Union{AbstractVector, Base.Iterators.ProductIterator}, zval::Function, maxMoments::Integer, κ::Real)
  scale_factor = maximum(abs.(y))
  scaled_moments = [m for m in NormCentralMoment(maxMoments, 1.0/scale_factor)]
  discreteApprox!(P, y, S, zval, normpdf, scaled_moments, scale_factor, maxMoments, κ)
end

function discreteNormalApprox(y::AbstractVector{T}, S::Union{AbstractVector, Base.Iterators.ProductIterator}, zval::Function, maxMoments::Integer=2, κ::Real=1e-8) where {T}
  n = length(y)
  P = Matrix{T}(undef,n,n)
  out = discreteNormalApprox!(P, y, S, zval, maxMoments, κ)
  return (P, out...)
end
