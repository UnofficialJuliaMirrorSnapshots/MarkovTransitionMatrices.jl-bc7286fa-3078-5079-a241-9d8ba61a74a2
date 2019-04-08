import Base: iterate, getindex, length

abstract type AbstractNormCentralMoment end

struct stdNormCentralMoment <: AbstractNormCentralMoment
    n::Int
end

struct NormCentralMoment{T<:Real} <: AbstractNormCentralMoment
    n::Int
    σ::T
end

Base.length(iter::AbstractNormCentralMoment) = iter.n

σ(iter::stdNormCentralMoment) = 1.0
σ(iter::NormCentralMoment) = iter.σ

Base.getindex(iter::AbstractNormCentralMoment, k::Integer) = iseven(k) ? 0.0 : 2^(-k/2) * factorial(k) / factorial(k/2) * σ(iter)^k

function Base.iterate(iter::AbstractNormCentralMoment, state=(1, 1))
    # Even moment p is σᵖ(p-1)!! where `(p-1)!! = (p-1)(p-3)...(3)(1)`
    #    Note: k!! is a double factorial (product of numbers 1 to p with same parity (even/odd) as p)
    # i:  keeps track of which moment (1st, 2nd, ..., pth, etc...)
    # nf: keeps track of the double-factorial part
    i, nf = state

    # If done
    if i > iter.n
        return nothing

    # Odd moments are centered around 0
    elseif isodd(i)
        return (0.0, (i+1, nf))

    else
        nf *= (i-1)
        return (nf*σ(iter)^i, (i+1, nf))
    end
end
