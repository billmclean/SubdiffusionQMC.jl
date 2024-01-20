module FEM1D

import ..Vec64, ..OVec64, ..OMat64
import ArgCheck: @argcheck
import LinearAlgebra: SymTridiagonal, I

function load_vector(f::Function, x::OVec64, ξ::Vec64, w::Vec64)
    Nₕ = lastindex(x)
    F = Vec64(undef, Nₕ-1)
    load_vector!(f, F, x, ξ, w)
    return F
end

function load_vector!(f::Function, F::Vec64, x::OVec64, ξ::Vec64, w::Vec64)
    for n in eachindex(F)
	Σ₁, Σ₂ = 0.0, 0.0
	for j in eachindex(ξ)
	    xⱼ = ( 1 - ξ[j] ) * x[n-1] + ξ[j] * x[n]
	    Σ₁ += w[j] * f(xⱼ) * ξ[j]
	    xⱼ = ( 1 - ξ[j] ) * x[n] + ξ[j] * x[n+1]
	    Σ₂ += w[j] * f(xⱼ) * (1 - ξ[j])
	end
	F[n] = (x[n] - x[n-1]) * Σ₁ + (x[n+1] - x[n]) * Σ₂
    end
end

function stiffness_matrix(κ::Function, x::OVec64, ξ::Vec64, w::Vec64)
    Nₕ = lastindex(x)
    dv = Vec64(undef, Nₕ-1)
    ev = Vec64(undef, Nₕ-2)
    A = SymTridiagonal(dv, ev)
    stiffness_matrix!(A, κ, x, ξ, w)
    return A
end

function stiffness_matrix!(A::SymTridiagonal, κ::Function, x::OVec64, 
	                   ξ::Vec64, w::Vec64)
    h = diff(collect(x))
    Nₕ = lastindex(x)
    Σ = zeros(Nₕ)
    for n in eachindex(Σ)
	for j in eachindex(ξ)
	    xⱼ = (1 - ξ[j]) * x[n-1] + ξ[j] * x[n]
	    Σ[n] += w[j] * κ(xⱼ)
	end
	Σ[n] /= h[n]
    end
    for n in eachindex(A.dv)
	A.dv[n] = Σ[n] + Σ[n+1]
    end
    for n in eachindex(A.ev)
	A.ev[n] = -Σ[n+1]
    end
end

function mass_matrix(x::OVec64)
    h = diff(collect(x))
    Nₕ = lastindex(x)
    dv = Vec64(undef, Nₕ-1)
    ev = Vec64(undef, Nₕ-2)
    for n in eachindex(dv)
	dv[n] = ( h[n] + h[n+1] ) / 3
    end
    for n in eachindex(ev)
	ev[n] = h[n] / 6
    end
    SymTridiagonal(dv, ev)
end

end # module
