module TimeStepping

import ..Vec64, ..Mat64, ..OVec64, ..OMat64
import ..crank_nicolson!, ..graded_mesh, ..weights
import LinearAlgebra: SymTridiagonal
import OffsetArrays: OffsetVector
import SpecialFunctions: gamma

"""
    crank_nicolson!(U, M, A, F, t)

Solves the system of ODEs `Mu̇ + Au = F(t)` using the Crank-Nicolson scheme
to compute `U[j,n] ≈ u(x[j], t[n])`.  Assumes that the calling program
has initialised `U[:,0]` to the initial data, with `F[j,n] ≈ f(x[j]`.

The array dimensions are as follows.

    U[j,n]	for 0 ≤ j ≤ Nₕ, 0 ≤ n ≤ Nₜ,
    A[i,j]	for 1 ≤ i ≤ Nₕ-1, 1 ≤ j ≤ Nₕ-1,
    M[i,j]	for 1 ≤ i ≤ Nₕ-1, 1 ≤ j ≤ Nₕ-1,
    F[j,n]	for 1 ≤ j ≤ Nₕ-1, 1 ≤ n ≤ Nₜ,
    t[n]	for 0 ≤ n ≤ N_t.
"""
function crank_nicolson!(U::OMat64, M::SymTridiagonal, A::SymTridiagonal, 
	                 F::Mat64, t::OVec64)
    Nₜ = lastindex(t)
    for n = 1:Nₜ
        τ = t[n] - t[n-1]
        B = M + (τ/2) * A
	rhs = τ * F[:,n] - τ * (A * U[1:end-1,n-1])
	W = B \ rhs
        U[0,n] = 0.0
	U[1:end-1,n] = U[1:end-1,n-1] + W
        U[end,n] = 0.0
    end
end

function graded_mesh(Nₜ::Integer, γ::T, final_time::T) where T <: AbstractFloat
    τ = final_time^(1/γ) / Nₜ
    t = OffsetVector{T}(undef, 0:Nₜ)
    t[0] = zero(T)
    for n = 1:Nₜ-1
	t[n] = (n * τ)^γ
    end
    t[Nₜ] = final_time
    return t
end

function weights(α::T, t::OffsetVector{T}) where T <: AbstractFloat
    Nₜ = lastindex(t)
    ω = Vector{Vector{T}}(undef, Nₜ)
    for n in eachindex(ω)
	ω[n] = Vector{T}(undef, n)
    end
    weights!(ω, α, t)
    return ω
end

function weights!(ω::Vector{Vector{T}}, α::T, 
	          t::OffsetVector{T}) where T <: AbstractFloat
    Γ = gamma(3-α)
    τ₁ = t[1] - t[0]
    ω[1][1] = τ₁^(1-α) / Γ
    for n = 2:lastindex(ω)
	τₙ = t[n] - t[n-1]
	for j = 1:n-2
	    τⱼ = t[j] - t[j-1]
	    D = (t[n] + t[n-1])/2 - (t[j] + t[j-1])/2
	    δ⁺ = (τₙ + τⱼ) / (2D)
	    δ⁻ = (τₙ - τⱼ) / (2D)
	    ω[n][j] = D^(2-α) * (  (1 + δ⁺)^(2-α) 
				 - (1 - δ⁻)^(2-α)
				 - (1 + δ⁻)^(2-α)
				 + (1 - δ⁺)^(2-α) ) / ( Γ * τₙ ) 
	end
	j = n-1
        τⱼ = t[j] - t[j-1]
	D = (t[n] + t[n-1])/2 - (t[j] + t[j-1])/2
	δ⁻ = (τₙ - τⱼ) / (2D)
	ω[n][j] = D^(2-α) * ( 2^(2-α) - (1 - δ⁻)^(2-α) 
			              - (1 + δ⁻)^(2-α)) / ( Γ * τₙ ) 
	ω[n][n] = τₙ^(1-α) / Γ
    end
end

function weights(α::Float64, t::OVec64, δ_threshold::Float64, tol::Float64, 
	         max_terms=20)
    Nₜ = lastindex(t)
    ω = Vector{Vec64}(undef, Nₜ)
    for n in eachindex(ω)
	ω[n] = Vec64(undef, n)
    end
    weights!(ω, α, t, δ_threshold, tol, max_terms)
    return ω
end

function weights!(ω::Vector{Vec64}, α::Float64, t::OVec64, 
	          δ_threshold::Float64, tol::Float64, max_terms)
    Γ = gamma(3-α)
    C = Vec64(undef, max_terms)
    δ⁺ = Vec64(undef, max_terms)
    δ⁻ = Vec64(undef, max_terms)
    Taylor_coefficients!(C, α)
    τ₁ = t[1] - t[0]
    ω[1][1] = τ₁^(1-α) / Γ
    for n = 2:lastindex(ω)
        τₙ = t[n] - t[n-1]
	for j = 1:n-2
	    τⱼ = t[j] - t[j-1]
	    D = (t[n] + t[n-1])/2 - (t[j] + t[j-1])/2
            δ⁺[1] = (τₙ + τⱼ) / (2D)
            δ⁻[1] = (τₙ - τⱼ) / (2D)
            if δ⁺[1] > δ_threshold
		ω[n][j] = ( ( D^(2-α) / ( Γ * τₙ ) ) 
			  * ( (1 + δ⁺[1])^(2-α) - (1 - δ⁻[1])^(2-α)
		            - (1 + δ⁻[1])^(2-α) + (1 - δ⁺[1])^(2-α) ) )
	    else
	        ω[n][j] = Taylor_series!(δ⁺, δ⁻, C, α, Γ, τₙ, τⱼ, D, tol)
	    end
	end
	j = n-1
        τⱼ = t[j] - t[j-1]
	D = (t[n] + t[n-1])/2 - (t[j] + t[j-1])/2
	δ⁻[1] = (τₙ - τⱼ) / (2D)
	ω[n][j] = D^(2-α) * ( 2^(2-α) - (1 - δ⁻[1])^(2-α) 
			              - (1 + δ⁻[1])^(2-α)) / ( Γ * τₙ ) 
        ω[n][n] = τₙ^(1-α) / Γ
    end
end

"""
    Taylor_coefficients!(C, α)

Computes the Taylor coefficients `C[m]` from the expansion

                                     ∞
    (1 + δ)^(2-α) + (1 - δ)^(2-α) =  Σ  Cₘ δ²ᵐ.
                                    m=1
"""
function Taylor_coefficients!(C::Vec64, α::Float64) 
    C[1] = (2 - α) * (1 - α) 
    for m = 2:lastindex(C)
	C[m] = C[m-1] * ( (4 - α - 2m) / (2m-1) ) * ( (3 - α - 2m) / (2m) ) 
    end
end

function Taylor_series!(δ⁺::Vec64, δ⁻::Vec64, C::Vec64, α::Float64, Γ::Float64, 
	                τₙ::Float64, τⱼ::Float64, D::Float64, tol::Float64)
    b = D^(1-α) / Γ
    if τₙ < τⱼ
	b *= τ[j] / τ[n]
    end
    δ⁻[1] = abs(δ⁻[1])
    outer_Σ = C[1] * ( δ⁺[1] + δ⁻[1] )
    for m = 2:lastindex(C)
	δ⁺[m] = δ⁺[1] * δ⁺[m-1]
	if C[m] * δ⁺[m]^2 < tol * ( 1 - δ⁺[2] )
	    return b * outer_Σ
	end
	δ⁻[m] = δ⁻[1] * δ⁻[m-1]
	inner_Σ = δ⁺[m-1] 
	for k = 2:m-1
	    inner_Σ += δ⁺[m-k] * δ⁻[k-1]
	end
	inner_Σ += δ⁻[m-1]
	outer_Σ += C[m] * ( δ⁺[m] + δ⁻[m] ) * inner_Σ
    end
    error("Truncation error did not meet specified error tolerance.")
end

end # module
