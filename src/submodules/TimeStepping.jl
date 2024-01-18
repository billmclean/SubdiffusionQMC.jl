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

#function weights!(ω::Vector{Vec64}, α::Float, t::OVec64, tol::Float64)

#end

end # module
