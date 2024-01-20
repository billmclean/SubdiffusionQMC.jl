module TimeStepping

import ..Vec64, ..Mat64, ..OVec64, ..OMat64
import ..crank_nicolson!, ..graded_mesh, ..weights
import LinearAlgebra: SymTridiagonal
import OffsetArrays: OffsetVector
import SpecialFunctions: gamma
import LinearAlgebra: mul!, ldiv!, cholesky!, axpby!
import LinearAlgebra.BLAS: scal!

"""
    generalised_crank_nicolson!(U, M, A, α, t, get_load_vector!)

Solves the spatially-discrete subdiffusion problem `Mu̇ + Au = F(t)`.
The Crank-Nicolson scheme is used to compute `U[j,n] ≈ u(x[j], t[n])`,
assuming that the calling program has initialised `U[:,0]` to the initial data, 
and `U[0,n]` and `U[Nₕ,n]` to the boundary data at time `t[n]`.  
The function call

    get_load_vector!(F, t, parameters)

computes the load vector `F` at time `t`.

The array dimensions are as follows.

    U[j,n]	for 0 ≤ j ≤ Nₕ, 0 ≤ n ≤ Nₜ,
    A[i,j]	for 1 ≤ i ≤ Nₕ-1, 1 ≤ j ≤ Nₕ-1,
    M[i,j]	for 1 ≤ i ≤ Nₕ-1, 1 ≤ j ≤ Nₕ-1,
    t[n]	for 0 ≤ n ≤ Nₜ.
"""
function generalized_crank_nicolson!(U::OMat64, M::SymTridiagonal, 
	                             A::SymTridiagonal, α::Float64, t::OVec64, 
				     get_load_vector!::Function, parameters...)
    Nₜ = lastindex(t)
    Nₕ = size(U, 1) - 1
    ω = weights(α, t)
    F = Vec64(undef, Nₕ-1)
    B = SymTridiagonal(zeros(Nₕ-1), zeros(Nₕ-2))
    rhs = similar(F)
    Σ = similar(F)
    for n = 1:Nₜ
        τ = t[n] - t[n-1]
	@. B = ω[n][n] * M + (τ/2) * A
	midpoint = (t[n] + t[n-1]) / 2
	get_load_vector!(F, midpoint, parameters...)
	rhs .= τ * F - (τ/2) * (A * U[1:end-1,n-1])
	Σ .= ω[n][1] * U[1:Nₕ-1,0]
	for j = 1:n-1
	    Σ .+= (ω[n][j+1] - ω[n][j]) * U[1:Nₕ-1,j]
	end
	rhs .= rhs + M * Σ
	U[1:Nₕ-1,n] = B \ rhs
    end
end

"""
    crank_nicolson!(U, M, A, t, get_load_vector!)

Solves the spatially-discrete diffusion problem `Mu̇ + Au = F(t)`.
The Crank-Nicolson scheme is used to compute `U[j,n] ≈ u(x[j], t[n])`,
assuming that the calling program has initialised `U[:,0]` to the initial data, 
and `U[0,n]` and `U[Nₕ,n]` to the boundary data at time `t[n]`.  
The function call

    get_load_vector!(F, t, parameters)

computes the load vector `F` at time `t`.

The array dimensions are as follows.

    U[j,n]	for 0 ≤ j ≤ Nₕ, 0 ≤ n ≤ Nₜ,
    A[i,j]	for 1 ≤ i ≤ Nₕ-1, 1 ≤ j ≤ Nₕ-1,
    M[i,j]	for 1 ≤ i ≤ Nₕ-1, 1 ≤ j ≤ Nₕ-1,
    t[n]	for 0 ≤ n ≤ Nₜ.
"""
function crank_nicolson!(U::OMat64, M::SymTridiagonal, A::SymTridiagonal, 
	                 t::OVec64, get_load_vector!::Function,
			 parameters...)
    Nₜ = lastindex(t)
    Nₕ = lastindex(U, 1)
    F = Vec64(undef, Nₕ-1)
    B = SymTridiagonal(zeros(Nₕ-1), zeros(Nₕ-2))
    rhs = similar(F)
    for n = 1:Nₜ
        τ = t[n] - t[n-1]
        @. B = M + (τ/2) * A
	midpoint = (t[n] + t[n-1]) / 2
	get_load_vector!(F, midpoint, parameters...)
	mul!(rhs, A, U[1:Nₕ-1,n-1])
	rhs .= F - rhs
	scal!(τ, rhs) # rhs = τ F - τ A Uⁿ⁻¹
	ldiv!(B, rhs) # rhs = ΔUⁿ
	U[1:Nₕ-1,n] .= U[1:Nₕ-1,n-1] .+ rhs
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
#	    if n == lastindex(ω)
#		println("j = $j, δ⁺ = ", δ⁺[1])
#	    end
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
    if τₙ > τⱼ
	b *= τⱼ / τₙ
    end
    δ⁻[1] = abs(δ⁻[1])
    outer_Σ = C[1] * ( δ⁺[1] + δ⁻[1] )
    for m = 2:lastindex(C)
	δ⁺[m] = δ⁺[1] * δ⁺[m-1]
	if C[m-1] * δ⁺[m]^2 < tol * ( 1 - δ⁺[2] )
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
