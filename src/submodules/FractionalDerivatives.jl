module FractionalDerivatives

import OffsetArrays: OffsetVector
import ..Vec64, ..OVec64, ..OMat64, ..ExponentialSumStore
import ..graded_mesh, ..weights, ..weights!, ..exponential_sum

function graded_mesh(Nₜ::Integer, γ::Float64, final_time::Float64)
    t = OVec64(undef, 0:Nₜ)
    t[0] = 0.0
    τ = final_time^(1/γ) / Nₜ
    for n = 1:Nₜ-1
	t[n] = (n * τ)^γ
    end
    t[Nₜ] = final_time
    return t
end

function ExponentialSumStore(t::OVec64, Nₕ::Integer, α::Float64, r::Integer, 
	                     tol::Float64, Δx::Float64; 
			     δ_threshold=0.2, max_terms=20)
    if r < 2 
        error("r must be at least 2")
    end
    Nₜ = lastindex(t)
    ω = Vector{OVec64}(undef, Nₜ)
    for n = 1:r
        ω[n] = OVec64(undef, 1:n)
    end
    for n = r+1:Nₜ
        ω[n] = OVec64(undef, n-r+1:n)
    end
    weights!(ω, α, t, δ_threshold, tol, max_terms)

    δ = Inf
    for n = r:Nₜ
	next = t[n-1] - t[n-r] 
	δ = min(next, δ)
    end
    M₋ = 1
    while true
	m = -M₋
	wₘ, aₘ = exponential_sum_parameters(m, Δx, α)
	if wₘ * exp(-aₘ * δ/t[Nₜ]) < tol * t[Nₜ]^α
	    break
	else
	    M₋ += 1
	end
    end
    M₊ = 1
    while true
	m = M₊
	wₘ, aₘ = exponential_sum_parameters(m, Δx, α)
	if wₘ * exp(-aₘ * δ/t[Nₜ]) < tol * t[Nₜ]^α
	    break
	else
	    M₊ += 1
	end
    end
    a = OVec64(undef, -M₋:M₊)
    w = OVec64(undef, -M₋:M₊)
    for m in eachindex(w)
	w[m], a[m] = exponential_sum_parameters(m, Δx, α)
	w[m] /= t[Nₜ]^α
	a[m] /= t[Nₜ]
    end

    S = OMat64(undef, 1:Nₕ, -M₋:M₊)
    ExponentialSumStore(α, r, t, ω, S, a, w, δ, tol)
end

function exponential_sum_parameters(m::Integer, Δx::Float64, α::Float64)
    xₘ = m * Δx
    pₘ = xₘ - exp(-xₘ)
    aₘ = exp(pₘ)
    wₘ = exp(α * pₘ) * ( 1 + exp(-xₘ) ) * Δx
    return wₘ, aₘ
end

function exponential_sum(t::Float64, estore::ExponentialSumStore)
    w, a, tol, α = estore.w, estore.a, estore.tol, estore.α
    Σ₋ = 0.0
    M₋ = -1
    for m = -1:-1:firstindex(w)
        next_term = w[m] * exp(-a[m] * t)
        Σ₋ += next_term
        if next_term < tol
            M₋ = -m
            break
        end
    end
    if M₋ == -1
        error("M₋ is too small")
    end

    Σ₊ = 0.0
    M₊ = -1
    for m = 0:lastindex(w)
        next_term = w[m] * exp(-a[m] * t)
        Σ₊ += next_term
        if next_term < tol
            M₊ = m
            break
        end
    end
    if M₊ == -1
        error("M₊ is too small")
    end
    Σ = (sinpi(α) / π ) * (Σ₊ + Σ₋)
    return Σ, M₊, M₋
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
            ω[n][j] = ( ( D^(2-α) / ( Γ * τⱼ ) ) 
                      * ( (1 + δ⁺[1])^(2-α) - (1 - δ⁻[1])^(2-α)
                        - (1 + δ⁻[1])^(2-α) + (1 - δ⁺[1])^(2-α) ) )
	end
	j = n-1
        τⱼ = t[j] - t[j-1]
	D = (t[n] + t[n-1])/2 - (t[j] + t[j-1])/2
	δ⁻ = (τₙ - τⱼ) / (2D)
	ω[n][j] = D^(2-α) * ( 2^(2-α) - (1 - δ⁻)^(2-α) 
			              - (1 + δ⁻)^(2-α)) / ( Γ * τⱼ ) 
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

function weights!(ω::Vector{OVec64}, α::Float64, t::OVec64, 
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
	for j = firstindex(ω[n]):n-2
	    τⱼ = t[j] - t[j-1]
	    D = (t[n] + t[n-1])/2 - (t[j] + t[j-1])/2
            δ⁺[1] = (τₙ + τⱼ) / (2D)
#	    if n == lastindex(ω)
#		println("j = $j, δ⁺ = ", δ⁺[1])
#	    end
            δ⁻[1] = (τₙ - τⱼ) / (2D)
            if δ⁺[1] > δ_threshold
		ω[n][j] = ( ( D^(2-α) / ( Γ * τⱼ ) ) 
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
			              - (1 + δ⁻[1])^(2-α)) / ( Γ * τⱼ ) 
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
	b *= τₙ / τⱼ
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
