module Timestepping

import ..Vec64, ..Mat64, ..OVec64, ..OMat64, ..AMat64, ..ExponentialSumStore
import ..generalised_crank_nicolson!, 
       ..crank_nicolson_1D!, ..crank_nicolson_2D!,
       ..graded_mesh, ..weights
import LinearAlgebra: SymTridiagonal, norm
import OffsetArrays: OffsetVector
import SpecialFunctions: gamma
import LinearAlgebra: mul!, ldiv!, cholesky!, axpby!, cholesky
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
function generalised_crank_nicolson!(U::OMat64, M::SymTridiagonal, 
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
	rhs .= τ .* F - (τ/2) .* (A * U[1:end-1,n-1])
	Σ .= ω[n][1] * U[1:Nₕ-1,0]
	for j = 1:n-1
	    Σ .+= (ω[n][j+1] - ω[n][j]) .* U[1:Nₕ-1,j]
	end
	rhs .= rhs + M * Σ
	U[1:Nₕ-1,n] = B \ rhs
    end
end

function generalised_crank_nicolson!(U::OMat64, M::SymTridiagonal, 
	                             A::SymTridiagonal, 
				     get_load_vector!::Function, 
				     estore::ExponentialSumStore, parameters...)
    (;α, r, t, ω, S, a, w) = estore
    Nₜ = lastindex(t)
    Nₕ = size(U, 1) - 1
    F = Vec64(undef, Nₕ-1)
    B = SymTridiagonal(zeros(Nₕ-1), zeros(Nₕ-2))
    rhs = similar(F)
    Σ = similar(F)
    for n = 1:r
        τₙ = t[n] - t[n-1]
	@. B = ω[n][n] * M + (τₙ/2) * A
	midpoint = (t[n] + t[n-1]) / 2
	get_load_vector!(F, midpoint, parameters...)
	rhs .= τₙ .* F - (τₙ/2) .* (A * U[1:end-1,n-1])
	Σ .= ω[n][1] .* U[1:Nₕ-1,0]
	for j = 1:n-1
	    Σ .+= (ω[n][j+1] - ω[n][j]) .* U[1:Nₕ-1,j]
	end
	rhs .= rhs + M * Σ
	U[1:Nₕ-1,n] = B \ rhs
    end
    fill!(S, 0.0)
    for n = r+1:Nₜ
        τₙ   = t[n] - t[n-1]
	τₙ₋₁ = t[n-1] - t[n-2]
	τₙ₋ᵣ = t[n-r] - t[n-r-1]
	@. B = ω[n][n] * M + (τₙ/2) * A
	midpoint = (t[n] + t[n-1]) / 2
	get_load_vector!(F, midpoint, parameters...)
	rhs .= τₙ .* F - (τₙ/2) .* (A * U[1:end-1,n-1])
	fill!(Σ, 0.0)
	for m = firstindex(a):lastindex(a)
	    c = expm1(-a[m] * τₙ)
	    μ = ( w[m] * exp(-a[m] * (t[n-1] - t[n-r])) * (c / a[m])
		       * (expm1(-a[m] * τₙ₋ᵣ) / (a[m] * τₙ₋ᵣ)) )
	    ν = -c / expm1(a[m] * τₙ₋₁)
	    @. S[:,m] = μ * (U[1:Nₕ-1,n-r] - U[1:Nₕ-1,n-r-1]) + ν * S[:,m]
	    Σ .+= S[:,m]
	end
	@. Σ = ω[n][n-r+1] * U[1:Nₕ-1,n-r] - (sinpi(α) / π) * Σ
	for j = n-r+1:n-1
	    Σ .+= (ω[n][j+1] - ω[n][j]) .* U[1:Nₕ-1,j]
	end
	rhs .= rhs + M * Σ
	U[1:Nₕ-1,n] = B \ rhs
    end
end

"""
    crank_nicolson_1D!(U, M, A, t, get_load_vector!)

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
function crank_nicolson_1D!(U::OMat64, M::AMat64, A::AMat64, 
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

function crank_nicolson_2D!(U::OMat64, M::AMat64, A::AMat64, 
                            t::OVec64, get_load_vector!::Function,
                            parameters...)
    Nₜ = lastindex(t)
    Nₕ = lastindex(U, 1)
    F = Vec64(undef, Nₕ)
    rhs = similar(F)
    for n = 1:Nₜ
      τ = t[n] - t[n-1]
      B = M + (τ/2) * A
      midpoint = (t[n] + t[n-1]) / 2
      get_load_vector!(F, midpoint, parameters...)
      mul!(rhs, A, U[1:Nₕ,n-1])
      rhs .= F - rhs
      scal!(τ, rhs) # rhs = τ F - τ A Uⁿ⁻¹
      #ldiv!(B, rhs) # rhs = ΔUⁿ
      ΔUⁿ = B \ rhs
      U[1:Nₕ,n] .= U[1:Nₕ,n-1] .+ ΔUⁿ
    end
end

function crank_nicolson_2D!(U::OMat64, M::AMat64, A::AMat64, 
    T::Float64, Nₜ::Integer, get_load_vector!::Function,
    parameters...)
    t = collect(range(0, T, Nₜ+1))
    t = OVec64(t, 0:Nₜ)
    Nₕ = lastindex(U, 1)
    F = Vec64(undef, Nₕ)
    rhs = similar(F)
    τ = T / Nₜ
    B = M + (τ/2) * A
    R = cholesky(B)
    for n = 1:Nₜ
        midpoint = (t[n] + t[n-1]) / 2
        get_load_vector!(F, midpoint, parameters...)
        mul!(rhs, A, U[1:Nₕ,n-1])
        rhs .= F - rhs
        scal!(τ, rhs) # rhs = τ F - τ A Uⁿ⁻¹
        #ldiv!(B, rhs) # rhs = ΔUⁿ
        ΔUⁿ = R \ rhs
        U[1:Nₕ,n] .= U[1:Nₕ,n-1] .+ ΔUⁿ
    end
end

function euler_2D!(U::OMat64, M::AMat64, A::AMat64, 
	           t::OVec64, get_load_vector!::Function,
		   parameters...)
    Nₜ = lastindex(t)
    Nₕ = lastindex(U, 1)
    F = Vec64(undef, Nₕ)
    rhs = similar(F)
    for n = 1:Nₜ
        τ = t[n] - t[n-1]
        B = M + τ * A
        get_load_vector!(F, t[n], parameters...)
        rhs = τ * F + M * U[1:Nₕ,n-1]
        U[1:Nₕ,n] = B \ rhs
    end
end
end # module
