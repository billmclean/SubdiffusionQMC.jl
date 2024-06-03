module RandomDiffusivity

import ..DiffusivityStore1D, ..DiffusivityStore2D, 
       ..IdxPair, ..Vec64, ..AVec64, ..Mat64, ..double_indices, ..interpolate_κ!,
       ..slow_κ
import FFTW: plan_r2r, RODFT00, r2rFFTWPlan
using Interpolations: cubic_spline_interpolation
import SpecialFunctions: zeta
import ArgCheck: @argcheck

function DiffusivityStore1D(z::Int64, p::Float64, resolution::Integer,
                          min_κ₀::Float64)
    Mₚ = zeta(1/p) / min_κ₀
    if z > resolution 
        error("Interpolation grid is too coarse.")
    end
    coef = zeros(resolution)
    vals = zeros(resolution+2)
    plan = plan_r2r(coef, RODFT00)
    DiffusivityStore1D(p, Mₚ, coef, vals, plan)
end

function double_indices(n::Int64)
    idx = Vector{IdxPair}(undef, n*(n+1) ÷ 2)
    j = 1
    for k_plus_l = 2:n+1
        for l = 1:k_plus_l-1
            k = k_plus_l - l
            idx[j] = (k, l)
            j += 1
        end
    end
    return idx
end

function DiffusivityStore2D(idx::Vector{IdxPair}, z::Int64, p::Float64, 
	resolution::IdxPair, min_κ₀::Float64)
    s = 2 / p
    Mₚ = (zeta(s-1) - zeta(s)) / min_κ₀
    max_k₁, max_k₂ = 0, 0
    for j in eachindex(idx)
	k₁, k₂ = idx[j]
	max_k₁ = max(k₁, max_k₁)
	max_k₂ = max(k₂, max_k₂)
    end
    N₁, N₂ = resolution
    if max_k₁ > N₁÷2 || max_k₂ > N₂÷2
        error("Interpolation grid is too coarse to resolve Fourier modes")
    end
    coef = zeros(N₁-2, N₂-2)
    vals = zeros(N₁, N₂)
    plan = plan_r2r(coef, RODFT00)
    DiffusivityStore2D(idx, p, Mₚ, coef, vals, plan)
end

function interpolate_κ!(y::Vec64, κ₀::Vec64, dstore::DiffusivityStore1D)
    vals = dstore.vals
    @argcheck length(κ₀) == length(vals)
    KL_expansion!(y, κ₀, dstore)
    x = range(0, 1, length(vals))
    cubic_spline_interpolation(x, vals)
end

function interpolate_κ!(y::AVec64, κ₀::Mat64, dstore::DiffusivityStore2D)
    (; idx, vals) = dstore
    @argcheck length(idx) == length(y)
    KL_expansion!(y, κ₀, dstore)
    N₁, N₂ = size(vals)
    x₁ = range(0, 1, length=N₁)
    x₂ = range(0, 1, length=N₂)
    cubic_spline_interpolation((x₁, x₂), vals)
end

function slow_κ(x₁::Float64, x₂::Float64, y::Vec64, κ₀::Function,
	dstore::DiffusivityStore2D)
    (; idx, p, Mₚ) = dstore
    Σ = 0.0
    for j in eachindex(idx)
	k₁, k₂ = idx[j]
	Σ += y[j] * sinpi(k₁ * x₁) * sinpi(k₂ * x₂) / (k₁ + k₂)^(2/p)
    end
    return κ₀(x₁, x₂) + Σ / Mₚ
end

function KL_expansion!(y::AVec64, κ₀::Vec64, dstore::DiffusivityStore1D)
    (;p, Mₚ, coef, vals, plan) = dstore
    for j in eachindex(y)
	coef[j] = y[j] / j^(1/p)
    end
    sin_sum!(vals, coef, plan)
    for j in eachindex(vals)
        vals[j] = κ₀[j] + vals[j] / Mₚ
    end
end

function KL_expansion!(y::AVec64, κ₀::Mat64, dstore::DiffusivityStore2D)
    (;idx, p, Mₚ, coef, vals, plan) = dstore
    fill!(coef, 0.0)
    for j in eachindex(idx)
	k₁, k₂ = idx[j]
	coef[k₁, k₂] = y[j] / (k₁ + k₂)^(2/p)
    end
    sin_sin_sum!(vals, coef, plan)
    N₁, N₂ = size(vals)
    for j = 1:N₂, i = 1:N₁
        vals[i,j] = κ₀[i,j] + vals[i,j] / Mₚ
    end
end

"""
    sin_sum!(S, a, plan)

Computes the sin sum

            N
    S[j] =  Σ  aₖ sin(kπxⱼ)    for xⱼ = j/(N+1)   and   0 ≤ j ≤ N+1.
           k=1

For best preformance, `N = length(a)` should be a power of 2.
"""
function sin_sum!(S::Vec64, a::Vec64, plan::r2rFFTWPlan)
    S[1] = 0.0
    S[end] = 0.0
    S[2:end-1] .= plan * a / 2
end

"""
    sin_sin_sum!(S, a, plan)

Computes the double sin sum

             N₁-1  N₂-1
    S[i, j] =  Σ     Σ    aₖₗ sin(kπx₁) sin(lπx₂)
             k = 1 l = 1

for

    (x₁,x₂) = (i/N₁, j/N₂), 0 ≤ i ≤ N₁,  0 ≤ j ≤ N₂.
"""
function sin_sin_sum!(S::Mat64, a::Mat64, plan::r2rFFTWPlan)
    S[1,:] .= 0.0
    S[:,1] .= 0.0
    S[end,:] .= 0.0
    S[:,end] .= 0.0
    S[2:end-1, 2:end-1] .= plan * a / 4
end

end # module
