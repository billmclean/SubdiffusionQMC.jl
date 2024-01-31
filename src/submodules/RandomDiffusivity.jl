module RandomDiffusivity

import ..DiffusivityStore1D, ..Vec64, ..interpolate_κ!
import FFTW: plan_r2r, RODFT00, r2rFFTWPlan
using Interpolations: cubic_spline_interpolation
import SpecialFunctions: zeta
import ArgCheck: @argcheck

function DiffusivityStore1D(z::Int64, α::Float64, resolution::Integer,
                          min_κ₀::Float64)
    M_α = zeta(α) / min_κ₀
    if lastindex(z) > resolution ÷ 2
        error("Interpolation grid is too coarse.")
    end
    coef = zeros(resolution)
    vals = zeros(resolution+2)
    plan = plan_r2r(coef, RODFT00)
    return DiffusivityStore1D(α, M_α, coef, vals, plan)
end

function interpolate_κ!(y::Vec64, κ₀::Vec64, dstore::DiffusivityStore1D)
    vals = dstore.vals
    @argcheck length(κ₀) == length(vals)
    KL_expansion!(y, κ₀, dstore)
    x = range(0, 1, length(vals))
    cubic_spline_interpolation(x, vals)
end

function KL_expansion!(y::Vec64, κ₀::Vec64, dstore::DiffusivityStore1D)
    (;α, M_α, coef, vals, plan) = dstore
    for j in eachindex(y)
        coef[j] = y[j] / (M_α * j^α)
    end
    sin_sum!(vals, coef, plan)
    for j in eachindex(vals)
        vals[j] = κ₀[j] + vals[j]
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

end # module
