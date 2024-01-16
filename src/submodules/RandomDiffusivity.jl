module RandomDiffusivity

import ..RandomDiffusivity1D, ..Vec64

import SpecialFunctions: zeta

function DiffusivityStore(z::Int64, α::Float64, resolution::Integer,
                          min_κ₀::Float64)
    M_α = zeta(α) * min_κ₀
    if lastindex(z) > resolution ÷ 2
        error("Interpolation grid is too coarse.")
    end
    coef = zeros(resolution-2)
    vals = zeros(resolution)
    plan = plan_r2r(coef, RODFT00)
    return DiffusivityStore(α, M_α, coef, vals, plan)
end

function KL_expansion!(y::Vec64, κ₀::Vec64, dstore::DiffusivityStore1D)
    for j in eachindex(y)
        coef[j] = y[j] / (M_α * j^α)
    end
    sin_sum!(vals, coef, plan)
    for j in eachindex(vals)
        vals[j] = κ₀[j] + vals[j]
    end
end

end # module
