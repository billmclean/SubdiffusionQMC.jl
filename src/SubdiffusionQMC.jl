module SubdiffusionQMC

import FFTW

export DiffusivityStore1D

const Vec64 = Vector{Float64}

struct DiffusivityStore1D
    α::Float64
    M_α::Float64
    coef::Vec64 # Coefficients in the KL expansion.
    vals::Vec64 # Values of KL expansion at interpolation points.
    plan::FFTW.r2rFFTWPlan
end

include("submodules/RandomDiffusivity.jl")

end # module SubdiffusionQMC
