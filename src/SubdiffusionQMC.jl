module SubdiffusionQMC

import FFTW
using OffsetArrays

export Vec64, OVec64, Mat64, OMat64
export DiffusivityStore1D, interpolate_κ!
export generalised_crank_nicolson!, crank_nicolson!, graded_mesh, weights

const Vec64 = Vector{Float64} 
const OVec64 = OffsetVector{Float64}
const Mat64 = Matrix{Float64}
const OMat64 = OffsetMatrix{Float64}
const AMat64 = AbstractMatrix{Float64}

struct DiffusivityStore1D
    α::Float64
    M_α::Float64
    coef::Vec64 # Coefficients in the KL expansion.
    vals::Vec64 # Values of KL expansion at interpolation points.
    plan::FFTW.r2rFFTWPlan
end

function interpolate_κ! end
include("submodules/RandomDiffusivity.jl")

include("submodules/FEM1D.jl")

function generalised_crank_nicolson! end
function crank_nicolson! end
function graded_mesh end
function weights end
include("submodules/Timestepping.jl")

end # module SubdiffusionQMC
