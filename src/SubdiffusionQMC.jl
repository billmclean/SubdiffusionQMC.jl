module SubdiffusionQMC

import FFTW
using OffsetArrays

export Vec64, OVec64, Mat64, OMat64
export DiffusivityStore1D, DiffusivityStore2D, ExponentialSumStore 
export double_indices, interpolate_κ!, slow_κ
export graded_mesh, weights, weights!, exponential_sum
export generalised_crank_nicolson!, crank_nicolson!

const Vec64 = Vector{Float64} 
const OVec64 = OffsetVector{Float64}
const Mat64 = Matrix{Float64}
const OMat64 = OffsetMatrix{Float64}
const AMat64 = AbstractMatrix{Float64}
const IdxPair = Tuple{Int64, Int64}

struct DiffusivityStore1D
    p::Float64
    Mₚ::Float64
    coef::Vec64 # Coefficients in the KL expansion.
    vals::Vec64 # Values of KL expansion at interpolation points.
    plan::FFTW.r2rFFTWPlan
end

struct DiffusivityStore2D
    idx::Vector{IdxPair}
    p::Float64
    Mₚ::Float64
    coef::Mat64
    vals::Mat64
    plan::FFTW.r2rFFTWPlan
end

struct ExponentialSumStore
    α::Float64
    r::Integer
    t::OVec64
    ω::Vector{OVec64}
    S::OMat64
    a::OVec64
    w::OVec64
    δ::Float64
    tol::Float64
end

function double_indices end
function interpolate_κ! end
function slow_κ end
include("submodules/RandomDiffusivity.jl")

include("submodules/FEM1D.jl")

function graded_mesh end
function weights end
function weights! end
function exponential_sum end
include("submodules/FractionalDerivatives.jl")

function generalised_crank_nicolson! end
function crank_nicolson! end
function weights end
include("submodules/Timestepping.jl")

end # module SubdiffusionQMC
