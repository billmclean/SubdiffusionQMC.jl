module SubdiffusionQMC

import FFTW
import SparseArrays
import SimpleFiniteElements
using OffsetArrays

export Vec64, OVec64, Mat64, OMat64
export DiffusivityStore1D, DiffusivityStore2D, ExponentialSumStore, PDEStore 
export PDEStore_integrand
export double_indices, interpolate_κ!, slow_κ
export graded_mesh, weights, weights!, exponential_sum
export generalised_crank_nicolson_1D!, generalised_crank_nicolson_2D!, crank_nicolson_1D!, crank_nicolson_2D!
export SPOD_points, pcg!, cg!
export integrand_init!, integrand!, slow_integrand!
export simulations!, slow_simulations!

const Vec64 = Vector{Float64} 
const OVec64 = OffsetVector{Float64}
const AVec64 = AbstractVector{Float64}
const Mat64 = Matrix{Float64}
const OMat64 = OffsetMatrix{Float64}
const AMat64 = AbstractMatrix{Float64}
const SparseCholeskyFactor = SparseArrays.CHOLMOD.Factor{Float64}
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

struct PDEStore
    κ₀::Function
    dof::SimpleFiniteElements.DegreesOfFreedom
    b::Vec64
    solver::Symbol
    P::SparseCholeskyFactor
    wkspace::Mat64
    u::Vec64
    pcg_tol::Float64
    pcg_maxiterations::Integer
end

struct PDEStore_integrand
    κ₀::Function
    dof::SimpleFiniteElements.DegreesOfFreedom
    b::Vec64
    solver::Symbol
    P::Vector{SparseCholeskyFactor}
    wkspace::Mat64
    u_free_det::Vec64
    u_free::Vec64
    pcg_tol::Float64
    pcg_maxits::Int
    bilinear_forms_M::Dict
end

function double_indices end
function interpolate_κ! end
function slow_κ end
include("submodules/RandomDiffusivity.jl")

include("submodules/FEM1D.jl")

include("submodules/FEM2D.jl")

function graded_mesh end
function weights end
function weights! end
function exponential_sum end
include("submodules/FractionalDerivatives.jl")

function generalised_crank_nicolson_1D! end
function generalised_crank_nicolson_2D! end
function crank_nicolson_1D! end
function crank_nicolson_2D! end
function weights end
include("submodules/Timestepping.jl")

function SPOD_points end
function pcg! end
function cg! end
include("submodules/Utils.jl")

function integrand_init! end
function integrand! end
function slow_integrand! end
include("submodules/PDE.jl")

function simulations! end
function slow_simulations! end
include("submodules/QMC.jl")

end # module SubdiffusionQMC
