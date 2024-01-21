using SubdiffusionQMC
import SubdiffusionQMC.FEM1D: load_vector!, stiffness_matrix, mass_matrix
import GaussQuadrature: legendre
import SpecialFunctions: erfcx
using PyPlot

E_half(x) = erfcx(-x)

function get_load_vector!(F::Vec64, t::Float64, 
	                  f::Function, x::OVec64, ξ::Vec64, w::Vec64)
    load_vector!(x->f(x, t), F, x, ξ, w)
end

function IBVP_solution(x::OVec64, t::OVec64, α::Float64, κ::Float64, 
	f::Function, u₀::Function, ξ::Vec64, w::Vec64, ε=0.0)
    Nₕ = lastindex(x) 
    Nₜ = lastindex(t)
    A = stiffness_matrix(x -> κ, x, ξ, w)
    M = mass_matrix(x)
    U = OMat64(zeros(Nₕ+1, Nₜ+1), 0:Nₕ, 0:Nₜ)
    U[:,0] .= u₀.(x)
    generalised_crank_nicolson!(U, M, A, α, t, get_load_vector!, f, x, ξ, w)
    return U
end

Nₕ = 15
Nₜ = 40
T = 1.0
ε = 0.1
γ = 2.0
x = collect(range(0, 1, Nₕ+1))
x = OVec64(x, 0:Nₕ)
x[1:Nₕ-1] .+= (ε/Nₕ) * randn(Nₕ-1)
t = graded_mesh(Nₜ, γ, T)

ξ, w = legendre(4) # Gauss-Legendre rule on (-1, 1).
ξ = ( ξ .+ 1 ) / 2 # Shifted to (0, 1).
w = w / 2

α = 0.5
κ = 0.02
k = 1
λ = κ * (k * π)^2
u_homogeneous(x, t) = E_half(-λ * sqrt(t)) * sinpi(k * x)
f_homogeneous(x, t) = 0.0
u₀_homogeneous(x) = u_homogeneous(x, 0.0)

U = IBVP_solution(x, t, α, κ, f_homogeneous, u₀_homogeneous, ξ, w)

figure(1)
xx = range(0, 1, 201)
plot(xx, u_homogeneous.(xx, T),
     x, U[:,Nₜ], "o")
xlabel(L"$x$")
grid(true)
title("Solution at t = $T when f ≡ 0")

#u_inhomogeneous(x, t) = t * exp(-t) * sinpi(x)
#f_inhomogeneous(x, t) = (1 + (κ * π^2 - 1) * t) * exp(-t) * sinpi(x)

#U = IBVP_solution(x, t, κ, f_inhomogeneous, x -> 0.0, ξ, w)

#figure(2)
#plot(xx, u_inhomogeneous.(xx, T),
#     x, U[:,Nₜ], "o")
#xlabel(L"$x$")
#grid(true)
#title("Solution at t = $T when u₀ ≡ 0")

