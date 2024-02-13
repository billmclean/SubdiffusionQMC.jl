using SubdiffusionQMC
import SubdiffusionQMC.FEM1D: load_vector!, stiffness_matrix, mass_matrix
import GaussQuadrature: legendre
import OffsetArrays: OffsetArray
using PyPlot


function get_load_vector!(F::Vec64, t::Float64, 
	                  f::Function, x::OVec64, ξ::Vec64, w::Vec64)
    load_vector!(x->f(x, t), F, x, ξ, w)
end

function IBVP_solution(x::OVec64, t::OVec64, κ::Function, f::Function,
	               u₀::Function, ξ::Vec64, w::Vec64, ε=0.0)
    Nₕ = lastindex(x) 
    Nₜ = lastindex(t)
    A = stiffness_matrix(κ, x, ξ, w)
    M = mass_matrix(x)
    U = OMat64(zeros(Nₕ+1, Nₜ+1), 0:Nₕ, 0:Nₜ)
    U[:,0] .= u₀.(x)
    crank_nicolson!(U, M, A, t, get_load_vector!, f, x, ξ, w)
    return U
end

Nₕ = 15
Nₜ = 20
T = 1.0
ε = 0.1
x = collect(range(0, 1, Nₕ+1))
x = OVec64(x, 0:Nₕ)
x[1:Nₕ-1] .+= (ε/Nₕ) * randn(Nₕ-1)
t = collect(range(0, T, Nₜ+1))
t = OVec64(t, 0:Nₜ)
t[1:Nₜ-1] .+= (ε/Nₜ) * randn(Nₜ-1)

ξ, w = legendre(4)
ξ = ( ξ .+ 1 ) / 2
w = ξ / 2

κ_const = 0.02
k = 2
λ = κ_const * (k * π)^2
u_homogeneous(x, t) = exp(-λ * t) * sinpi(k * x)
f_homogeneous(x, t) = 0.0
u₀_homogeneous(x) = u_homogeneous(x, 0.0)

U = IBVP_solution(x, t, x -> κ_const, f_homogeneous, u₀_homogeneous, ξ, w)

figure(1)
xx = range(0, 1, 201)
plot(xx, u_homogeneous.(xx, T),
     x, U[:,Nₜ], "o")
xlabel(L"$x$")
grid(true)
title("Solution at t = $T when f ≡ 0")

u_inhomogeneous(x, t) = t * exp(-t) * sinpi(x)
f_inhomogeneous(x, t) = (1 + (κ_const * π^2 - 1) * t) * exp(-t) * sinpi(x)

U = IBVP_solution(x, t, x -> κ_const, f_inhomogeneous, x -> 0.0, ξ, w)

figure(2)
plot(xx, u_inhomogeneous.(xx, T),
     x, U[:,Nₜ], "o")
xlabel(L"$x$")
grid(true)
title("Solution at t = $T when u₀ ≡ 0")

p = 0.5
resolution = 256
z = 50
a = 0.03
κ₀(x) = a * exp(-x)
min_κ₀ = a * exp(-1)
dstore = DiffusivityStore1D(z, p, resolution, min_κ₀)
u₀_bent(x) = 5x^2 * (1 - x)

Nₕ = 40
Nₜ = 50
x = collect(range(0, 1, Nₕ+1))
x = OVec64(x, 0:Nₕ)
t = collect(range(0, T, Nₜ+1))
t = OVec64(t, 0:Nₜ)

U_det = IBVP_solution(x, t, κ₀, f_homogeneous, u₀_bent, ξ, w)
M = 5
z = 30
U = OffsetArray(zeros(Nₕ+1, Nₜ+1, 5), 0:Nₕ, 0:Nₜ, 1:M)
for m = 1:M
    local y
    y = rand(z) .- 1/2
    x_interpolation = range(0, 1, length(dstore.vals))
    κ_ = interpolate_κ!(y, κ₀.(x_interpolation), dstore)
    U[:,:,m] = IBVP_solution(x, t, x -> κ_(x), f_homogeneous, u₀_bent, ξ, w)
end

figure(3)
plot(x, U_det[:,Nₜ], "k", x, U[:,Nₜ,:])
xlabel(L"$x$")
grid(true)
title("Solutions at t = $T with different κ")
