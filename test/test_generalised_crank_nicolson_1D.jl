using SubdiffusionQMC
import SubdiffusionQMC.FEM1D: load_vector!, stiffness_matrix, mass_matrix
import GaussQuadrature: legendre
import SpecialFunctions: erfcx
using PyPlot
import Printf: @printf

fast_method = true
if fast_method
    @printf("Using exponential sum approximation.\n")
else
    @printf("Using direct evaluation of the memory term.\n")
end

E_half(x) = erfcx(-x)

function get_load_vector!(F::Vec64, t::Float64, 
	                  f::Function, x::OVec64, ξ::Vec64, w::Vec64)
    load_vector!(x->f(x, t), F, x, ξ, w)
end

function IBVP_solution(x::OVec64, t::OVec64, α::Float64, κ::Float64, 
	f::Function, u₀::Function, ξ::Vec64, w::Vec64)
    Nₕ = lastindex(x) 
    Nₜ = lastindex(t)
    A = stiffness_matrix(x -> κ, x, ξ, w)
    M = mass_matrix(x)
    U = OMat64(zeros(Nₕ+1, Nₜ+1), 0:Nₕ, 0:Nₜ)
    U[:,0] .= u₀.(x)
    generalised_crank_nicolson!(U, M, A, α, t, get_load_vector!, f, x, ξ, w)
    return U
end

function IBVP_solution(x::OVec64, κ::Float64, 
	f::Function, u₀::Function, ξ::Vec64, w::Vec64, 
	estore::ExponentialSumStore)
    (;t) = estore
    Nₕ = lastindex(x) 
    Nₜ = lastindex(t)
    A = stiffness_matrix(x -> κ, x, ξ, w)
    M = mass_matrix(x)
    U = OMat64(zeros(Nₕ+1, Nₜ+1), 0:Nₕ, 0:Nₜ)
    U[:,0] .= u₀.(x)
    generalised_crank_nicolson!(U, M, A, get_load_vector!, estore, f, x, ξ, w)
    return U
end

α = 0.5
κ = 0.02
k = 1
λ = κ * (k * π)^2
u_homogeneous(x, t) = E_half(-λ * sqrt(t)) * sinpi(k * x)
f_homogeneous(x, t) = 0.0
u₀_homogeneous(x) = u_homogeneous(x, 0.0)

Nₕ = 16
Nₜ = 20
T = 1.0
ε = 0.1
γ = 2 / α
#γ = 1.0
x = collect(range(0, 1, Nₕ+1))
x = OVec64(x, 0:Nₕ)
x[1:Nₕ-1] .+= (ε/Nₕ) * randn(Nₕ-1)
t = graded_mesh(Nₜ, γ, T)

ξ, w = legendre(4) # Gauss-Legendre rule on (-1, 1).
ξ = ( ξ .+ 1 ) / 2 # Shifted to (0, 1).
w = w / 2

if fast_method
    r = 2
    tol = 1e-8
    Δx = 0.5
    estore = ExponentialSumStore(t, Nₕ, α, r, tol, Δx)
    U = IBVP_solution(x, κ, f_homogeneous, u₀_homogeneous, ξ, w, estore)
else
    U = IBVP_solution(x, t, α, κ, f_homogeneous, u₀_homogeneous, ξ, w)
end

figure(1)
xx = range(0, 1, 201)
plot(xx, u_homogeneous.(xx, T),
     x, U[:,Nₜ], "o")
xlabel(L"$x$")
grid(true)
title("Solution at t = $T when f ≡ 0")

Nₜ = 40
Nₕ = 40
nrows = 5
max_error = zeros(nrows)
@printf("\n%6s  %6s  %10s  %8s  %8s\n\n", 
	"Nₜ", "Nₕ", "Error", "rate", "seconds")
for row = 1:nrows
    local U, x, t, r, tol, Δx, estore
    global Nₕ, Nₜ
    Nₜ *= 2
    Nₕ *= 2
    x = collect(range(0, 1, Nₕ+1))
    x = OVec64(x, 0:Nₕ)
    t = graded_mesh(Nₜ, γ, T)
    start = time()
    if fast_method
        r = 2
        tol = 1e-8
        Δx = 0.5
        estore = ExponentialSumStore(t, Nₕ, α, r, tol, Δx)
        U = IBVP_solution(x, κ, f_homogeneous, u₀_homogeneous, ξ, w, estore)
    else
        U = IBVP_solution(x, t, α, κ, f_homogeneous, u₀_homogeneous, ξ, w)
    end
    elapsed = time() - start
    max_error[row] = maximum(abs, U - u_homogeneous.(x, t'))
    if row == 1
	@printf("%6d  %6d  %10.2e  %8s  %8.3f\n", 
		Nₜ, Nₕ, max_error[row], "", elapsed)
    else
	rate = log2(max_error[row-1]/max_error[row])
	@printf("%6d  %6d  %10.2e  %8.3f  %8.3f\n", 
		Nₜ, Nₕ, max_error[row], rate, elapsed)
    end
end

