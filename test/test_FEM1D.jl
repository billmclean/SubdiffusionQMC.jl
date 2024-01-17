using SubdiffusionQMC
import SubdiffusionQMC.FEM1D: load_vector, stiffness_matrix
import GaussQuadrature: legendre
import Printf: @printf
using PyPlot

function FEM_solution(Nₕ::Integer, κ::Function, f::Function, 
	              ξ::Vec64, w::Vec64, ε=0.0)
    x = collect(range(0, 1, Nₕ+1))
    x = OVec64(x, 0:Nₕ)
    x[1:Nₕ-1] .+= (ε/Nₕ) * randn(Nₕ-1)
    F = load_vector(f, x, ξ, w)
    A = stiffness_matrix(κ, x, ξ, w)
    U = OVec64(zeros(Nₕ+1), 0:Nₕ)
    U[1:Nₕ-1] .= A \ F
    return U, x
end

κ(x) = exp(x)
f(x) = π * exp(x) * ( π * sinpi(x) - cospi(x) )
u(x) = sinpi(x)

points, weights = legendre(4)
ξ = ( points .+ 1 ) / 2
w = weights / 2

Nₕ = 10
U, x = FEM_solution(Nₕ, κ, f, ξ, w)

figure(1)
xx = range(0, 1, 201)
plot(xx, u.(xx), x, U, "o")
grid(true)
legend(("Exact solution", "FEM"), loc="lower center")
xlabel(L"$x$")
ylabel(L"$u$")

nrows = 5
max_error = Vec64(undef, nrows)
max_error[1] = maximum(abs, U - u.(x))
@printf("%5s  %10s  %8s\n\n", "Nₕ", "error", "rate")
@printf("%5d  %10.3e  %8s\n", Nₕ, max_error[1], "")
for row = 2:nrows
    global Nₕ
    local U, x
    Nₕ *= 2
    U, x = FEM_solution(Nₕ, κ, f, ξ, w, 0.2)
    max_error[row] = maximum(abs, U - u.(x))
    rate = log2(max_error[row-1] / max_error[row])
    @printf("%5d  %10.3e  %8.3f\n", Nₕ, max_error[row], rate)
end

α = 2.0
resolution = 128
z = 50
κ₀(x) = exp(-x)
min_κ₀ = exp(-1)
f(x) = 1.0

dstore = DiffusivityStore1D(z, α, resolution, min_κ₀)
x = range(0, 1, resolution+2)

M = 5
Nₕ = 50
U = zeros(Nₕ+1, M)
x_FEM = range(0, 1, Nₕ+1)
x_interpolate = range(0, 1, resolution+2)
for sample in 1:M
    local y, κ, κ_
    global U
    y = rand(z) .- 1/2
    κ_ = interpolate_κ!(y, κ₀.(x_interpolate), dstore)
    κ(x) = κ_(x)
    U[:,sample], _ = FEM_solution(Nₕ, κ, f, ξ, w)
end
U_det, _ = FEM_solution(Nₕ, κ₀, f, ξ, w)

figure(2)
plot(x_FEM, U_det, "k", x_FEM, U, ":")
grid(true)
xlabel(L"$x$")
ylabel(L"$u$")
title("Deterministic solution (solid line) and five random solutions")

