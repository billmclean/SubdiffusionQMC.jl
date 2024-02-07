using SubdiffusionQMC
using Printf
using PyPlot

p = 1/2
resolution = (256, 256)
n = 15
idx = double_indices(n)
z = lastindex(idx)
κ₀(x₁, x₂) = 2 + x₁ * x₂
min_κ₀ = 2.0
dstore = DiffusivityStore2D(idx, z, p, resolution, min_κ₀)

y = rand(z) .- 1/2

N₁, N₂ = resolution
@printf("Using %d x %d interpolation grid.\n", N₁, N₂)

x₁_vals = range(0, 1, N₁)
x₂_vals = range(0, 1, N₂)
#x₁_vals = range(0, 1, 2(N₁+1)+1)
#x₂_vals = range(0, 1, 2(N₂+1)+1)
κ₀_vals = Float64[ κ₀(x₁, x₂) for x₁ in x₁_vals, x₂ in x₂_vals ]
κ = interpolate_κ!(y, κ₀_vals, dstore)
κ_vals = Float64[ κ(x₁, x₂) for x₁ in x₁_vals, x₂ in x₂_vals ]

slow_κ_vals = Float64[ slow_κ(x₁, x₂, y, κ₀, dstore) 
		       for x₁ in x₁_vals, x₂ in x₂_vals ]
#max_error = maximum(abs, κ_vals - slow_κ_vals)
max_error = maximum(abs, κ_vals - slow_κ_vals)
@printf("maximum interpolation error in sample = %0.3e\n", max_error)

figure(1)
contourf(x₁_vals, x₂_vals, κ₀_vals)
grid(true)
title("κ₀(x₁, x₂)")
colorbar()
xlabel(L"$x_1$")
ylabel(L"$x_2$")

figure(2)
contourf(x₁_vals, x₂_vals, κ_vals)
grid(true)
title("κ(x₁, x₂)")
colorbar()
xlabel(L"$x_1$")
ylabel(L"$x_2$")

