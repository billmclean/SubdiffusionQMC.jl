using SubdiffusionQMC
using PyPlot

p = 1/2
resolution = 256
z = 50
κ₀(x) = exp(-x)
min_κ₀ = exp(-1)

dstore = DiffusivityStore1D(z, p, resolution, min_κ₀)
x = range(0, 1, resolution+2)


figure(1)
K = zeros(resolution+2, 5)
for j = 1:5
    local y
    y = rand(z) .- 1/2
    K[:,j] = interpolate_κ!(y, κ₀.(x), dstore)(x)
end
plot(x, κ₀.(x), x, K, ":")
axis([0, 1, 0, 1.2])
grid(true)
title("κ₀ and κ(⋅,y) for five choices of y")
