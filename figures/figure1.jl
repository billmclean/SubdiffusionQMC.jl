using SubdiffusionQMC
using SimpleFiniteElements
import SimpleFiniteElements.Utils: gmsh2pyplot
using JLD2
using PyPlot
using Printf

u₀_bent(x, y) = 144 * (x^2 * (1 - x) * y^2 * (1 - y))

#figure(1, figsize=(6,4))
figure(1)
xx = range(0, 1, length=100)
yy = range(0, 1, length=100)

contourf(xx, yy, u₀_bent.(xx',yy), 10)
axis("equal")
xlabel(L"$x_1$")
ylabel(L"$x_2$")
title(L"$g(\mathbf{x},\mathbf{y})$")
colorbar()
savefig("fig1.pdf")
