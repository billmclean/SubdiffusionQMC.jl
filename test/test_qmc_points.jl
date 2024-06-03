using DelimitedFiles
using SubdiffusionQMC
using PyPlot

path = joinpath("..", "qmc_points", "SPOD_dim256.jld2")
Nvals, pts = SPOD_points(3, path)

k = 4
z = pts[k]

figure(1)
axis("off")
scatter3D(z[1,:], z[2,:], z[3,:], color="c")
xlabel(L"$x_1$")
ylabel(L"$x_2$")
zlabel(L"$x_3$")
s = latexstring("\$N = $(Nvals[k])\$ QMC Points in 3D")
title(s)

#Nvals, pts = SPOD_points(2, path)
#k = 5
#z = pts[k]
#figure(2)
#plot(z[1,:], z[2, :], "o")
#s = latexstring("\$N = $(Nvals[k])\$ QMC Points in 2D")
#title(s)