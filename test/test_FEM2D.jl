using SubdiffusionQMC
using  SimpleFiniteElements
import SimpleFiniteElements.Utils: gmsh2pyplot
using PyPlot

solver = :direct
path = joinpath("..", "spatial_domains", "unit_square.geo")
gmodel = GeometryModel(path)

essential_bcs = [ ("Gamma", 0.0) ]
h = 0.2
mesh = FEMesh(gmodel, h)
dof = DegreesOfFreedom(mesh, essential_bcs)
f(x₁, x₂) = 10 * exp(-(x₁-0.1)^2-(x₂+0.3)^2)
κ₀(x₁, x₂) = 1.0 + 0.2 * sinpi(x₁ + 2x₂)
pcg_tol, pcg_maxiterations = 1e-8, 100 # not used
pstore = PDEStore(κ₀, f, dof, solver, pcg_tol, pcg_maxiterations)
(;b, P) = pstore
u_free = P \ b
u_fix = zeros(dof.num_fixed)
uh = [ u_free; u_fix]

figure(1)
x₁, x₂, triangles = gmsh2pyplot(dof)
triplot(x₁, x₂, triangles)
axis("equal")

figure(2)
plot_trisurf(x₁, x₂, triangles, uh, cmap="cool")
xlabel(L"$x_1$")
ylabel(L"$x_2$")

κ₀_vec = get_nodal_values(κ₀, dof)
figure(3)
tricontourf(x₁, x₂, triangles, κ₀_vec)
grid(true)
colorbar()
xlabel(L"$x_1$")
ylabel(L"$x_2$")
