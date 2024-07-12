using SimpleFiniteElements
import SimpleFiniteElements.Poisson: ∫∫a_∇u_dot_∇v!, ∫∫f_v!, ∫g_v!
import SimpleFiniteElements.Utils: gmsh2pyplot
import LinearAlgebra: factorize
using PyPlot
using Printf

path = joinpath("..", "spatial_domains", "circle_domain.geo")
gmodel = GeometryModel(path)

function solve_poisson_equation(a, f, gD, hmax)
    # Bilinear forms
    bilinear_forms = Dict("Omega" => (∫∫a_∇u_dot_∇v!, a))

    # Linear functions
    linear_funcs = Dict("Omega" => (∫∫f_v!, f))

    # Dirichlet boundary condition
    essential_bcs = [("Gamma", gD)]

    # Generate mesh
    mesh = FEMesh(gmodel, hmax)
    dof = DegreesOfFreedom(mesh, essential_bcs)

    # Assemble matrices and vectors
    A_free, A_fix = assemble_matrix(dof, bilinear_forms)
    b_free, u_fix = assemble_vector(dof, linear_funcs)

    # Combine fixed and free parts
    b = b_free - A_fix * u_fix

    # Solve the linear system
    F = factorize(A_free)
    u_free = F \ b
    uh = [u_free; u_fix]

    # Convert mesh information for plotting
    x, y, triangles = gmsh2pyplot(dof)

    return x, y, triangles, uh, dof
end

# Define Poisson equation coefficients
a(x, y) = 1.0  # Diffusion coefficient
f(x, y) = 4.0  # Source term
u(x, y) = 1 - x^2 -y^2
# Dirichlet boundary condition
gD(x, y) = 0.0

# Maximum element size
hmax = 0.5

# Solve the Poisson equation
x, y, triangles, uh, dof = solve_poisson_equation(a, f, gD, hmax)
exact_u = get_nodal_values(u, dof)

# Plot results
figure(1)
triplot(x, y, triangles)
axis("equal")

figure(2)
tricontourf(x, y, triangles, uh, 20)
grid(true)
colorbar()
axis("equal")

figure(3)
plot_trisurf(x, y, triangles, uh, cmap="cool")
xlabel("x")
ylabel("y")
zlabel("u")
title("Finite Element Solution uₕ")

figure(4)
plot_trisurf(x, y, triangles, exact_u, cmap="cool", alpha=0.5)
grid(true)
xlabel(L"$x$")
ylabel(L"$y$")
zlabel(L"$u$")
title("Analytical Solution")

nrows = 5
max_err = zeros(nrows)
max_err[1] = maximum(abs, uh - exact_u)
@printf("%5s  %10s %10s %8s\n\n", "Nₕ", "hmax", "error", "rate")
@printf("%5d  %10.3e %10.3e %8s\n", dof.num_free, hmax, max_err[1], "")
for row = 2:nrows
    global hmax
    local uh, x, y, triangles, dof, exact_u
    hmax /= 2
    x, y, triangles, uh, dof = solve_poisson_equation(a, f, gD, hmax)
    exact_u = get_nodal_values(u, dof)
    max_err[row] = maximum(abs, uh - exact_u)
    rate = log2(max_err[row-1] / max_err[row])
    @printf("%5d  %10.3e %10.3e %8.3f\n", dof.num_free, hmax, max_err[row], rate)
end
