using SimpleFiniteElements
import SimpleFiniteElements.Utils: gmsh2pyplot
import SimpleFiniteElements.FEM: assemble_vector!
using SubdiffusionQMC
import OffsetArrays: OffsetArray
using PyPlot
using Printf
import SimpleFiniteElements.Poisson: ∫∫a_∇u_dot_∇v!, ∫∫c_u_v!, ∫∫f_v!
using LinearAlgebra
import SubdiffusionQMC.Timestepping: euler_2D!

# Define the solver and path to the geometry file
#solver = :pcg
solver = :direct
path = joinpath("..", "spatial_domains", "unit_square.geo")

# Create the geometry model and specify essential boundary conditions
gmodel = GeometryModel(path)
essential_bcs = [("Gamma", 0.0)]
    
# Specify the mesh size and create the finite element mesh
h = 0.1
mesh = FEMesh(gmodel, h)
    
# Determine the degrees of freedom
dof = DegreesOfFreedom(mesh, essential_bcs)
    
# Create a PDEStore object to store information about the PDE
pcg_tol, pcg_maxiterations = 1e-8, 100 #not used

function get_load_vector!(F::Vec64, t::Float64, pstore::PDEStore, f::Function)
#    linear_functionals = Dict("Omega" => (∫∫f_v!, (x, y) -> f(x, y, t)))
#    F[:], u_fix = assemble_vector(pstore.dof, linear_functionals)
    fill!(F, 0.0)
    assemble_vector!(F, "Omega", [(∫∫f_v!, (x, y) -> f(x, y, t))], dof, 1)
#    println("t = $t, ||F|| = $(norm(F))")
end

function IBVP_solution(t::OVec64, κ::Function, f::Function,
    u₀::Function, pstore::PDEStore)
    # Set A and M to the free part of the assembled matrices
    dof = pstore.dof
    bilinear_forms_A = Dict("Omega" => (∫∫a_∇u_dot_∇v!, κ))
    bilinear_forms_M = Dict("Omega" => (∫∫c_u_v!, 1.0))
    A_free, A_fix = assemble_matrix(dof, bilinear_forms_A)
    M_free, M_fix = assemble_matrix(dof, bilinear_forms_M)
    A = A_free
    M = M_free
    # Initialize the solution matrix U
    Nₜ = lastindex(t)
    # Set the initial condition for the solution
    U_free = OMat64(zeros(dof.num_free, Nₜ+1), 1:dof.num_free, 0:Nₜ)
    uh0 = get_nodal_values(u₀, dof) 
    U_free[:,0] = uh0[1:dof.num_free]
    # Use the Crank-Nicolson method to solve the PDE
#    euler_2D!(U_free, M, A, t, get_load_vector!, pstore, f)
    crank_nicolson_2D!(U_free, M, A, t, get_load_vector!, pstore, f)
    return U_free
end

# Set up the time domain
T = 1.0
Nₜ = 20  # Number of time steps
t = collect(range(0, T, Nₜ+1))
t = OVec64(t, 0:Nₜ)
x, y, triangles = gmsh2pyplot(dof)
#Define equation coefficients
κ_const = 0.02
kx, ky = 1, 1
λ = κ_const * (kx^2 + ky^2) * π^2
u_homogeneous(x, y, t) = exp(-λ * t) * sinpi(kx * x) * sinpi(ky * y)
exact_u_homogeneous = get_nodal_values((x, y) -> u_homogeneous(x, y, T), dof)
f_homogeneous(x, y, t) = 0.0
u₀_homogeneous(x, y) = u_homogeneous(x, y, 0.0)
pstore = PDEStore((x, y) -> κ_const, f_homogeneous, dof, 
                 solver, pcg_tol, pcg_maxiterations)
#Compute the numerical solution
U_free = IBVP_solution(t, (x, y) -> κ_const, f_homogeneous, u₀_homogeneous, pstore)
U_fix = OMat64(zeros(dof.num_fixed, Nₜ+1), 1:dof.num_fixed, 0:Nₜ)
U = [U_free; U_fix]

figure(1)
plot_trisurf(x, y, triangles, U[:, Nₜ], cmap="cool")
xlabel(L"$x$")
ylabel(L"$y$")
zlabel(L"$U$")
grid(true)
title("Numerical Solution at t = $T when f ≡ 0")

figure(2)
plot_trisurf(x, y, triangles, exact_u_homogeneous, cmap="cool", alpha=0.5)
xlabel(L"$x$")
ylabel(L"$y$")
zlabel(L"$u$")
title("Exact Solution at t = $T when f ≡ 0")

u_inhomogeneous(x, y, t) = t * exp(-t) * sinpi(x) * sinpi(y)
u₀_inhomogeneous(x, y) = u_inhomogeneous(x, y, 0.0)
f_inhomogeneous(x, y, t) = (1 - t + κ_const * 2* π^2 * t) * exp(-t) *
                           sinpi(x) * sinpi(y)
exact_u_inhomogeneous = get_nodal_values((x, y) -> u_inhomogeneous(x, y, T), dof)
pstore = PDEStore((x, y) -> κ_const, f_inhomogeneous, dof, 
                   solver, pcg_tol, pcg_maxiterations)
U_free = IBVP_solution(t, (x, y) -> κ_const, f_inhomogeneous, 
                       u₀_inhomogeneous, pstore)
U = [U_free; U_fix]

figure(3)
plot_trisurf(x, y, triangles, U[:, Nₜ], cmap="cool")
xlabel(L"$x$")
ylabel(L"$y$")
zlabel(L"$U$")
grid(true)
title("Numerical Solution at t = $T when u₀ ≡ 0")

figure(4)
plot_trisurf(x, y, triangles, exact_u_inhomogeneous, cmap="cool", alpha=0.5)
xlabel(L"$x$")
ylabel(L"$y$")
zlabel(L"$u$")
title("Exact Solution at t = $T when u₀ ≡ 0")