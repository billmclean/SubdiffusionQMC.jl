using SimpleFiniteElements
import SimpleFiniteElements.Utils: gmsh2pyplot
import SimpleFiniteElements.FEM: assemble_vector!, average_field
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

T = 1.0
f_homogeneous(x, y, t) = 0.0
κ₀(x, y) = 0.1 * (2 + x * y) 
min_κ₀ = κ₀(0.0, 0.0)
x, y, triangles = gmsh2pyplot(dof)
pstore = PDEStore(κ₀, f_homogeneous, dof, 
                  solver, pcg_tol, pcg_maxiterations)
p = 0.5
resolution = (256, 256)
n = 15
idx = double_indices(n)
z = lastindex(idx)
dstore = DiffusivityStore2D(idx, z, p, resolution, min_κ₀)
u₀_bent(x, y) = 5 * (x^2 * (1 - x) + y^2 * (1 - y))  # Example initial condition in 2D 
Nₜ = 50
t = collect(range(0, T, Nₜ+1))
t = OVec64(t, 0:Nₜ)
U_det = IBVP_solution(t, κ₀, f_homogeneous, u₀_bent, pstore)
U_det_fix = OMat64(zeros(dof.num_fixed, Nₜ+1), 1:dof.num_fixed, 0:Nₜ)
Nₛ = dof.num_free + dof.num_fixed
uh = [U_det[:,Nₜ]; U_det_fix[:,Nₜ]]
L₀, _ = average_field(uh, "Omega", dof)
M = 100
L = zeros(Float64, M+1)
L[1] = L₀
U_free = OffsetArray(zeros(dof.num_free, Nₜ+1, M), 1:dof.num_free, 0:Nₜ, 1:M)
U_fix = OffsetArray(zeros(dof.num_fixed, Nₜ+1, M), 1:dof.num_fixed, 0:Nₜ, 1:M)
Nₛ = dof.num_free + dof.num_fixed
U = OffsetArray(zeros(Nₛ, Nₜ+1, M), 1:Nₛ, 0:Nₜ, 1:M)
for m = 1:M
    local y_vals, pstore, uh
    global U
    y_vals = rand(z) .- 1/2
    (N₁, N₂) = resolution
    x₁_vals = range(0, 1, N₁)
    x₂_vals = range(0, 1, N₂)
    κ₀_vals = Float64[ κ₀(x, y) for x in x₁_vals, y in x₂_vals ]
    κ_ = interpolate_κ!(y_vals, κ₀_vals, dstore)
    pstore = PDEStore((x, y) -> κ_(x, y), f_homogeneous, dof, 
                  solver, pcg_tol, pcg_maxiterations)
    U_free[:,:,m] = IBVP_solution(t, (x, y) -> κ_(x, y), f_homogeneous, u₀_bent, pstore)    
    uh = [ U_free[:, Nₜ, m]; U_fix[:, Nₜ, m] ]
    L[m+1],_ = average_field(uh, "Omega", dof)
end

figure(1)
bar(1:M+1, L)
title("Histogram of all different values")
xlabel("M")
ylabel("L[m]")
grid(true)
show()