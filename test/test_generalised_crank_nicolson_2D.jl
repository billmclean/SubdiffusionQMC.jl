using SimpleFiniteElements
import SimpleFiniteElements.Utils: gmsh2pyplot
using SubdiffusionQMC
import OffsetArrays: OffsetArray
using PyPlot
using Printf
import SimpleFiniteElements.Poisson: ∫∫a_∇u_dot_∇v!, ∫∫c_u_v!, ∫∫f_v!

fast_method = false
if fast_method
    @printf("Using exponential sum approximation.\n")
else
    @printf("Using direct evaluation of the memory term.\n")
end

#E_half(x) = erfcx(-x)

# Define the solver and path to the geometry file
#solver = :pcg
solver = :direct
path = joinpath("..", "spatial_domains", "unit_square.geo")

# Create the geometry model and specify essential boundary conditions
gmodel = GeometryModel(path)
essential_bcs = [("Gamma", 0.0)]
    
# Specify the mesh size and create the finite element mesh
h = 0.2
mesh = FEMesh(gmodel, h)
    
# Determine the degrees of freedom
dof = DegreesOfFreedom(mesh, essential_bcs)
    
# Create a PDEStore object to store information about the PDE
pcg_tol, pcg_maxiterations = 1e-8, 100 #not used

function get_load_vector!(F::Vec64, t::Float64, pstore::PDEStore, f::Function)
    linear_functionals = Dict("Omega" => (∫∫f_v!, (x, y) -> f(x, y, t)))
    F, u_fix = assemble_vector(pstore.dof, linear_functionals)
end

function IBVP_solution(t::OVec64, κ::Function, f::Function,
    u₀::Function, pstore::PDEStore)
    # Set A and M to the free part of the assembled matrices
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
    #generalised_crank_nicolson_2D!(U_free, M, A, t, α, get_load_vector!, f, pstore)
    crank_nicolson_2D!(U_free, M, A, t, get_load_vector!, pstore, f)
    return U_free
end

# Set up the time domain
T = 1.0
ε = 0.1
Nₜ = 40  # Number of time steps
t = graded_mesh(Nₜ, 1.0, T) #uniform
x, y, triangles = gmsh2pyplot(dof)

#Define equation coefficients
α = 0.5
κ_const = 0.02
kx = 1
ky = 2
λ = κ_const * ( kx^2 + ky^2 ) * π^2
u_homogeneous(x, y, t) = exp(-λ * t) * sinpi(kx * x) * sinpi(ky * y)
exact_u_homogeneous = get_nodal_values((x, y) -> u_homogeneous(x, y, T), dof)
f_homogeneous(x, y, t) = 0.0
u₀_homogeneous(x, y) = u_homogeneous(x, y, 0.0)
pstore = PDEStore((x, y) -> κ_const, f_homogeneous, dof, 
                 solver, pcg_tol, pcg_maxiterations)
#Solve the numerical solution
U_free = IBVP_solution(t, α, (x, y) -> κ_const, f_homogeneous, u₀_homogeneous, pstore)
U_fix = OMat64(zeros(dof.num_fixed, Nₜ+1), 1:dof.num_fixed, 0:Nₜ)
U = [U_free; U_fix]

Nₜ = 40
h = 0.2
nrows = 5
max_error = zeros(nrows)
@printf("\n%6s  %6s  %6s  %10s  %8s  %8s\n\n", 
	"Nₜ", "Nₕ", "h", "Error", "rate", "seconds")
for row = 1:nrows
    local U, U_free, U_fix, t, x, y, triangles, pstore, exact_u_homogeneous
    global h, Nₜ
    Nₜ *= 2
    h /= 2
    x, y, triangles = gmsh2pyplot(dof)
    t = graded_mesh(Nₜ, 1.0, T) #uniform
    start = time()
    pstore = PDEStore((x, y) -> κ_const, f_homogeneous, dof, 
                 solver, pcg_tol, pcg_maxiterations)
    U_free = IBVP_solution(t, (x, y) -> κ_const, f_homogeneous, u₀_homogeneous, pstore)
    U_fix = OMat64(zeros(dof.num_fixed, Nₜ+1), 1:dof.num_fixed, 0:Nₜ)
    U = [U_free; U_fix]
    exact_u_homogeneous = get_nodal_values((x, y) -> u_homogeneous(x, y, T), dof)
    elapsed = time() - start
    max_error[row] = maximum(abs, U[:,Nₜ] - exact_u_homogeneous)
    if row == 1
	@printf("%6d  %6d  %6.2f  %10.2e  %8s  %8.3f\n", 
		Nₜ, dof.num_free, h, max_error[row], "", elapsed)
    else
	rate = log2(max_error[row-1]/max_error[row])
	@printf("%6d  %6d  %6d  %10.2e  %8.3f  %8.3f\n", 
		Nₜ, dof.num_free, h, max_error[row], rate, elapsed)
    end
end