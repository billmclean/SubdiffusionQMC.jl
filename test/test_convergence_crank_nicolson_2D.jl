using SimpleFiniteElements
import SimpleFiniteElements.Utils: gmsh2pyplot
import SimpleFiniteElements.FEM: assemble_vector!
using SubdiffusionQMC
import OffsetArrays: OffsetArray
using PyPlot
using Printf
import SimpleFiniteElements.Poisson: ∫∫a_∇u_dot_∇v!, ∫∫c_u_v!, ∫∫f_v!
using LinearAlgebra

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
    T = t[Nₜ]
    crank_nicolson_2D!(U_free, M, A, T, Nₜ, get_load_vector!, pstore, f)
    return U_free
end

# Set up the time domain
T = 1.0
#Define equation coefficients
κ_const = 0.02
k₁, k₂ = 1, 1     
λ = κ_const * (k₁^2 + k₂^2) * π^2
u_homogeneous(x, y, t) = exp(-λ * t) * sinpi(k₁ * x) * sinpi(k₂ * y)
f_homogeneous(x, y, t) = 0.0
u₀_homogeneous(x, y) = u_homogeneous(x, y, 0.0)

function compute_error_time(nrows::Int, hmax::Float64, Nₜ::Int)
    mesh = FEMesh(gmodel, hmax)
    dof = DegreesOfFreedom(mesh, essential_bcs)
    pstore = PDEStore((x, y) -> κ_const, f_homogeneous, dof, 
                solver, pcg_tol, pcg_maxiterations)
    max_error = zeros(nrows)
    @printf("\n%6s  %6s  %6s %10s  %8s  %8s\n\n", 
        "Nₜ", "hmax", "τ", "Error", "rate", "seconds")
    for k = 1 : nrows
        local uh_free, t
        Nₜ *= 2
        t = collect(range(0, T, Nₜ+1))
        t = OVec64(t, 0:Nₜ)
        start = time()
        uh_free = IBVP_solution(t, (x, y) -> κ_const, f_homogeneous, u₀_homogeneous, pstore)
        for i = 0 : lastindex(t)
            u = get_nodal_values((x, y) -> u_homogeneous(x, y, t[i]), dof)
            max_i = maximum(abs, uh_free[:,i] - u[1:dof.num_free])
            max_error[k] = max(max_error[k], max_i)
        end
        elapsed = time() - start
        τ = T / Nₜ
        if k == 1
            @printf("%6d  %6f %6f  %10.2e  %8s  %8.3f\n", 
                Nₜ, hmax, τ, max_error[k], "", elapsed)
        else
            rate = log2(max_error[k-1]/max_error[k])
            @printf("%6d  %6f %6f  %10.2e  %8.3f  %8.3f\n", 
                Nₜ, hmax, τ, max_error[k], rate, elapsed)
        end
    end
end

function compute_error_space(nrows::Int, hmax::Float64, Nₜ::Int)
    t = collect(range(0, T, Nₜ+1))
    t = OVec64(t, 0:Nₜ)
    max_error = zeros(nrows)
    @printf("\n%6s  %6s  %6s %10s  %8s  %8s\n\n", 
        "Nₜ", "hmax", "τ", "Error", "rate", "seconds")
    for k = 1 : nrows
        local uh_free, mesh, dof, pstore
        hmax /= 2
        mesh = FEMesh(gmodel, hmax)
        dof = DegreesOfFreedom(mesh, essential_bcs)
        pstore = PDEStore((x, y) -> κ_const, f_homogeneous, dof, 
                    solver, pcg_tol, pcg_maxiterations)
        start = time()
        uh_free = IBVP_solution(t, (x, y) -> κ_const, f_homogeneous, u₀_homogeneous, pstore)
        for i = 0 : lastindex(t)
            u = get_nodal_values((x, y) -> u_homogeneous(x, y, t[i]), dof)
            max_i = maximum(abs, uh_free[:,i] - u[1:dof.num_free])
            max_error[k] = max(max_error[k], max_i)
        end
        elapsed = time() - start
        τ = T / Nₜ
        if k == 1
            @printf("%6d  %6f %6f  %10.2e  %8s  %8.3f\n", 
                Nₜ, hmax, τ, max_error[k], "", elapsed)
        else
            rate = log2(max_error[k-1]/max_error[k])
            @printf("%6d  %6f %6f  %10.2e  %8.3f  %8.3f\n", 
                Nₜ, hmax, τ, max_error[k], rate, elapsed)
        end
    end
end

# Usage_space
#nrows = 4, hmax = 1/4, Nₜ = 80
#compute_error_space(nrows, hmax, Nₜ)
compute_error_space(5, 1/2, 32)

# Usage_time
#nrows = 4, hmax = 1/1200, Nₜ = 2
#compute_error_time(nrows, hmax, Nₜ)
compute_error_time(4, 1/1200, 2)