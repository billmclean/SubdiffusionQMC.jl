using SimpleFiniteElements
import SimpleFiniteElements.Utils: gmsh2pyplot
import SimpleFiniteElements.FEM: assemble_vector!
using SubdiffusionQMC
import OffsetArrays: OffsetArray
using PyPlot
using Printf
import SimpleFiniteElements.Poisson: ∫∫a_∇u_dot_∇v!, ∫∫c_u_v!, ∫∫f_v!
using LinearAlgebra
import SpecialFunctions: erfcx
solver = :direct
path = joinpath("..", "spatial_domains", "unit_square.geo")
gmodel = GeometryModel(path)
essential_bcs = [("Gamma", 0.0)]
h = 0.1
mesh = FEMesh(gmodel, h)
dof = DegreesOfFreedom(mesh, essential_bcs)
pcg_tol, pcg_maxiterations = 1e-8, 100 #not used

E_half(x) = erfcx(-x)

fast_method = false

if fast_method
    @printf("Using exponential sum approximation.\n")
else
    @printf("Using direct evaluation of the memory term.\n")
end

function get_load_vector!(F::Vec64, t::Float64, pstore::PDEStore, f::Function)
    fill!(F, 0.0)
    assemble_vector!(F, "Omega", [(∫∫f_v!, (x, y) -> f(x, y, t))], dof, 1)
end

function IBVP_solution(t::OVec64, α::Float64, κ::Function, f::Function,
    u₀::Function, pstore::PDEStore)
    dof = pstore.dof
    bilinear_forms_A = Dict("Omega" => (∫∫a_∇u_dot_∇v!, κ))
    bilinear_forms_M = Dict("Omega" => (∫∫c_u_v!, 1.0))
    A_free, A_fix = assemble_matrix(dof, bilinear_forms_A)
    M_free, M_fix = assemble_matrix(dof, bilinear_forms_M)
    A = A_free
    M = M_free
    Nₜ = lastindex(t)
    U_free = OMat64(zeros(dof.num_free, Nₜ+1), 1:dof.num_free, 0:Nₜ)
    uh0 = get_nodal_values(u₀, dof) 
    U_free[:,0] = uh0[1:dof.num_free]
    generalised_crank_nicolson_2D!(U_free, M, A, α, t, get_load_vector!, pstore, f)
    return U_free
end

function IBVP_solution(κ::Function, f::Function,
    u₀::Function, pstore::PDEStore, estore::ExponentialSumStore)
    (;t, ω) = estore
    dof = pstore.dof
    bilinear_forms_A = Dict("Omega" => (∫∫a_∇u_dot_∇v!, κ))
    bilinear_forms_M = Dict("Omega" => (∫∫c_u_v!, 1.0))
    A_free, A_fix = assemble_matrix(dof, bilinear_forms_A)
    M_free, M_fix = assemble_matrix(dof, bilinear_forms_M)
    A = A_free
    M = M_free
    Nₜ = lastindex(t)
    U_free = OMat64(zeros(dof.num_free, Nₜ+1), 1:dof.num_free, 0:Nₜ)
    Nₕ = lastindex(U_free,1)
    uh0 = get_nodal_values(u₀, dof) 
    U_free[:,0] = uh0[1:dof.num_free]
    generalised_crank_nicolson_2D!(U_free, M, A, get_load_vector!, estore, pstore, f)
    return U_free
end