using SimpleFiniteElements
import SimpleFiniteElements.Utils: gmsh2pyplot
import SimpleFiniteElements.FEM: assemble_vector!
import SimpleFiniteElements.Poisson: ∫∫a_∇u_dot_∇v!, ∫∫c_u_v!, ∫∫f_v!
using SubdiffusionQMC
import OffsetArrays: OffsetArray
using PyPlot
using Printf
using LinearAlgebra
import LinearAlgebra: mul!, ldiv!, cholesky!, axpby!, cholesky
import LinearAlgebra.BLAS: scal!

#tol
tol = 1e-7
#exact solution and functions
κ₀ = 0.02
k₁, k₂ = 1, 1     
λ = κ₀ * (k₁^2 + k₂^2) * π^2
u_homogeneous(x, y, t) = exp(-λ * t) * sinpi(k₁ * x) * sinpi(k₂ * y)
u₀_homogeneous(x, y) = u_homogeneous(x, y, 0.0)
f_homogeneous(x, y, t) = 0.0
#space
path = joinpath("..", "spatial_domains", "unit_square.geo")
gmodel = GeometryModel(path)
essential_bcs = [("Gamma", 0.0)]
bilinear_forms_A = Dict("Omega" => (∫∫a_∇u_dot_∇v!, κ₀))
bilinear_forms_M = Dict("Omega" => (∫∫c_u_v!, 1.0))
#mesh
h = 0.02
mesh = FEMesh(gmodel, h)
dof = DegreesOfFreedom(mesh, essential_bcs)
#matrices
A, _ = assemble_matrix(dof, bilinear_forms_A)
M, _ = assemble_matrix(dof, bilinear_forms_M)

function get_load_vector!(F::Vec64, t::Float64, f::Function, dof)
    fill!(F, 0.0)
    assemble_vector!(F, "Omega", [(∫∫f_v!, (x, y) -> f(x, y, t))], dof, 1)
end

uh0 = get_nodal_values(u₀_homogeneous, dof)
T = 1.0
Nₜ = 50
U_free = OMat64(zeros(dof.num_free, Nₜ+1), 1:dof.num_free, 0:Nₜ)
Nₕ = lastindex(U_free, 1)
F = Vec64(undef, Nₕ)
rhs = similar(F)
wkspace = zeros(Nₕ, 4)
U_free[:,0] = uh0[1:dof.num_free]
ΔU = zeros(dof.num_free)

#direct passing PDEStore
solver = :direct
pcg_tol, pcg_maxiterations = 1e-8, 100 #not used
pstore = PDEStore((x, y) -> κ₀, f_homogeneous, dof, 
                solver, pcg_tol, pcg_maxiterations)
#direct solving

function get_load_vector_!(F::Vec64, t::Float64, pstore::PDEStore, f::Function)
    fill!(F, 0.0)
    assemble_vector!(F, "Omega", [(∫∫f_v!, (x, y) -> f(x, y, t))], dof, 1)
#    println("t = $t, ||F|| = $(norm(F))")
end


function IBVP_solution(t::OVec64, κ::Function, f::Function,
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
    crank_nicolson_2D!(U_free, M, A, t, get_load_vector_!, pstore, f)
    return U_free
end