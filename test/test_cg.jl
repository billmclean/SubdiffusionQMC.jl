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
h = 0.2
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
t = collect(range(0, T, Nₜ+1))
t = OVec64(t, 0:Nₜ)
U_free = OMat64(zeros(dof.num_free, Nₜ+1), 1:dof.num_free, 0:Nₜ)
Nₕ = lastindex(U_free, 1)
F = Vec64(undef, Nₕ)
rhs = similar(F)
wkspace = zeros(Nₕ, 4)
U_free[:,0] = uh0[1:dof.num_free]
ΔU = zeros(dof.num_free)
residual_norm = zeros(Float64, Nₜ)
error_norm_max = zeros(Float64, Nₜ)
for n = 1:Nₜ
  global U_free,  ΔU, τ
  τ = t[n] - t[n-1]
  B = M + (τ/2) * A
  midpoint = (t[n] + t[n-1]) / 2
  get_load_vector!(F, midpoint, f_homogeneous, dof)
  mul!(rhs, A, U_free[1:Nₕ,n-1])
  scal!(-τ, rhs) # rhs = τ F - τ A Uⁿ⁻¹
  fill!(ΔU, 0.0)
  start = time()
  num_its = SubdiffusionQMC.cg!(ΔU, B, rhs, tol, wkspace)
  elapsed = time() - start
  #num_its = 0
  #ΔU = B \ rhs
  U_free[:,n] .= ΔU + U_free[:,n-1]
  println("num_its=$num_its")
  println("rate for cg!:", elapsed, "seconds")
  #calculate residual norm
  residual_norm[n] = max(residual_norm[n], norm(rhs - B * ΔU)/norm(rhs))
  #calculate error norm at nodal values
  u = get_nodal_values((x, y) -> u_homogeneous(x, y, t[n]), dof)
  max_n = maximum(abs, U_free[:,n] - u[1:dof.num_free])
  error_norm_max[n] = max(error_norm_max[n], max_n)
end

figure(1)
title("Residuals")
semilogy(t[1:Nₜ], residual_norm, label="Residual Norm")
xlabel("t")
grid(true)

figure(2)
plot(t[1:Nₜ], error_norm_max, label="Error Maximum Norm")
xlabel("t")
grid(true)