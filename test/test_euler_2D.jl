using SimpleFiniteElements
import SimpleFiniteElements.Utils: gmsh2pyplot
using SubdiffusionQMC
import OffsetArrays: OffsetArray
using PyPlot
using Printf
import SimpleFiniteElements.Poisson: ∫∫a_∇u_dot_∇v!, ∫∫c_u_v!, ∫∫f_v!
using LinearAlgebra
import SubdiffusionQMC.Timestepping: euler_2D!

path = joinpath("..", "spatial_domains", "unit_square.geo")
gmodel = GeometryModel(path)
essential_bcs = [("Gamma", 0.0)]
    
h = 0.4
mesh = FEMesh(gmodel, h)
dof = DegreesOfFreedom(mesh, essential_bcs)

T = 1.0
Nₜ = 20  # Number of time steps
t = collect(range(0, T, Nₜ+1))
t = OVec64(t, 0:Nₜ)
    
κ_const = 0.02
kx, ky = 1, 1
λ = κ_const * (kx^2 + ky^2) * π^2
u_homogeneous(x, y, t) = exp(-λ * t) * sinpi(kx * x) * sinpi(ky * y)
exact_u_homogeneous = get_nodal_values((x, y) -> u_homogeneous(x, y, T), dof)
f_homogeneous(x, y, t) = 0.0
u₀_homogeneous(x, y) = u_homogeneous(x, y, 0.0)

function get_load_vector!(F::Vec64, t::Float64, f::Function)
    linear_functionals = Dict("Omega" => (∫∫f_v!, (x, y) -> f(x, y, t)))
    F[:], u_fix = assemble_vector(dof, linear_functionals)
    println("t = $t, ΣF = $(sum(F))")
end

bilinear_forms_A = Dict("Omega" => (∫∫a_∇u_dot_∇v!, (x, y) -> κ_const))
bilinear_forms_M = Dict("Omega" => (∫∫c_u_v!, 1.0))
A_free, A_fix = assemble_matrix(dof, bilinear_forms_A)
M_free, M_fix = assemble_matrix(dof, bilinear_forms_M)
A = A_free
M = M_free

U_free = OMat64(zeros(dof.num_free, Nₜ+1), 1:dof.num_free, 0:Nₜ)
uh0 = get_nodal_values(u₀_homogeneous, dof) 
U_free[:,0] = uh0[1:dof.num_free]
euler_2D!(U_free, M, A, t, get_load_vector!, f_homogeneous)

x, y, triangles = gmsh2pyplot(dof)

uh_homogeneous = [ U_free[:,Nₜ]; zeros(dof.num_fixed) ]

figure(1)
plot_trisurf(x, y, triangles, uh_homogeneous, cmap="cool")
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

uh0 = get_nodal_values(u₀_inhomogeneous, dof) 
U_free[:,0] = uh0[1:dof.num_free]
euler_2D!(U_free, M, A, t, get_load_vector!, f_inhomogeneous)
uh_inhomogeneous = [ U_free[:,Nₜ]; zeros(dof.num_fixed) ]

figure(3)
plot_trisurf(x, y, triangles, uh_inhomogeneous, cmap="cool")
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
