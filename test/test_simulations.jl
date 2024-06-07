using SubdiffusionQMC
using SimpleFiniteElements
import SimpleFiniteElements.Utils: gmsh2pyplot
import SimpleFiniteElements.FEM: assemble_vector!
import SimpleFiniteElements.Poisson: ∫∫a_∇u_dot_∇v!, ∫∫c_u_v!, ∫∫f_v!
using JLD2

path = joinpath("..", "spatial_domains", "unit_square.geo")
gmodel = GeometryModel(path)
h = 0.2
mesh = FEMesh(gmodel, h)
essential_bcs = [("Gamma", 0.0)]
dof = DegreesOfFreedom(mesh, essential_bcs)
pcg_tol= 1e-07
pcg_maxits=100
#solver = :direct
solver = :pcg

T = 1.0
Nₜ = 20
α = 0.5
γ = 2 / α
t = graded_mesh(Nₜ, γ, T)

f_homogeneous(x, y, t) = 0.0
κ₀(x, y) = 0.1 * (2 + x * y) 
min_κ₀ = κ₀(0.0, 0.0)
pstore = PDEStore_integrand(κ₀, dof, solver, pcg_tol, pcg_maxits)
x, y, triangles = gmsh2pyplot(dof)

u₀_bent(x, y) = 5 * (x^2 * (1 - x) + y^2 * (1 - y))

p = 0.5
resolution = (256, 256)
n = 15
idx = double_indices(n)
z = lastindex(idx)
dstore = DiffusivityStore2D(idx, z, p, resolution, min_κ₀)

r = 2
tol = 1e-8
Δx = 0.5
Nₕ = dof.num_free
estore = ExponentialSumStore(t, Nₕ, α, r, tol, Δx)

function get_load_vector!(F::Vec64, t::Float64, pstore::PDEStore_integrand, f::Function)
    fill!(F, 0.0)
    assemble_vector!(F, "Omega", [(∫∫f_v!, (x, y) -> f(x, y, t))], dof, 1)
end

(N₁, N₂) = resolution
x₁_vals = range(0, 1, N₁)
x₂_vals = range(0, 1, N₂)
κ₀_vals = Float64[ κ₀(x, y) for x in x₁_vals, y in x₂_vals ]

qmc_path = joinpath("..", "qmc_points", "SPOD_dim256.jld2")
Nvals, pts = SPOD_points(z, qmc_path)

use_fft = true

if use_fft
    ϕ, ϕ_det, pcg_its = simulations!(pts[9], solver, κ₀_vals, f_homogeneous, 
                         get_load_vector!, pstore, estore, dstore, u₀_bent)
else
    ϕ, ϕ_det, pcg_its = slow_simulations!(pts[9], solver, κ₀, f_homogeneous, 
                         get_load_vector!, pstore, estore, dstore, u₀_bent)
end