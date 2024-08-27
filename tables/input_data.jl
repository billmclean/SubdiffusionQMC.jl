using SimpleFiniteElements
using SubdiffusionQMC
import SimpleFiniteElements.FEM: assemble_vector!
import SimpleFiniteElements.Poisson: ∫∫f_v!
using Printf

# Finite element mesh
path = joinpath("..", "spatial_domains", "unit_square.geo")
gmodel = GeometryModel(path)
hmax = 0.02
mesh = FEMesh(gmodel, hmax; save_msh_file=false)
essential_bcs = [("Gamma", 0.0)]
dof = DegreesOfFreedom(mesh, essential_bcs)

h_finest = max_elt_diameter(mesh)
h_string = @sprintf("%0.4f", h_finest)

# Time stepping grid
T = 1.0
Nₜ = 150
α = 0.5
γ = 2 / α
t = graded_mesh(Nₜ, γ, T)

# PDE
f_homogeneous(x, y, t) = 0.0
κ₀(x, y) = 0.1 * (2 + x * y) 
u₀_bent(x, y) = 144 * (x^2 * (1 - x) * y^2 * (1 - y))

function get_load_vector!(F::Vec64, t::Float64, pstore::PDEStore_integrand, 
                          f::Function)
    fill!(F, 0.0)
    assemble_vector!(F, "Omega", [(∫∫f_v!, (x, y) -> f(x, y, t))], dof, 1)
end

solver = :pcg  # :direct or :pcg
pcg_tol = 1e-10
pcg_maxits = 100
pstore = PDEStore_integrand(κ₀, dof, solver, pcg_tol, pcg_maxits)

# Random coefficient
p = 0.5
resolution = (256, 256)
q = 22
idx = double_indices(q)
z = lastindex(idx)
min_κ₀ = κ₀(0.0, 0.0)
dstore = DiffusivityStore2D(idx, z, p, resolution, min_κ₀)

# QMC points
qmc_path = joinpath("..", "qmc_points", "SPOD_dim256.jld2")
Nvals, pts = SPOD_points(z, qmc_path)

# FFT grid
(N₁, N₂) = resolution
x₁_vals = range(0, 1, N₁)
x₂_vals = range(0, 1, N₂)
κ₀_vals = Float64[ κ₀(x, y) for x in x₁_vals, y in x₂_vals ]

# Exponential sum approximation 
r = 2
tol = 1e-8
Δx = 0.5
Nₕ = dof.num_free
estore = ExponentialSumStore(t, Nₕ, α, r, tol, Δx)

# Descriptive message
msg = """
Solving subdiffusion equation with random diffusivity.
Generalised Crank-Nicolson time stepping and finite elements.
Fractional time derivative of order α = $α.
Solver is $solver with tol = $pcg_tol.
Employing $(Threads.nthreads()) threads.
SPOD QMC points with z = $z.
Finest FEM mesh has $(dof.num_free) degrees of freedom and h = $h_string.
Using Nₜ = $Nₜ time steps with mesh grading parameter γ = $γ.
Using $N₁ x $N₂ grid to interpolate κ."""

println(msg)

