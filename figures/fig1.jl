using SubdiffusionQMC
import SubdiffusionQMC.PDE:slove_expsum_pcg!
using SimpleFiniteElements
import SimpleFiniteElements.Utils: gmsh2pyplot
import SimpleFiniteElements.FEM: assemble_vector!
import SimpleFiniteElements.Poisson: ∫∫a_∇u_dot_∇v!, ∫∫c_u_v!, ∫∫f_v!
using JLD2
using PyPlot
using Printf

path = joinpath("..", "spatial_domains", "unit_square.geo")
gmodel = GeometryModel(path)
h = 0.02
mesh = FEMesh(gmodel, h)
essential_bcs = [("Gamma", 0.0)]
dof = DegreesOfFreedom(mesh, essential_bcs)
pcg_tol= 1e-07
pcg_maxits=100
#solver = :direct
solver = :pcg

T = 1.0
Nₜ = 150
α = 0.5
γ = 2 / α
t = graded_mesh(Nₜ, γ, T)

f_homogeneous(x, y, t) = 0.0
κ₀(x, y) = 0.1 * (2 + x * y) 
min_κ₀ = κ₀(0.0, 0.0)
pstore = PDEStore_integrand(κ₀, dof, solver, pcg_tol, pcg_maxits)
x, y, triangles = gmsh2pyplot(dof)

u₀_bent(x, y) = 144 * (x^2 * (1 - x) * y^2 * (1 - y))
uh0 = get_nodal_values(u₀_bent, dof)

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

qmc_path = joinpath("..", "qmc_points", "SPOD_dim256.jld2")
Nvals, pts = SPOD_points(z, qmc_path)

function generate_τ_values(lo, hi)
    τ_values = [10.0^k for k in lo : hi]
    return τ_values
end

function get_load_vector!(F::Vec64, t::Float64, pstore::PDEStore_integrand, f::Function)
    fill!(F, 0.0)
    assemble_vector!(F, "Omega", [(∫∫f_v!, (x, y) -> f(x, y, t))], dof, 1)
end


y_vals = view(pts[9], :, 2)
(N₁, N₂) = resolution
x₁_vals = range(0, 1, N₁)
x₂_vals = range(0, 1, N₂)
κ₀_vals = Float64[ κ₀(x, y) for x in x₁_vals, y in x₂_vals ]


κ_ = interpolate_κ!(y_vals, κ₀_vals, dstore)
bilinear_forms_A = Dict("Omega" => (∫∫a_∇u_dot_∇v!, (x, y) -> κ_(x, y)))
(; κ₀, dof, P, wkspace, pcg_tol, bilinear_forms_M) = pstore
(;α, r, t, ω, S, a, w) = estore
Nₜ = lastindex(t)
A_free, _ = assemble_matrix(dof, bilinear_forms_A)
M_free, _ = assemble_matrix(dof, bilinear_forms_M)
num_free, num_fixed = dof.num_free, dof.num_fixed
u_free = OMat64(zeros(num_free, Nₜ+1), 1:num_free, 0:Nₜ)
u_fix = OMat64(zeros(num_fixed, Nₜ+1), 1:num_fixed, 0:Nₜ)
u0h = get_nodal_values(u₀_bent, dof)
u_free[:,0] = u0h[1:dof.num_free]
lo = floor(Int, log10(t[1]-t[0]))
hi = ceil(Int, log10(t[end] - t[end-1]))
τ_values = generate_τ_values(lo, hi)
num_its = Vec64(zeros(Nₜ))
if solver == :direct
    generalised_crank_nicolson_2D!(u_free, M_free, A_free, 
                                             get_load_vector!, estore, pstore, f_homogeneous)
    else solver == :pcg
    num_its = slove_expsum_pcg!(u_free, M_free, A_free, get_load_vector!,
                                 estore, pstore, f_homogeneous, τ_values)
end
uh = [u_free[:,Nₜ]; u_fix[:,Nₜ]]

figure(1)
tricontourf(x, y, u0h, triangles=triangles)
colorbar()
xlabel(L"$x_1$")
ylabel(L"$x_2$")
axis("equal")
savefig("initial_time.pdf")

figure(2)
tricontourf(x, y, uh, triangles=triangles)
colorbar()
xlabel(L"$x_1$")
ylabel(L"$x_2$")
axis("equal")
savefig("final_time.pdf")
