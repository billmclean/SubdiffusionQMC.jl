include("average_profile.jl")
tol = 1e-7
h = 0.1
mesh = FEMesh(gmodel, h)
dof = DegreesOfFreedom(mesh, essential_bcs)

pcg_tol= 1e-07
pcg_maxits=100
solver = :pcg

T = 1.0
f_homogeneous(x, y, t) = 0.0
κ₀(x, y) = 0.1 * (2 + x * y) 
min_κ₀ = κ₀(0.0, 0.0)

pstore = PDEStore_integrand(κ₀, dof, solver, pcg_tol, pcg_maxits)
x, y, triangles = gmsh2pyplot(dof)
bilinear_forms_A = Dict("Omega" => (∫∫a_∇u_dot_∇v!, κ₀))
bilinear_forms_M = Dict("Omega" => (∫∫c_u_v!, 1.0))
A, _ = assemble_matrix(dof, bilinear_forms_A)
M_free, _ = assemble_matrix(dof, bilinear_forms_M)

function get_load_vector!(F::Vec64, t::Float64, pstore::PDEStore_integrand, f::Function)
    fill!(F, 0.0)
    assemble_vector!(F, "Omega", [(∫∫f_v!, (x, y) -> f(x, y, t))], dof, 1)
end

#dstore
p = 0.5
resolution = (256, 256)
n = 15
idx = double_indices(n)
z = lastindex(idx)
dstore = DiffusivityStore2D(idx, z, p, resolution, min_κ₀)

#u₀
u₀_bent(x, y) = 5 * (x^2 * (1 - x) + y^2 * (1 - y))
uh0 = get_nodal_values(u₀_bent, dof) 

Nₜ = 50 
α = 0.5
γ = 2 / α
t = graded_mesh(Nₜ, γ, T)


Nₛ = dof.num_free + dof.num_fixed
Nₕ = dof.num_free
F_det = Vec64(undef, Nₕ)
rhs_det = similar(F_det)

L = zeros(Float64, 1)
U_free = OffsetArray(zeros(dof.num_free, Nₜ+1), 1:dof.num_free, 0:Nₜ)
U_fix = OffsetArray(zeros(dof.num_fixed, Nₜ+1), 1:dof.num_fixed, 0:Nₜ)

num_its_2 = Vec64(zeros(Nₜ))

function generate_τ_values(lo, hi)
    τ_values = [10.0^k for k in lo : hi]
    return τ_values
end
lo = floor(Int, log10(t[1]-t[0]))
hi = ceil(Int, log10(t[end] - t[end-1]))
τ_values = generate_τ_values(lo, hi)

start = time()
    y_vals = rand(z) .- 1/2
    (N₁, N₂) = resolution
    x₁_vals = range(0, 1, N₁)
    x₂_vals = range(0, 1, N₂)
    κ₀_vals = Float64[ κ₀(x, y) for x in x₁_vals, y in x₂_vals ]
    κ_ = interpolate_κ!(y_vals, κ₀_vals, dstore)
    bilinear_forms_A = Dict("Omega" => (∫∫a_∇u_dot_∇v!, (x,y) -> κ_(x, y)))
    A, _ = assemble_matrix(dof, bilinear_forms_A)
    U_free[:,0] = uh0[1:dof.num_free]
    #pcg method
        num_its_2 = solve_pcg!(U_free, M_free, A, get_load_vector!,
                                  α, t, pstore, f_homogeneous, τ_values)
    uh = [ U_free[:, Nₜ]; U_fix[:, Nₜ] ]
    L,_ = average_field(uh, "Omega", dof)
elapsed = time() - start
println("Use $elapsed secs")
average_num_its_2 = mean(num_its_2, dims=2)
figure(1)
plot(t[1:Nₜ],average_num_its_2)
title("Average iterations of pcg for time step")