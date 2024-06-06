include("average_profile.jl")
tol = 1e-7
h = 0.1
mesh = FEMesh(gmodel, h)
dof = DegreesOfFreedom(mesh, essential_bcs)

function get_load_vector!(F::Vec64, t::Float64, f::Function, dof)
    fill!(F, 0.0)
    assemble_vector!(F, "Omega", [(∫∫f_v!, (x, y) -> f(x, y, t))], dof, 1)
end

T = 1.0
f_homogeneous(x, y, t) = 0.0
κ₀(x, y) = 0.1 * (2 + x * y) 
min_κ₀ = κ₀(0.0, 0.0)
x, y, triangles = gmsh2pyplot(dof)
bilinear_forms_A = Dict("Omega" => (∫∫a_∇u_dot_∇v!, κ₀))
bilinear_forms_M = Dict("Omega" => (∫∫c_u_v!, 1.0))
A, _ = assemble_matrix(dof, bilinear_forms_A)
M_free, _ = assemble_matrix(dof, bilinear_forms_M)

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
wkspace = zeros(Nₕ, 4)

M = 200
L = zeros(Float64, M)
U_free = OffsetArray(zeros(dof.num_free, Nₜ+1, M), 1:dof.num_free, 0:Nₜ, 1:M)
U_fix = OffsetArray(zeros(dof.num_fixed, Nₜ+1, M), 1:dof.num_fixed, 0:Nₜ, 1:M)

F = Vec64(undef, Nₕ)
rhs = similar(F)
Σ = similar(F)
num_its = Mat64(zeros(Nₜ,M))

τ_values = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
P = Vector{SparseCholeskyFactor}(undef, length(τ_values))
for k in eachindex(τ_values)
    P[k] = cholesky(M_free + (τ_values[k]/2) * A)
end
ω = weights(α, t)

start = time()
for m = 1:M
    global num_its
    local y_vals, uh, bilinear_forms_A, A
    y_vals = rand(z) .- 1/2
    (N₁, N₂) = resolution
    x₁_vals = range(0, 1, N₁)
    x₂_vals = range(0, 1, N₂)
    κ₀_vals = Float64[ κ₀(x, y) for x in x₁_vals, y in x₂_vals ]
    κ_ = interpolate_κ!(y_vals, κ₀_vals, dstore)
    bilinear_forms_A = Dict("Omega" => (∫∫a_∇u_dot_∇v!, (x,y) -> κ_(x, y)))
    A, _ = assemble_matrix(dof, bilinear_forms_A)
    U_free[:,0,m] = uh0[1:dof.num_free]
    #pcg method
    for n = 1:Nₜ
        global U_free, ΔU, τ, num_its, B, τ_pre, B_pre, P
        local v
        τ = t[n] - t[n-1]
        index_τ = argmin(abs.(τ_values .- τ))
        B = ω[n][n] * M_free + (τ/2) * A
        midpoint = (t[n] + t[n-1]) / 2
        #get_load_vector!(F, midpoint, f_homogeneous, dof)
        #rhs .= τ .* F - (τ/2) .* (A * U_free[1:end-1,n-1,m])
        rhs .= - (τ/2) .* (A * U_free[:, n-1, m])
        Σ .= ω[n][1] * U_free[:, 0, m]
        for j = 1:n-1
            Σ .+= (ω[n][j+1] - ω[n][j]) .* U_free[:, j, m]
        end
        rhs .= rhs + M_free * Σ 
        v = view(U_free, 1:Nₕ, n, m)
        num_its[n,m] = pcg!(v, B, rhs, P[index_τ], tol, wkspace)
    end
    uh = [ U_free[:, Nₜ, m]; U_fix[:, Nₜ, m] ]
    L[m],_ = average_field(uh, "Omega", dof)
end
elapsed = time() - start
println("Use $elapsed secs")
average_num_its = mean(num_its, dims=2)

figure(1)
hist(L)
title("Histogram of all different values")

figure(2)
plot(t[1:Nₜ],average_num_its)
title("Average iterations of pcg for time step")