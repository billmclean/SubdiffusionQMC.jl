include("average_profile.jl")

tol = 1e-7

path = joinpath("..", "spatial_domains", "unit_square.geo")
gmodel = GeometryModel(path)
essential_bcs = [("Gamma", 0.0)]

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
t = collect(range(0, T, Nₜ+1))
t = OVec64(t, 0:Nₜ)

U_det = OMat64(zeros(dof.num_free, Nₜ+1), 1:dof.num_free, 0:Nₜ)
U_det_fix = OMat64(zeros(dof.num_fixed, Nₜ+1), 1:dof.num_fixed, 0:Nₜ)
Nₛ = dof.num_free + dof.num_fixed
U_det[:,0] = uh0[1:dof.num_free]
ΔU_det = zeros(dof.num_free)
Nₕ = lastindex(U_det, 1)
F_det = Vec64(undef, Nₕ)
rhs_det = similar(F_det)
wkspace = zeros(Nₕ, 4)

#pcg method for u₀_bent
#for n = 1:Nₜ
#    global U_det, ΔU_det, τ
#    local num_its
#    τ = t[n] - t[n-1]
#    B = M_free + (τ/2) *A
#    P = cholesky(B)
#    midpoint = (t[n] + t[n-1]) / 2
#    get_load_vector!(F_det, midpoint, f_homogeneous, dof)
#    mul!(rhs_det, A, U_det[1:Nₕ,n-1])
#    scal!(-τ, rhs_det)
#    fill!(ΔU_det, 0.0)
#    num_its = SubdiffusionQMC.pcg!(ΔU_det, B, rhs_det, P, tol, wkspace)
#    U_det[:,n] .= ΔU_det + U_det[:,n-1]
#end
#uh = [U_det[:,Nₜ]; U_det_fix[:,Nₜ]]
#L₀, _ = average_field(uh, "Omega", dof)

M = 200
L = zeros(Float64, M)
U_free = OffsetArray(zeros(dof.num_free, Nₜ+1, M), 1:dof.num_free, 0:Nₜ, 1:M)
U_fix = OffsetArray(zeros(dof.num_fixed, Nₜ+1, M), 1:dof.num_fixed, 0:Nₜ, 1:M)
Nₛ = dof.num_free + dof.num_fixed
ΔU = zeros(dof.num_free)
Nₕ = lastindex(U_free, 1)
F = Vec64(undef, Nₕ)
rhs = similar(F)
num_its = Mat64(zeros(Nₜ,M))
τ_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

start = time()
τ = 1 / Nₜ
τ_pre = 4 * τ
B = M_free + (τ/2) *A
B_pre = M_free + (τ_pre/2) * A
P = cholesky(B_pre)
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
        global U_free, ΔU, τ, num_its
        #τ = t[n] - t[n-1]
        midpoint = (t[n] + t[n-1]) / 2
        get_load_vector!(F, midpoint, f_homogeneous, dof)
        mul!(rhs, A, U_free[1:Nₕ,n-1,m])
        scal!(-τ, rhs) # rhs = τ F - τ A Uⁿ⁻¹
        fill!(ΔU, 0.0)
        num_its[n,m] = SubdiffusionQMC.pcg!(ΔU, B, rhs, P, tol, wkspace)
        U_free[:,n,m] .= ΔU + U_free[:,n-1,m]
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