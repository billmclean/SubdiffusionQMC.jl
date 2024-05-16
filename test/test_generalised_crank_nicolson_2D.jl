include("generalised_c_k_2D_profile.jl")

κ_const = 0.02
kx, ky = 1, 1
λ = κ_const * (kx^2 + ky^2) * π^2

x, y, triangles = gmsh2pyplot(dof)
T = 1.0
Nₜ = 20
α = 0.5
γ = 2 / α
t = graded_mesh(Nₜ, γ, T)

u_homogeneous(x, y, t) = E_half(-λ * sqrt(t)) * sinpi(kx * x) * sinpi(ky * y)
exact_u_homogeneous = get_nodal_values((x, y) -> u_homogeneous(x, y, T), dof)
f_homogeneous(x, y, t) = 0.0
u₀_homogeneous(x, y) = u_homogeneous(x, y, 0.0)
pstore = PDEStore((x, y) -> κ_const, f_homogeneous, dof, 
                 solver, pcg_tol, pcg_maxiterations)
if fast_method
    r = 2
    tol = 1e-8
    Δx = 0.5
    Nₕ = dof.num_free
    estore = ExponentialSumStore(t, Nₕ, α, r, tol, Δx)
    U_free = IBVP_solution((x, y) -> κ_const, f_homogeneous, u₀_homogeneous, pstore, estore)
    else
    U_free = IBVP_solution(t, α, (x, y) -> κ_const, f_homogeneous, u₀_homogeneous, pstore)
end
U_fix = OMat64(zeros(dof.num_fixed, Nₜ+1), 1:dof.num_fixed, 0:Nₜ)
U = [U_free; U_fix]

figure(1)
plot_trisurf(x, y, triangles, U[:, Nₜ], cmap="cool")
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
