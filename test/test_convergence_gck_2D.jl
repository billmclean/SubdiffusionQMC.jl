include("generalised_c_k_2D_profile.jl")

κ_const = 0.02
kx, ky = 1, 1
λ = κ_const * (kx^2 + ky^2) * π^2
T = 1.0
α = 0.5
γ = 2 / α

u_homogeneous(x, y, t) = E_half(-λ * sqrt(t)) * sinpi(kx * x) * sinpi(ky * y)
f_homogeneous(x, y, t) = 0.0
u₀_homogeneous(x, y) = u_homogeneous(x, y, 0.0)

function compute_error_time(nrows::Int, hmax::Float64, Nₜ::Int)
    mesh = FEMesh(gmodel, hmax)
    dof = DegreesOfFreedom(mesh, essential_bcs)
    x, y, triangles = gmsh2pyplot(dof)
    pstore = PDEStore((x, y) -> κ_const, f_homogeneous, dof, 
                solver, pcg_tol, pcg_maxiterations)
    max_error = zeros(nrows)
    @printf("\n%6s  %6s  %10s  %8s  %8s\n\n", 
        "Nₜ", "hmax", "Error", "rate", "seconds")
    for k = 1 : nrows
        local uh_free, t
        Nₜ *= 2
        t = graded_mesh(Nₜ, γ, T)
        start = time()
        if fast_method
            r = 2
            tol = 1e-8
            Δx = 0.5
            Nₕ = dof.num_free
            estore = ExponentialSumStore(t, Nₕ, α, r, tol, Δx)
            uh_free = IBVP_solution((x, y) -> κ_const, f_homogeneous, u₀_homogeneous, pstore, estore)
            else
            uh_free = IBVP_solution(t, α, (x, y) -> κ_const, f_homogeneous, u₀_homogeneous, pstore)
        end
        for i = 0 : lastindex(t)
            u = get_nodal_values((x, y) -> u_homogeneous(x, y, t[i]), dof)
            max_i = maximum(abs, uh_free[:,i] - u[1:dof.num_free])
            max_error[k] = max(max_error[k], max_i)
        end
        elapsed = time() - start
        if k == 1
            @printf("%6d  %6f  %10.2e  %8s  %8.3f\n", 
                Nₜ, hmax, max_error[k], "", elapsed)
        else
            rate = log2(max_error[k-1]/max_error[k])
            @printf("%6d  %6f   %10.2e  %8.3f  %8.3f\n", 
                Nₜ, hmax, max_error[k], rate, elapsed)
        end
    end
end

function compute_error_space(nrows::Int, hmax::Float64, Nₜ::Int)
    mesh = FEMesh(gmodel, hmax; refinements=nrows-1)
    t = graded_mesh(Nₜ, γ, T)
    max_error = zeros(nrows)
    @printf("\n%6s  %8s  %6s  %10s  %8s  %8s\n\n", 
        "Nₜ", "dof", "hmax", "Error", "rate", "seconds")
    for k = 1 : nrows
        dof = DegreesOfFreedom(mesh[k], essential_bcs)
        x, y, triangles = gmsh2pyplot(dof)
        hmax /= 2
        hmax = max_elt_diameter(mesh[k])
        pstore = PDEStore((x, y) -> κ_const, f_homogeneous, dof, 
                    solver, pcg_tol, pcg_maxiterations)
        start = time()
        if fast_method
            r = 2
            tol = 1e-8
            Δx = 0.5
            Nₕ = dof.num_free
            estore = ExponentialSumStore(t, Nₕ, α, r, tol, Δx)
            uh_free = IBVP_solution((x, y) -> κ_const, f_homogeneous, u₀_homogeneous, pstore, estore)
            else
            uh_free = IBVP_solution(t, α, (x, y) -> κ_const, f_homogeneous, u₀_homogeneous, pstore)
        end
        for i = 0 : lastindex(t)
            u = get_nodal_values((x, y) -> u_homogeneous(x, y, t[i]), dof)
            max_i = maximum(abs, uh_free[:,i] - u[1:dof.num_free])
            max_error[k] = max(max_error[k], max_i)
        end
        elapsed = time() - start
        if k == 1
            @printf("%6d  %8d  %6f  %10.2e  %8s  %8.3f\n", 
                Nₜ, dof.num_free, hmax, max_error[k], "", elapsed)
        else
            rate = log2(max_error[k-1]/max_error[k])
            @printf("%6d  %8d  %6f  %10.2e  %8.3f  %8.3f\n", 
                Nₜ, dof.num_free, hmax, max_error[k], rate, elapsed)
        end
    end
end


# Usage_space
#compute_error_space(nrows, hmax, Nₜ)
compute_error_space(5, 1/2, 1024)

# Usage_time
#nrows = 4, hmax = 1/1200, Nₜ = 2
#compute_error_time(nrows, hmax, Nₜ)
#compute_error_time(4, 1/1200, 2)