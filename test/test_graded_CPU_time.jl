include("CPU_time_profile.jl")
include("graded_time_mesh.jl")
for n = 1:Nₜ
    global U_free, ΔU, τ
    τ = t[n] - t[n-1]
    B = M + (τ/2) * A
    P = cholesky(B)
    midpoint = (t[n] + t[n-1]) / 2
    get_load_vector!(F, midpoint, f_homogeneous, dof)
    mul!(rhs, A, U_free[1:Nₕ,n-1])
    scal!(-τ, rhs) # rhs = τ F - τ A Uⁿ⁻¹
    fill!(ΔU, 0.0)
    #cg time
    start = time()
    num_its_cg = cg!(ΔU, B, rhs, tol, wkspace)
    elapsed_cg = time() - start
    #pcg time 
    start = time()
    num_its_pcg = pcg!(ΔU, B, rhs, P, tol, wkspace)
    elapsed_pcg = time() - start
    U_free[:,n] .= ΔU + U_free[:,n-1]
    println("num_its_cg=$num_its_cg")
    println("rate for cg!:", elapsed_cg, "seconds")
    #println("num_its_pcg=$num_its_pcg")
    println("rate for pcg!:", elapsed_pcg, "seconds")
    #direct method
    start = time()
    uh_free = IBVP_solution(t, (x, y) -> κ₀, f_homogeneous, u₀_homogeneous, pstore)
    elapsed_direct = time() - start
    println("rate for direct:", elapsed_direct, "seconds")
end
