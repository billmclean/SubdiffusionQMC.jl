module PDE

using SimpleFiniteElements
import SimpleFiniteElements.FEM: average_field, assemble_vector!
import LinearAlgebra: BLAS, cholesky
import SimpleFiniteElements.Poisson: ∫∫a_∇u_dot_∇v!, ∫∫c_u_v!, ∫∫f_v!
import ..PDEStore_integrand, ..DiffusivityStore2D, ..ExponentialSumStore, ..pcg!
import ..generalised_crank_nicolson_2D!, ..weights
import ..Vec64, ..AVec64, ..Mat64, ..OMat64, ..OVec64, ..AMat64, ..SparseCholeskyFactor, ..IdxPair
import ..interpolate_κ!, ..slow_κ
import ..SparseCholeskyFactor

import ..integrand_init!, ..integrand!, ..slow_integrand!

function PDEStore_integrand(κ₀::Function, dof::DegreesOfFreedom,
              solver, pcg_tol::Float64, pcg_maxits::Int)
    u_free_det = Vec64(undef, dof.num_free)
    b = Vector{Float64}(undef, dof.num_free)
    P = Vector{SparseCholeskyFactor}()
    wkspace = Mat64(undef, dof.num_free, 4)
    u_free = Vec64(undef, dof.num_free)
    u_free_det = Vec64(undef, dof.num_free)
    bilinear_forms_M = Dict("Omega" => (∫∫c_u_v!, 1.0))
    return PDEStore_integrand(κ₀, dof, b, solver, P,
                    wkspace, u_free_det, u_free, pcg_tol, pcg_maxits, bilinear_forms_M)
end

function integrand_init!(estore::ExponentialSumStore, pstore::PDEStore_integrand, 
                          f::Function, get_load_vector!::Function, u₀::Function)
    κ₀ = pstore.κ₀
    bilinear_forms_A = Dict("Omega" => (∫∫a_∇u_dot_∇v!, κ₀))
    deterministic_solve!(estore, pstore, bilinear_forms_A, get_load_vector!, f, u₀)
end

function integrand_init!(α::Float64, t::OVec64, pstore::PDEStore_integrand, 
                          f::Function, get_load_vector!::Function, u₀::Function)
    κ₀ = pstore.κ₀
    bilinear_forms_A = Dict("Omega" => (∫∫a_∇u_dot_∇v!, κ₀))
    deterministic_solve!(α, t, pstore, bilinear_forms_A, get_load_vector!, f, u₀)
end

function generate_τ_values(lo, hi)
    τ_values = [10.0^k for k in lo : hi]
    return τ_values
end

function deterministic_solve!(estore::ExponentialSumStore, pstore::PDEStore_integrand, 
                              bilinear_forms_A::Dict,
                              get_load_vector!::Function, f::Function, u₀::Function)
    (; κ₀, dof, P, wkspace, pcg_tol, bilinear_forms_M) = pstore
    (;t, ω) = estore
    Nₜ = lastindex(t)
    A_free, _ = assemble_matrix(dof, bilinear_forms_A)
    M_free, _ = assemble_matrix(dof, bilinear_forms_M)
    num_free, num_fixed = dof.num_free, dof.num_fixed
    u_free_det = OMat64(zeros(num_free, Nₜ+1), 1:num_free, 0:Nₜ)
    u_fix = OMat64(zeros(num_fixed, Nₜ+1), 1:num_fixed, 0:Nₜ)
    u0h = get_nodal_values(u₀, dof)
    u_free_det[:,0] = u0h[1:num_free]
    generalised_crank_nicolson_2D!(u_free_det, M_free, A_free, 
                                     get_load_vector!, estore, pstore, f)
    uh_det = [u_free_det[:,Nₜ]; u_fix[:,Nₜ]]
    Φ_det, _ = average_field(uh_det, "Omega", dof)
    return Φ_det
end

function deterministic_solve!(α::Float64, t::OVec64, pstore::PDEStore_integrand, 
                              bilinear_forms_A::Dict,
                              get_load_vector!::Function, f::Function, u₀::Function)
    (; κ₀, dof, pcg_tol, bilinear_forms_M) = pstore
    Nₜ = lastindex(t)
    A_free, _ = assemble_matrix(dof, bilinear_forms_A)
    M_free, _ = assemble_matrix(dof, bilinear_forms_M)
    num_free, num_fixed = dof.num_free, dof.num_fixed
    u_free_det = OMat64(zeros(num_free, Nₜ+1), 1:num_free, 0:Nₜ)
    u_fix = OMat64(zeros(num_fixed, Nₜ+1), 1:num_fixed, 0:Nₜ)
    u0h = get_nodal_values(u₀, dof)
    u_free_det[:,0] = u0h[1:num_free]
    generalised_crank_nicolson_2D!(u_free_det, M_free, A_free, α, t,
                                     get_load_vector!, pstore, f)
    uh_det = [u_free_det[:,Nₜ]; u_fix[:,Nₜ]]
    Φ_det, _ = average_field(uh_det, "Omega", dof)
    return Φ_det
end

function integrand!(y_vals::AVec64, κ₀_vals::Mat64, estore::ExponentialSumStore, 
                     pstore::PDEStore_integrand, dstore::DiffusivityStore2D, 
                     solver, f::Function, get_load_vector!::Function, u₀::Function)
    κ_ = interpolate_κ!(y_vals, κ₀_vals, dstore)
    bilinear_forms_A = Dict("Omega" => (∫∫a_∇u_dot_∇v!, (x, y) -> κ_(x, y)))
    random_solve!(solver, estore, pstore, bilinear_forms_A, get_load_vector!, f, u₀)
end

function integrand!(y_vals::AVec64, κ₀_vals::Mat64, α::Float64, t::OVec64,
                     pstore::PDEStore_integrand, dstore::DiffusivityStore2D, 
                     solver, f::Function, get_load_vector!::Function, u₀::Function)
    κ_ = interpolate_κ!(y_vals, κ₀_vals, dstore)
    bilinear_forms_A = Dict("Omega" => (∫∫a_∇u_dot_∇v!, (x, y) -> κ_(x, y)))
    random_solve!(solver, α, t, pstore, bilinear_forms_A, get_load_vector!, f, u₀)
end

function random_solve!(solver, α::Float64, t::OVec64, pstore::PDEStore_integrand, 
                        bilinear_forms_A::Dict, 
                        get_load_vector!::Function, f::Function, u₀::Function)
    (; κ₀, dof, P, wkspace, pcg_tol, bilinear_forms_M) = pstore
    Nₜ = lastindex(t)
    A_free, _ = assemble_matrix(dof, bilinear_forms_A)
    M_free, _ = assemble_matrix(dof, bilinear_forms_M)
    num_free, num_fixed = dof.num_free, dof.num_fixed
    wkspace = Mat64(undef, num_free, 4)
    u_free = OMat64(zeros(num_free, Nₜ+1), 1:num_free, 0:Nₜ)
    u_fix = OMat64(zeros(num_fixed, Nₜ+1), 1:num_fixed, 0:Nₜ)
    u0h = get_nodal_values(u₀, dof)
    u_free[:,0] = u0h[1:dof.num_free]
    lo = floor(Int, log10(t[1]-t[0]))
    hi = ceil(Int, log10(t[end] - t[end-1]))
    τ_values = generate_τ_values(lo, hi)
    if solver == :direct
        generalised_crank_nicolson_2D!(u_free, M_free, A_free, α, t, 
                                                 get_load_vector!, pstore, f)
        else solver == :pcg
        solve_pcg!(u_free, M_free, A_free, get_load_vector!, α, t, pstore, f, τ_values)
    end
    uh = [u_free[:,Nₜ]; u_fix[:,Nₜ]]
    Φ, _ = average_field(uh, "Omega", dof)
    return Φ
end

function random_solve!(solver, estore::ExponentialSumStore, pstore::PDEStore_integrand, 
                        bilinear_forms_A::Dict, 
                        get_load_vector!::Function, f::Function, u₀::Function)
    (; κ₀, dof, P, wkspace, pcg_tol, bilinear_forms_M) = pstore
    (;α, r, t, ω, S, a, w) = estore
    Nₜ = lastindex(t)
    A_free, _ = assemble_matrix(dof, bilinear_forms_A)
    M_free, _ = assemble_matrix(dof, bilinear_forms_M)
    num_free, num_fixed = dof.num_free, dof.num_fixed
    u_free = OMat64(zeros(num_free, Nₜ+1), 1:num_free, 0:Nₜ)
    u_fix = OMat64(zeros(num_fixed, Nₜ+1), 1:num_fixed, 0:Nₜ)
    u0h = get_nodal_values(u₀, dof)
    u_free[:,0] = u0h[1:dof.num_free]
    lo = floor(Int, log10(t[1]-t[0]))
    hi = ceil(Int, log10(t[end] - t[end-1]))
    τ_values = generate_τ_values(lo, hi)
    if solver == :direct
        generalised_crank_nicolson_2D!(u_free, M_free, A_free, 
                                                 get_load_vector!, estore, pstore, f)
        else solver == :pcg
        slove_expsum_pcg!(u_free, M_free, A_free, get_load_vector!, estore, pstore, f, τ_values)
    end
    uh = [u_free[:,Nₜ]; u_fix[:,Nₜ]]
    Φ, _ = average_field(uh, "Omega", dof)
    return Φ
end

function solve_pcg!(U::OMat64, M::AMat64, A::AMat64, get_load_vector!::Function, 
                     α::Float64, t::OVec64, pstore::PDEStore_integrand, f::Function,
                    τ_values::Vec64)
    (; dof, P, wkspace, pcg_tol) = pstore
    ω = weights(α, t)
    num_free = size(U, 1)
    Nₜ = lastindex(t)
    F = Vec64(undef, num_free)
    rhs = similar(F)
    Σ = similar(F)
    num_its = Vec64(zeros(Nₜ))
    wkspace = Mat64(undef, num_free, 4)
    P = Vector{SparseCholeskyFactor}(undef, length(τ_values))
    for k in eachindex(τ_values)
        P[k] = cholesky(M + (τ_values[k]/2) * A)
    end
    for n = 1 : Nₜ
        τ = t[n] - t[n-1]
        index_τ = argmin(abs.(τ_values .- τ))
        B = ω[n][n] * M + (τ/2) * A
        midpoint = (t[n] + t[n-1]) / 2
        get_load_vector!(F, midpoint, pstore, f)
        rhs .= τ .* F - (τ/2) .* (A * U[1:num_free,n-1])
        Σ .= ω[n][1] * U[1:num_free, 0]
        for j = 1:n-1
            if j + 1 <= length(ω[n])
                Σ .+= (ω[n][j+1] - ω[n][j]) .* U[1:num_free, j]
            end
        end
        rhs .= rhs + M * Σ 
        v = view(U, 1:num_free, n)
        num_its[n] = pcg!(v, B, rhs, P[index_τ], pcg_tol, wkspace)
    end
end

function slove_expsum_pcg!(U::OMat64, M::AMat64, A::AMat64, get_load_vector!::Function, 
                         estore::ExponentialSumStore, pstore::PDEStore_integrand, f::Function, 
                         τ_values::Vec64)
    (; dof, P, wkspace, pcg_tol) = pstore
    (;α, r, t, ω, S, a, w) = estore
    num_free = size(U, 1)
    wkspace = Mat64(undef, num_free, 4)
    Nₜ = lastindex(t)
    F = Vec64(undef, num_free)
    rhs = similar(F)
    Σ = similar(F)
    num_its = Vec64(zeros(Nₜ))
    P = Vector{SparseCholeskyFactor}(undef, length(τ_values))
    for k in eachindex(τ_values)
        P[k] = cholesky(M + (τ_values[k]/2) * A)
    end
    for n = 1:r
        τₙ = t[n] - t[n-1]
        index_τ = argmin(abs.(τ_values .- τₙ))
        B = ω[n][n] * M + (τₙ/2) * A
        midpoint = (t[n] + t[n-1]) / 2
        get_load_vector!(F, midpoint, pstore, f)
        rhs .= τₙ .* F - (τₙ/2) .* (A * U[1:num_free,n-1])
        Σ .= ω[n][1] .* U[1:num_free,0]
        for j = 1:n-1
            Σ .+= (ω[n][j+1] - ω[n][j]) .* U[1:num_free,j]
        end
        rhs .= rhs + M * Σ
        v = view(U, 1:num_free, n)
        num_its[n] = pcg!(v, B, rhs, P[index_τ], pcg_tol, wkspace)
    end
    fill!(S, 0.0)
    for n = r+1:Nₜ
        τₙ   = t[n] - t[n-1]
        τₙ₋₁ = t[n-1] - t[n-2]
        τₙ₋ᵣ = t[n-r] - t[n-r-1]
        index_τ = argmin(abs.(τ_values .- τₙ))
        B = ω[n][n] * M + (τₙ/2) * A
        midpoint = (t[n] + t[n-1]) / 2
        get_load_vector!(F, midpoint, pstore, f)
        rhs .= τₙ .* F - (τₙ/2) .* (A * U[1:num_free,n-1])
        fill!(Σ, 0.0)
        for m = firstindex(a):lastindex(a)
            c = expm1(-a[m] * τₙ)
            μ = ( w[m] * exp(-a[m] * (t[n-1] - t[n-r])) * (c / a[m])
                   * (expm1(-a[m] * τₙ₋ᵣ) / (a[m] * τₙ₋ᵣ)) )
            ν = -c / expm1(a[m] * τₙ₋₁)
            S[:,m] .= μ * (U[1:num_free,n-r] - U[1:num_free,n-r-1]) + ν * S[:,m]
            Σ .+= S[:,m]
        end
        Σ .= ω[n][n-r+1] * U[1:num_free,n-r] - (sinpi(α) / π) * Σ
        for j = n-r+1:n-1
            Σ .+= (ω[n][j+1] - ω[n][j]) .* U[1:num_free,j]
        end
        rhs .= rhs + M * Σ
        v = view(U, 1:num_free, n)
        num_its[n] = pcg!(v, B, rhs, P[index_τ], pcg_tol, wkspace)
    end
end

function slow_integrand!(y_vals::AVec64, κ₀::Function,
                     estore::ExponentialSumStore, 
                     pstore::PDEStore_integrand, dstore::DiffusivityStore2D, 
                     solver, f::Function, get_load_vector!::Function, u₀::Function)
    slow_κ_(x₁, x₂) = slow_κ( x₁, x₂, y_vals, κ₀, dstore)
    bilinear_forms_A = Dict("Omega" => (∫∫a_∇u_dot_∇v!, slow_κ_))
    random_solve!(solver, estore, pstore, bilinear_forms_A, get_load_vector!, f, u₀)
end

end #module