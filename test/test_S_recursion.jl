using SubdiffusionQMC
import SubdiffusionQMC.FEM1D: load_vector!, stiffness_matrix, mass_matrix
import SpecialFunctions: erfcx, gamma
using PyPlot
import Printf: @printf

α = 0.5
κ = 0.02
k = 1
λ = κ * (k * π)^2
E_half(t) = erfcx(-t)
u_homogeneous(t) = E_half(-λ * sqrt(t))

Nₜ = 5
T = 1.0
ε = 0.1
#γ = 2 / α
γ = 1.0
t = graded_mesh(Nₜ, γ, T)

U = OVec64(zeros(Nₜ+1), 0:Nₜ)
for n in eachindex(t)
    U[n] = u_homogeneous(t[n])
end
ω = weights(α, t)

r = 2
tol = 1e-8
Δx = 0.5
estore = ExponentialSumStore(t, Nₕ, α, r, tol, Δx)
(;α, r, t, S, a, w, δ) = estore
M₋ = -firstindex(a)
M₊ = lastindex(a)

npts = 500
t_minus_s = 10.0 .^ range(log10(estore.δ), log10(t[Nₜ]), npts)
ρ = similar(t_minus_s)
Σ = similar(t_minus_s)
for n in eachindex(t_minus_s)
    Σ[n], _, _ = exponential_sum(t_minus_s[n], estore)
    ρ[n] = 1 - gamma(1-α) * t_minus_s[n]^α * Σ[n]
end

npts = 500
t_minus_s = 10.0 .^ range(log10(estore.δ), log10(t[Nₜ]), npts)
figure(1)
semilogx(t_minus_s, ρ)
grid(true)
xlabel(L"$t-s$")
title(L"Relative error approximating $\omega_{1-\alpha}(t-s)$")

figure(2)
plot(t_minus_s, Σ, t_minus_s, t_minus_s.^(-α)/gamma(1-α), "--")
grid(true)
xlabel(L"$t-s$")
legend((L"$\omega_{1-\alpha}(t-s)$", L"sum"))

S = OMat64(undef, -M₋:M₊, r:Nₜ)
S[:,r] .= 0.0
J₂ = OVec64(undef, r:Nₜ)
fill!(J₂, 0.0)
Ĵ₂ = similar(J₂)
fill!(Ĵ₂, 0.0)
for n = r+1:Nₜ
    global S, J₂, Ĵ₂
    for j = 1:n-r
	J₂[n] += ω[n][j] * ( U[j] - U[j-1] )
    end
    τₙ   = t[n] - t[n-1]
    τₙ₋₁ = t[n-1] - t[n-2]
    τₙ₋ᵣ = t[n-r] - t[n-r-1]
    for m = firstindex(a):lastindex(a)
        c = expm1(-a[m] * τₙ)
        μ = ( w[m] * exp(-a[m] * (t[n-1] - t[n-r])) * (c / a[m])
                   * (expm1(-a[m] * τₙ₋ᵣ) / (a[m] * τₙ₋ᵣ)) )
        ν = -c / expm1(a[m] * τₙ₋₁)
        S[m,n] = μ * (U[n-r] - U[n-r-1]) + ν * S[m,n-1]
	Ĵ₂[n] += S[m,n]
    end
    Ĵ₂[n] *= sinpi(α) / π
end

max_error = maximum(abs, J₂ - Ĵ₂)
@printf("Maximum error = %8.3e\n", max_error)
