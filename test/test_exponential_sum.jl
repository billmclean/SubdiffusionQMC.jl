using SubdiffusionQMC
using PyPlot
using OffsetArrays
import SpecialFunctions: gamma

α = 0.75
Nₜ = 20
Nₕ = 15
γ = 2.5
final_time = 2.0
r = 3
tol = 1e-8
Δx = 0.5
t = graded_mesh(Nₜ, γ, final_time)
estore = ExponentialSumStore(t, Nₕ, α, r, tol, Δx)

npts = 500
t_minus_s = 10.0 .^ range(log10(estore.δ), log10(t[Nₜ]), npts)
ρ = similar(t_minus_s)
M₊ = OffsetVector{Int64}(undef, npts)
M₋ = similar(M₊)
for n in eachindex(t_minus_s)
    Σ, M₊[n], M₋[n] = exponential_sum(t_minus_s[n], estore)
    ρ[n] = 1 - gamma(1-α) * t_minus_s[n]^α * Σ
end

figure(1)
semilogx(t_minus_s, ρ)
grid(true)
xlabel(L"$t-s$")
title(L"Relative error approximating $\omega_{1-\alpha}(t-s)$")

figure(2)
plot(t_minus_s, M₊, t_minus_s, M₋)
grid(true)
legend((L"$M_+$", L"$M_-$"))
xlabel(L"$t$")
title("Numbers of terms used (Δx = $Δx)")

figure(3)
lo = -maximum(M₋)
hi = maximum(M₊)
semilogy(lo:hi, estore.w[lo:hi], "o", lo:hi, estore.a[lo:hi], "o")
grid(true)
legend((L"$w_n$", L"$a_n$"))
xlabel(L"$n$")
title("Weights and exponents")


