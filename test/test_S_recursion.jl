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

Nₜ = 20
Nₕ = 2 # only needed to create estore; not actuall used
T = 1.0
ε = 0.1
γ = 2 / α
#γ = 1.0
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
x_lo, x_hi, y_lo, y_hi = axis()
axis([x_lo, x_hi, 0.0, 5.0])
grid(true)
xlabel(L"$t-s$")
legend((L"$\omega_{1-\alpha}(t-s)$", L"sum"))

S = OMat64(undef, -M₋:M₊, r:Nₜ)
S[:,r] .= 0.0
μ = similar(S)
ν = similar(S)
J₂ = OVec64(undef, r:Nₜ)
fill!(J₂, 0.0)
Ĵ₂ = similar(J₂)
fill!(Ĵ₂, 0.0)
for n = r+1:Nₜ
    global S, J₂, Ĵ₂, μ, ν
    for j = 1:n-r
	J₂[n] += ω[n][j] * ( U[j] - U[j-1] )
    end
    τₙ   = t[n] - t[n-1]
    τₙ₋₁ = t[n-1] - t[n-2]
    τₙ₋ᵣ = t[n-r] - t[n-r-1]
    for m = firstindex(a):lastindex(a)
        c = expm1(-a[m] * τₙ)
	μ[m,n] = ( w[m] * exp(-a[m] * (t[n-1] - t[n-r])) * (c / a[m])
                   * (expm1(-a[m] * τₙ₋ᵣ) / (a[m] * τₙ₋ᵣ)) )
	ν[m,n] = -c / expm1(a[m] * τₙ₋₁)
	if isnan(μ[m,n])
	    println("μ[$m, $n] = $(μ[m,n])")
	end
	if isnan(ν[m,n])
	    println("ν[$m, $n] = $(ν[m,n])")
	end
	S[m,n] = μ[m,n] * (U[n-r] - U[n-r-1]) + ν[m,n] * S[m,n-1]
	Ĵ₂[n] += S[m,n]
    end
    Ĵ₂[n] *= sinpi(α) / π
end

max_error = maximum(abs, J₂ - Ĵ₂)
@printf("Maximum error = %8.3e\n", max_error)

figure(3, figsize=(12.8, 4.8))
subplot(1, 2, 1)
loglog(t[r+1:Nₜ], μ[0:M₊,r+1:Nₜ]', "o-")
grid(true)
x_lo, x_hi, y_lo, y_hi = axis()
y_hi = maximum(μ[0:M₊,r+1:Nₜ])
axis([x_lo, x_hi, 1e-16, y_hi])
xlabel(L"$t$")
title("μₙₘ for m ≥ 0")
subplot(1, 2, 2)
loglog(t[r+1:Nₜ], μ[-M₋:-1,r+1:Nₜ]', "o-")
grid(true)
x_lo, x_hi, y_lo, y_hi = axis()
axis([x_lo, x_hi, 1e-16, y_hi])
xlabel(L"$t$")
title("μₙₘ for m ≤ -1")

figure(4, figsize=(12.8, 4.8))
subplot(1, 2, 1)
loglog(t[r+1:Nₜ], ν[0:M₊,r+1:Nₜ]', "o-")
grid(true)
x_lo, x_hi, y_lo, y_hi = axis()
y_hi = maximum(ν[0:M₊,r+1:Nₜ])
axis([x_lo, x_hi, 1e-16, y_hi])
xlabel(L"$t$")
title("νₙₘ for m ≥ 0")
subplot(1, 2, 2)
semilogx(t[r+1:Nₜ], ν[-M₋:-1,r+1:Nₜ]', "o-")
grid(true)
x_lo, x_hi, y_lo, y_hi = axis()
y_lo = minimum(ν[-M₋:-1,r+1:Nₜ])
y_lo = max(floor(y_lo), 1e-16)
y_hi = maximum(ν[-M₋:-1,r+1:Nₜ])
axis([x_lo, x_hi, y_lo, y_hi])
xlabel(L"$t$")
title("νₙₘ for m ≤ -1")
