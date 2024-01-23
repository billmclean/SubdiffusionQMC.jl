using SubdiffusionQMC
import SpecialFunctions: gamma
using PyPlot

function weights_and_exponents(M::Integer, N::Integer, Δx::Float64, β::Float64)
    a = OVec64(undef, -M:N)
    w = OVec64(undef, -M:N)
    for n in eachindex(a)
	xₙ = n * Δx
        pₙ = xₙ - exp(-xₙ)
	a[n] = exp(pₙ)
	w[n] = exp(β * pₙ) * (1 + exp(-xₙ)) * Δx
    end
    return w, a
end

function relative_error(t::Float64, β::Float64, w::OVec64, a::OVec64, 
	tol::Float64)
    Σ₋ = 0.0
    M = -1
    for n = -1:-1:firstindex(a)
	next_term = w[n] * exp(-a[n] * t)
	Σ₋ += next_term
	if next_term < tol
	    M = -n
	    break
	end
    end
    if M == -1
	error("M is too small")
    end
    Σ₊ = 0.0
    N = -1
    for n = 0:lastindex(a)
	next_term = w[n] * exp(-a[n] * t)
	Σ₊ += next_term
	if next_term < tol
	    N = n
	    break
	end
    end
    if N == -1
	error("N is too small")
    end
    ρ = 1 - t^β * (Σ₊ + Σ₋) / gamma(β)
    return ρ, M, N
end

npts = 500
t = 10.0 .^ range(-6, -1, npts)
ρ = similar(t)
N = Vector{Int64}(undef, npts)
M = Vector{Int64}(undef, npts)

maxM = 50
maxN = 50
Δx = 0.5
β = 0.75
tol = 1e-5

w, a = weights_and_exponents(maxM, maxN, Δx, β)

for k in eachindex(t)
    ρ[k], M[k], N[k] = relative_error(t[k], β, w, a, tol)
end

figure(1)
semilogx(t, ρ)
grid(true)
xlabel(L"$t$")
title("Relative error")

figure(2)
plot(t, M, t, N)
grid(true)
legend((L"$M$", L"$N$"))
xlabel(L"$t$")
title("Numbers of terms used")

figure(3)
lo = -maximum(M)
hi = maximum(N)
semilogy(lo:hi, w[lo:hi], lo:hi, a[lo:hi])
grid(true)
legend((L"$w_n$", L"$a_n$"))
xlabel(L"$n$")
title("Weights and exponents")
