using SubdiffusionQMC
import SubdiffusionQMC.TimeStepping: Taylor_coefficients!
using PyPlot

α = 0.5
M = 10
C = zeros(M)
Taylor_coefficients!(C, α)

δ = collect(range(0, 1/2, 201))
S = similar(δ)
for j in eachindex(S)
    Σ = 0.0
    power_of_δ = 1.0
    for m in eachindex(C)
	power_of_δ *= δ[j]^2
	Σ += C[m] * power_of_δ
    end
    S[j] = 2 + Σ
end

figure(1)

semilogy(δ, (1 .+ δ).^(2-α) + (1 .- δ).^(2-α) - S)
grid(true)
title("Error in Taylor expansion with $M terms")
