using SubdiffusionQMC
import SubdiffusionQMC.Timestepping: Taylor_coefficients!
using PyPlot

α = 0.5
M = 8
C = zeros(M+1)
Taylor_coefficients!(C, α)

δ = collect(range(0, 1/2, 201))
Taylor_sum = similar(δ)
error_bound = similar(δ)
power_of_δ = similar(C)
for j in eachindex(δ)
    Σ = 0.0
    power_of_δ[1] = δ[j]^2
    for m in 1:M
	Σ += C[m] * power_of_δ[m]
	power_of_δ[m+1] = power_of_δ[1] * power_of_δ[m]
    end
    error_bound[j] = C[M+1] * power_of_δ[M+1]
    Taylor_sum[j] = 2 + Σ
end

figure(1)

error_bound = max.(error_bound, eps(Float64))
actual_error = (1 .+ δ).^(2-α) + (1 .- δ).^(2-α) - Taylor_sum

semilogy(δ, actual_error, δ, error_bound, "--")
legend(("Error", "Error bound"))
grid(true)
title("Error in Taylor expansion with $M terms")
