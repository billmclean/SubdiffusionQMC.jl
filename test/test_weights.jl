using SubdiffusionQMC
using PyPlot

Nₜ = 250
ω = Dict()
for T in [Float64, BigFloat]
    local α, γ, t, final_time
    α = parse(T, "0.5")
    γ = parse(T, "3.5")
    final_time = parse(T, "4.0")
    t = graded_mesh(Nₜ, γ, final_time)
    ω[T] = weights(α, t)
end

relative_error = abs.(ω[Float64][Nₜ] - ω[BigFloat][Nₜ]) ./ω[BigFloat][Nₜ]

α = 0.5
γ = 3.5
final_time = 4.0
t = graded_mesh(Nₜ, γ, final_time)
δ_threshold = 0.1
tol = 1e-16
ω_series = weights(α, t, δ_threshold, tol)
relative_error_series = abs.(ω_series[Nₜ] - ω[BigFloat][Nₜ]) ./ω[BigFloat][Nₜ]

figure(1)
semilogy(1:Nₜ, relative_error, 1:Nₜ, relative_error_series)
legend(("direct formula", "series modification"))
grid(true)
title("Relative rounding error in ωₙⱼ when n = $Nₜ")
xlabel("j")

