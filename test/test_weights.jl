using SubdiffusionQMC
using PyPlot

Nₜ = 100
ω = Dict()
for T in [Float64, BigFloat]
    local α, γ, t
    α = parse(T, "0.5")
    γ = parse(T, "3.5")
    final_time = parse(T, "4.0")
    t = graded_mesh(Nₜ, γ, final_time)
    ω[T] = weights(α, t)
end

relative_error = abs.(ω[Float64][Nₜ] - ω[BigFloat][Nₜ]) ./ω[BigFloat][Nₜ]
#relative_error = max.(relative_error, eps(Float64))

figure(1)
semilogy(relative_error)
grid(true)
title("Relative rounding error in ωₙⱼ when n = $Nₜ")
xlabel("j")
