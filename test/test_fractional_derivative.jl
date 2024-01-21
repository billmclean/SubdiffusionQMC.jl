using SubdiffusionQMC
import SpecialFunctions: gamma
import Printf: @printf

u(t) = t
α = 0.75
∂ᵅu(t) = t^(1-α) / gamma(2-α)
Nₜ = 50
γ = 2.0
t = graded_mesh(Nₜ, γ, 2.0)
ω = weights(α, t)

n = Nₜ ÷ 2
I = ( t[n]^(2-α) - t[n-1]^(2-α) ) / gamma(3-α)
U = u.(t)
ΔU = diff(U[0:Nₜ])
Q = ω[n][n] * ΔU[n] 
for j = 1:n-1
    global Q
    Q += ω[n][j] * ΔU[j]
end

@printf("Nₜ = %d\n", Nₜ)
@printf("Integral:   %12.8f\n", I)
@printf("Quadrature: %12.8f\n", Q)
