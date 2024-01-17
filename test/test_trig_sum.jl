using SubdiffusionQMC
import SubdiffusionQMC.RandomDiffusivity: sin_sum!
import FFTW: plan_r2r, RODFT00
import Printf: @printf

function slow_sin_sum(x::Float64, a::Vec64)
    Σ = 0.0
    for k in eachindex(a)
	Σ += a[k] * sinpi(k * x)
    end
    return Σ
end

z = 7
resolution = 16
a = zeros(resolution)
a[1:z] .= randn(z)
x = range(0, 1, length=resolution+2)
plan = plan_r2r(a, RODFT00)

slow_S = Float64[ slow_sin_sum(xⱼ, a) for xⱼ in x ]
fast_S = similar(slow_S)
sin_sum!(fast_S, a, plan)
@printf("Max error computing sin sum = %0.3e\n", maximum(abs, fast_S-slow_S))
