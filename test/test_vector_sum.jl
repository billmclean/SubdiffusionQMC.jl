using LinearAlgebra
using BenchmarkTools

N = 100
M = 100
x = Vector{Vector{Float64}}(undef, N)
for i in eachindex(x)
    x[i] = randn(M)
end
α = randn(N)
y = zeros(M)

function sum_version_A!(y, α, x)
    fill!(y, 0.0)
    for i = 1:lastindex(x)
	y += α[i] * x[i]
    end
end

function sum_version_B!(y, α, x)
    fill!(y, 0.0)
    for i = 1:lastindex(x)
	y .+= α[i] * x[i]
    end
end

function sum_version_C!(y, α, x)
    fill!(y, 0.0)
    for i = 1:lastindex(x)
	y .+= α[i] .* x[i]
    end
end

function sum_version_D!(y, α, x)
    fill!(y, 0.0)
    for i = 1:lastindex(x)
	axpy!(α[i], x[i], y)
    end
end

println("Version A:")
@btime sum_version_A!($y, $α, $x)

println("Version B:")
@btime sum_version_B!($y, $α, $x)

println("Version C:")
@btime sum_version_C!($y, $α, $x)

println("Version D:")
@btime sum_version_D!($y, $α, $x)
