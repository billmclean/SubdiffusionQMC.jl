using LinearAlgebra
import LinearAlgebra.BLAS: axpby!
using BenchmarkTools

N = 100
M = 100
x = Vector{Vector{Float64}}(undef, N)
A = Vector{Matrix{Float64}}(undef, N)
for i in eachindex(x)
    x[i] = randn(M)
    A[i] = randn(M, M)
end
α = randn(N)
y = zeros(M)

function sum_version_A!(y, α, A, x)
    fill!(y, 0.0)
    for i = 1:lastindex(x)
	y += α[i] * (A[i] * x[i])
    end
end

function sum_version_B!(y, α, A, x)
    fill!(y, 0.0)
    for i = 1:lastindex(x)
	y .+= α[i] * (A[i] * x[i])
    end
end

function sum_version_B!(y, α, A, x)
    fill!(y, 0.0)
    for i = 1:lastindex(x)
	y .+= α[i] * (A[i] * x[i])
    end
end

function sum_version_C!(y, α, A, x)
    fill!(y, 0.0)
    for i = 1:lastindex(x)
	y .+= α[i] .* (A[i] * x[i])
    end
end

function sum_version_D!(y, α, A, x)
    fill!(y, 0.0)
    for i = 1:lastindex(x)
	mul!(y, A[i], x[i], α[i], 1.0)
    end
end

println("Version A:")
@btime sum_version_A!($y, $α, $A, $x)

println("Version B:")
@btime sum_version_B!($y, $α, $A, $x)

println("Version C:")
@btime sum_version_C!($y, $α, $A, $x)

println("Version D:")
@btime sum_version_D!($y, $α, $A, $x)
