using PyPlot, JLD2, SubdiffusionQMC

include(joinpath("..", "tables", "input_data.jl"))
N = Nvals[6]
soln_file = joinpath("..", "tables", "soln_$N.jld2")
if isfile(soln_file)
    println("Loading solution from $soln_file.\n")
    EL = load(soln_file, "EL")
    VL = load(soln_file, "VL")
else
    error("File %s not found.")
end

σ = sqrt.(VL)
figure(2)
plot(t, EL, t, EL+3σ, ":", t, EL-3σ, ":")
grid(true)
xlabel(L"$t$")
#title(L"The expected value of $\mathcal{L}(u(t))$")
legend((L"$E(t)$", L"$E(t)+3\sigma(t)$", L"$E(t)-3\sigma(t)$"))
axis([-0.1, t[Nₜ]+0.1, 0, 1])
savefig("expected_value.pdf")
