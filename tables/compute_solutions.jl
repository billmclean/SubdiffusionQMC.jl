using JLD2
using OffsetArrays

include("input_data.jl")

nrows = 4
ref_row = nrows + 2

for row = 1:ref_row
    local N, soln_file, EL, VL
    N = Nvals[row]
    soln_file = "soln_$N.jld2"
    if isfile(soln_file)
        @printf("File %s already exists!\n", soln_file)
    else
        @printf("\t N = %d ...", N)
        start = time()
        Φ, Φ_det, pcg_its = simulations!(pts[row], solver, κ₀_vals, 
                                         f, get_load_vector!, 
                                         pstore, estore, dstore, u₀_bent)
        elapsed = time() - start
        @printf(" in %d seconds.\n", elapsed)
        EL = OffsetVector{Float64}(undef, 0:Nₜ) # Expected Value
        VL = similar(EL) # Variance
        for n in eachindex(t)
            s = 0.0
            for l = 1:N
                s += Φ[n,l]
            end
            EL[n] = s / N
            s = 0.0
            for l = 1:N
                s += (Φ[n,l] - EL[n])^2
            end
            VL[n] = s / N
        end
        jldsave(soln_file; EL, VL, pcg_its, elapsed)
    end
end

