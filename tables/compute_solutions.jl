using JLD2

include("input_data.jl")

nrows = 3
ref_row = nrows + 2
for row = 1:ref_row
    N = Nvals[row]
    soln_file = "soln_$N.jld2"
    if isfile(soln_file)
        @printf("File %s already exists!\n", soln_file)
    else
        @printf("\t N = %d ...", N)
        start = time()
        Φ, Φ_det, pcg_its = simulations!(pts[row], solver, κ₀_vals, 
                                         f_homogeneous, get_load_vector!, 
                                         pstore, estore, dstore, u₀_bent)
        elapsed = time() - start
        @printf(" in %d seconds.\n", elapsed)
        EL = sum(Φ, dims=2) / N
        EL_det = sum(Φ_det, dims=2) / N
        jldsave(soln_file; EL, EL_det, pcg_its, elapsed)
    end
end

