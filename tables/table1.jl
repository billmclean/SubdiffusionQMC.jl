using JLD2

include("input_data.jl")

@printf("\nTable 1.\n")

nrows = 3
ref_row = nrows + 2
Nref = Nvals[ref_row]
soln_file = "soln_$Nref.jld2"
if isfile(soln_file)
    @printf("Loading reference solution (N = %d) from %s.\n", 
            Nref, soln_file)
    EL_ref = load(soln_file, "EL")
else
    error("No file $refsoln")
end

EL = zeros(nrows)
EL_error = similar(EL)
elapsed = similar(EL)
for k = 1:nrows
    local soln_file
    N = Nvals[k]
    soln_file = "soln_$N.jld2"
    EL_k = load(soln_file, "EL")
    EL[k] = EL_k[Nₜ]
    elapsed[k] = load(soln_file, "elapsed")
    EL_error[k] = EL[k] - EL_ref
    if k == 1
        @printf("%6d  %14.10f  %10.3e  %8s  %8.3f\n",
                Nvals[k], EL[k], EL_error[k], "", elapsed[k])
    else
        rate = log2(abs(EL_error[k-1]/EL_error[k]))
        @printf("%6d  %14.10f  %10.3e  %8.3f  %8.3f\n",
                Nvals[k], EL[k], EL_error[k], rate, elapsed[k])
    end
end      
