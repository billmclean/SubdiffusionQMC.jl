using JLD2
using SubdiffusionQMC
using Printf

include("input_data.jl")
nrows = 4

function print_table(titlestring::String, nrows::Int64; latex=true)
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
    EL_error_T = similar(EL)
    EL_error_L2 = similar(EL)
    elapsed = similar(EL)
    @printf("%6s  %14s  %10s  %8s  %10s  %8s  %8s\n\n",
            "N", "E(T)", "T error", "rate", "L2 error", "rate", "secs")
    if latex
        row1_fmt = Printf.Format(
                   "%6d& %14.10f& %10.2e& %8s& %10.2e& %8s\\\\ %%%8.3f\n")
        rowk_fmt = Printf.Format(
                   "%6d& %14.10f& %10.2e& %8.3f& %10.2e& %8.3f\\\\ %%%8.3f\n")
    else
        row1_fmt = Printf.Format(
                   "%6d  %14.10f  %10.3e  %8s  %10.3e  %8s  %8.3f\n")
        rowk_fmt = Printf.Format(
                   "%6d  %14.10f  %10.3e  %8.3f  %10.3e  %8.3f  %8.3f\n")
    end
    for k = 1:nrows
        N = Nvals[k]
        soln_file = "soln_$N.jld2"
        EL_k = load(soln_file, "EL")
        EL[k] = EL_k[Nₜ]
        g = EL_k - EL_ref
        EL_error_T[k] = g[Nₜ]
        EL_error_L2[k] = L2_norm(g, t)
        elapsed[k] = load(soln_file, "elapsed")
        if k == 1
            Printf.format(stdout, row1_fmt, Nvals[k], 
                          EL[k], EL_error_T[k], "", 
                          EL_error_L2[k], "", elapsed[k])
        else
            rate_T = log2(abs(EL_error_T[k-1]/EL_error_T[k]))
            rate_L2 = log2(abs(EL_error_L2[k-1]/EL_error_L2[k]))
            Printf.format(stdout, rowk_fmt, Nvals[k], EL[k], 
                          EL_error_T[k], rate_T, 
                          EL_error_L2[k], rate_L2, elapsed[k])
        end
    end
    @printf("%6d  %14.10f\n", Nref, EL_ref[Nₜ])
end      

print_table("Table 1.", nrows, latex=true)
