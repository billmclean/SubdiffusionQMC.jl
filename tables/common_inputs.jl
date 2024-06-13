using SimpleFiniteElements
using SubdiffusionQMC
import SimpleFiniteElements.Utils: gmsh2pyplot
import SimpleFiniteElements.FEM: assemble_vector!
import SimpleFiniteElements.Poisson: ∫∫a_∇u_dot_∇v!, ∫∫c_u_v!, ∫∫f_v!
using Printf
using JLD2

path = joinpath("..", "spatial_domains", "unit_square.geo")
gmodel = GeometryModel(path)

hmax = 0.2

solver = :pcg  # :direct or :pcg
pcg_tol = 1e-10
pcg_maxits = 100

mesh = FEMesh(gmodel, hmax)
essential_bcs = [("Gamma", 0.0)]
dof = DegreesOfFreedom(mesh, essential_bcs)

h_finest = max_elt_diameter(mesh)
h_string = @sprintf("%0.4f", h_finest)

if exno == 2
    n = 22 
elseif exno == 3
    n = 15
    use_fft = false 
else
    error("Unknown example number exno = $exno.")
end

T = 1.0
Nₜ = 20
α = 0.5
γ = 2 / α
t = graded_mesh(Nₜ, γ, T)

f_homogeneous(x, y, t) = 0.0
κ₀(x, y) = 0.1 * (2 + x * y) 
min_κ₀ = κ₀(0.0, 0.0)
pstore = PDEStore_integrand(κ₀, dof, solver, pcg_tol, pcg_maxits)
x, y, triangles = gmsh2pyplot(dof)
num_free = pstore.dof.num_free

u₀_bent(x, y) = 5 * (x^2 * (1 - x) + y^2 * (1 - y))

r = 2
tol = 1e-8
Δx = 0.5
Nₕ = dof.num_free
estore = ExponentialSumStore(t, Nₕ, α, r, tol, Δx)

p = 0.5
resolution = (256, 256)
idx = double_indices(n)
z = lastindex(idx)
dstore = DiffusivityStore2D(idx, z, p, resolution, min_κ₀)

qmc_path = joinpath("..", "qmc_points", "SPOD_dim256.jld2")
Nvals, pts = SPOD_points(z, qmc_path)

(N₁, N₂) = resolution

function get_load_vector!(F::Vec64, t::Float64, pstore::PDEStore_integrand, f::Function)
    fill!(F, 0.0)
    assemble_vector!(F, "Omega", [(∫∫f_v!, (x, y) -> f(x, y, t))], dof, 1)
end

(N₁, N₂) = resolution
x₁_vals = range(0, 1, N₁)
x₂_vals = range(0, 1, N₂)
κ₀_vals = Float64[ κ₀(x, y) for x in x₁_vals, y in x₂_vals ]

msg1 = """
Example $exno.  Solving IBVP with non-conforming finite elements.
Solver is $solver with tol = $pcg_tol.
Employing $(Threads.nthreads()) threads.
SPOD QMC points with z = $z.
Finest FEM mesh has $num_free degrees of freedom and h = $h_string."""

println(msg1)

if exno == 3 && !use_fft
    msg2 = """
    Evaluating κ directly (no FFT).
    """
else
    msg2 = """
    Using $N₁ x $N₂ grid to interpolate κ.
    """
end

println(msg2)

function create_tables(exno::Int64; nrows=4)
    @printf("\nExample %d:\n", exno)
    ref_row = nrows + 2
    Nref = Nvals[ref_row]
	refsoln_file = "reference_soln_ex$(exno)_$(Nref)_nonconf.jld2"
    if isfile(refsoln_file)
        @printf("Loading reference solution (N = %d) from %s.\n", 
                Nref, refsoln_file)
        L_ref = load(refsoln_file, "L_ref")
        L_det = load(refsoln_file, "L_det")
        pcg_its = load(refsoln_file, "pcg_its")
    else
        @printf("Computing reference solution (N = %d) ...", Nref)
        start = time()
        if exno == 2
	    Φ, Φ_det, pcg_its = simulations!(
                 pts[ref_row], solver, κ₀_vals, f_homogeneous, 
                 get_load_vector!, pstore, estore, dstore, u₀_bent)
	    elseif exno == 3
	      if use_fft
	        Φ, Φ_det, pcg_its = simulations!(
                 pts[ref_row], solver, κ₀_vals, f_homogeneous, 
                 get_load_vector!, pstore, estore, dstore, u₀_bent)
	      else
	        Φ, Φ_det, pcg_its = slow_simulations!(
                 pts[ref_row], solver, κ₀, f_homogeneous, 
                 get_load_vector!, pstore, estore, dstore, u₀_bent)
	      end
        end
        elapsed_ref = time() - start
        @printf(" in %d seconds.\n", elapsed_ref)
        L_ref = sum(Φ) / Nref
        L_det = sum(Φ_det) / Nref
        jldsave(refsoln_file; L_ref, L_det, pcg_its)
    end
    @printf("\n%6s  %14s  %10s  %8s  %8s\n\n",
            "N", "L", "error", "rate", "seconds")
    L = zeros(nrows)
    L_error = similar(L)
    elapsed = similar(L)
    for k = 1:nrows    
        start = time()
        if exno == 2
	    Φ, _, _ = simulations!(pts[k], solver, κ₀_vals, f_homogeneous, 
                                 get_load_vector!, pstore, estore, dstore, u₀_bent)
	elseif exno == 3
	    if use_fft
	        Φ, _, _ = simulations!(pts[k], solver, κ₀_vals, f_homogeneous, 
                                     get_load_vector!, pstore, estore, dstore, u₀_bent)
	    else
	        Φ, _, _ = slow_simulations!(pts[k], solver, κ₀, f_homogeneous, 
                                         get_load_vector!, pstore, estore, dstore, u₀_bent)
	    end
        end
        elapsed[k] = time() - start
        L[k] = sum(Φ) / Nvals[k]
        L_error[k] = L[k] - L_ref
        if k == 1
             @printf("%6d  %14.10f  %10.3e  %8s  %8.3f\n",
                     Nvals[k], L[k], L_error[k], "", elapsed[k])
        else
	    rate = log2(abs(L_error[k-1]/L_error[k]))
            @printf("%6d  %14.10f  %10.3e  %8.3f  %8.3f\n",
                    Nvals[k], L[k], L_error[k], rate, elapsed[k])
        end
    end
    @printf("\n%6d  %14.10f  %s\n", Nref, L_ref, "Reference value")
    @printf("\nLaTeX-ready version:\n%6s  %14s  %10s  %8s\n\n",
            "N", "L", "error", "rate")
    for k = 1:nrows
        if k == 1
             @printf("%6d& %14.10f& %10.2e& %8s\\\\\n",
                     Nvals[k], L[k], L_error[k], "")
        else
	    rate = log2(abs(L_error[k-1]/L_error[k]))
            @printf("%6d& %14.10f& %10.2e& %8.3f\\\\\n",
                    Nvals[k], L[k], L_error[k], rate)
        end
    end
    return L_error, pcg_its
end




        
    
