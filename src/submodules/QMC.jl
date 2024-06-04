module QMC

import ..DiffusivityStore1D, ..DiffusivityStore2D, ..ExponentialSumStore, 
       ..IdxPair, ..Vec64, ..Mat64, ..double_indices, ..interpolate_κ!, ..AVec64
import ..PDEStore_integrand, ..integrand_init!, ..integrand!
using LinearAlgebra

import ..simulations!


function simulations!(pts::Mat64, solver, κ₀_vals::Mat64, f::Function, 
                      get_load_vector!::Function,
                      pstore::PDEStore_integrand, estore::ExponentialSumStore, 
                      dstore::DiffusivityStore2D, u₀::Function)
    Φ_det = integrand_init!(estore, pstore, f, get_load_vector!, u₀)
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    s, N = size(pts)
    chunks = collect(Iterators.partition(1:N, N ÷ Threads.nthreads()))
    Φ = zeros(N)
    Threads.@threads for chunk in chunks
#    for chunk in chunks
    pstore_local = deepcopy(pstore)
    dstore_local = deepcopy(dstore)
    estore_local = deepcopy(estore)
    for l in chunk
#    for l in 1:N
        y_vals = view(pts, :, l)
        Φ[l] = integrand!(y_vals, κ₀_vals, 
                          estore_local, pstore_local, dstore_local, solver, 
                          f, get_load_vector!, u₀)
    end
    end
    BLAS.set_num_threads(blas_threads)
    return Φ, Φ_det
end

end #module