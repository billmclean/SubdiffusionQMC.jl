module FEM2D

import ..PDEStore, ..Vec64, ..Mat64

using SimpleFiniteElements
import SimpleFiniteElements.Poisson: ∫∫a_∇u_dot_∇v!, ∫∫c_u_v!
import LinearAlgebra: cholesky

function PDEStore(κ₀::Function, dof::DegreesOfFreedom,
                  solver::Symbol, pcg_tol::Float64, pcg_maxiterations::Integer)
    bilinear_forms_A = Dict("Omega" => (∫∫a_∇u_dot_∇v!, κ₀))
    bilinear_forms_M = Dict("Omega" => (∫∫c_u_v!, 1.0))
    A_free, A_fix = assemble_matrix(dof, bilinear_forms_A)
    M_free, M_fix = assemble_matrix(dof, bilinear_forms_M)
    P = cholesky(M_free + A_free)
    b = Vec64(undef, dof.num_free)
    wkspace = Mat64(undef, num_free, 4)
    u = similar(b)
    PDEStore(κ₀, dof, b, solver, P, wkspace, u, pcg_tol, pcg_maxiterations)
end

end # module
