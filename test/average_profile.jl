using SimpleFiniteElements
import SimpleFiniteElements.Utils: gmsh2pyplot
import SimpleFiniteElements.FEM: assemble_vector!, average_field
using SubdiffusionQMC
import OffsetArrays: OffsetArray
using PyPlot
using Printf
import SimpleFiniteElements.Poisson: ∫∫a_∇u_dot_∇v!, ∫∫c_u_v!, ∫∫f_v!
using LinearAlgebra
import LinearAlgebra: mul!, ldiv!, cholesky!, axpby!, cholesky
import LinearAlgebra.BLAS: scal!
using Statistics
import SubdiffusionQMC: SparseCholeskyFactor
import SubdiffusionQMC.PDE: solve_pcg!
path = joinpath("..", "spatial_domains", "unit_square.geo")
gmodel = GeometryModel(path)
essential_bcs = [("Gamma", 0.0)]