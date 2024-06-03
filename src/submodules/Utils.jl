module Utils

import ..Vec64, ..Mat64
using ArgCheck
using LinearAlgebra
using JLD2

import ..SPOD_points, ..pcg!, ..cg!

function SPOD_points(s::Integer, path::String)
    D = load(path)
    if s > 256
	error("Dimension s can be at most 256")
    end
    Nvals = D["Nvals"]
    pts = Vector{Mat64}(undef, length(Nvals))
    for k in eachindex(Nvals)
	N = Nvals[k]
        name = "SPOD_N$(N)_dim256"
	pts[k] = D[name][1:s,:] .- 1/2
    end
    return Nvals, pts
end

function pcg!(x::AbstractVector{T}, A::AbstractMatrix{T}, b::Vector{T}, P, tol::T,
              wkspace::Matrix{T}) where T <: AbstractFloat
    n = lastindex(b)
    @argcheck size(A) == (n, n)
    @argcheck size(b) == (n,)
    @argcheck size(wkspace) == (n, 4)
    p  = view(wkspace, :, 1)
    r  = view(wkspace, :, 2)
    Ap = view(wkspace, :, 3)
    w  = view(wkspace, :, 4)
    r .= b - A*x
    if norm(r) < tol * norm(b)
        return 0 # zero iterations
    end
    w .= P \ r
    p .= w
    for j = 0:n-1
        mul!(Ap, A, p) # Ap = A * p
        w_dot_r = dot(w, r)
        α = w_dot_r / dot(p, Ap)
        r .-= α * Ap
        x .+= α * p
        if norm(r) < tol * norm(b)
            return j + 1
        end
        w .= P \ r
        β = dot(w, r) / w_dot_r
        p .= w + β * p
    end
    @warn "PCG failed to converge"
    return n
end

function cg!(x::Vector{T}, A::AbstractMatrix{T}, b::Vector{T}, tol::T,
        wkspace::Matrix{T}) where T <: AbstractFloat
    n = lastindex(b)
    @argcheck size(A) == (n, n)
    @argcheck size(b) == (n,)
    @argcheck size(wkspace) == (n, 4)
    p  = view(wkspace, :, 1)
    r  = view(wkspace, :, 2)
    Ap = view(wkspace, :, 3)
    r .= b - A*x
    p .= r
    normb = norm(b)
    if norm(r) < tol * normb
        return 0 # zero iterations
    end
    for j = 0:n-1
    mul!(Ap, A, p) # Ap = A * p
    γ = dot(r, r)
    α = γ / dot(p, Ap)
    r .-= α * Ap
    x .+= α * p
    if norm(r) < tol * normb
        return  j
    end
    β = dot(r, r) / γ
    p .= r + β * p
    end
    @warn "CG failed to converge"
    return  j+1
end

end # module