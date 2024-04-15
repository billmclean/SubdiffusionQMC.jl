# uniform mesh
T = 1.0 
Nₜ = 50
t = collect(range(0, T, Nₜ+1))
t = OVec64(t, 0:Nₜ)