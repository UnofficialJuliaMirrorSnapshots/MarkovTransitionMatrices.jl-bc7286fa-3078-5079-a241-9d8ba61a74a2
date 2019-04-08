using MarkovTransitionMatrices

# state grids
grid_x1 = 0.0:1.0:10.0
grid_x2 = -1.5:0.5:15.0

# probabilities must all be greater than minp
minp = 1e-8

# Correlated random walk
μ(s) = [s...]
Σ(s) = [1.0 0.5; 0.5 1.0]
P = markov_transition(μ, Σ, minp, grid_x1, grid_x2)

# Markov-switching process

# regime transition matrix (NOT tranpsosed! - rows sum to 1.0)
πswitch = [.9 .1; .4 .6]

μswitch(r::Int, s) = r==1 ? [s...] : [s...] .+ ones(2)
Σswitch(r::Int, s) = r==1 ? Matrix(1.0I,2,2) : [1.0 0.5; 0.5 1.0]
Pswitch = markovswitching_transition(μswitch, Σswitch, πswitch, 1e-8, grid_x1, grid_x2)
