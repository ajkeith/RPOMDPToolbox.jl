using RPOMDPToolbox
using RPOMDPModels
using RobustValueIteration
using Base.Test
const RPBVI = RobustValueIteration

unc = 0.1
prob = SimpleBaby2RPOMDP(-5.0, -10.0, 0.9, unc)
b0 = DiscreteBelief(prob, [0.5, 0.5])
@test pdf(b0,:hungry) == 0.5
@test pdf(b0,:full) == 0.5

# testing updater initialization
bs = [[b, 1-b] for b in 0.0:0.05:1.0]
solver = RPBVISolver(beliefpoints = bs, max_iterations = 100)
policy = RPBVI.solve(solver, prob)
up = RobustUpdater(prob, policy.alphas)
isd = initial_state_distribution(prob)
b4 = initialize_belief(up, isd)
@test pdf(b4, :hungry) == pdf(isd, :hungry)
@test pdf(b4, :full) == pdf(isd, :full)

# testing update function; if we feed baby, it won't be hungry
a = :feed
o = :crying
b4p = update(up, b4, a, o)
@show b4p.b
@test pdf(b4p,:hungry) ≈ 0.0 atol = 1e-3
@test pdf(b4p,:full) ≈ 1.0 atol = 1e-3

a = :nothing
o = :crying
b4p = update(up, b4, a, o)
@show b4p.b

@test isapprox(pdf(b4p,:hungry), 0.470588, atol=1e-4)
@test isapprox(pdf(b4p,:full), 0.52941, atol=1e-4)
