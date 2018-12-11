# RPOMDPToolbox.jl
This is a fork of the Julia 0.6 version of `POMDPToolbox.jl` that has been edited to be compatible with robust POMDPs in addition to standard POMDPs. It is primarily used for simulation and belief updating.

## Installation
This application is built for Julia 0.6. If not already installed, the application can be cloned using

```julia
Pkg.clone("https://github.com/ajkeith/RPOMDPToolbox.jl")
```

## Usage

See [POMDPToolbox.jl](https://github.com/JuliaPOMDP/POMDPToolbox.jl) for details on usage.

The robust extension to the base fork primarily focuses on belief updates and simulation.

```julia
using RPOMDPToolbox
using RPOMDPModels, RPOMDPs, RobustValueIteration

rpomdp = RockRIPOMDP()
b = [psample(zeros(4), ones(4)) for i = 1:10]
solver = RPBVISolver(beliefpoints = b, max_iterations = 10)
policy = RobustValueIteration.solve(solver, rpomdp)

rng = MersenneTwister(0)
bu = updater(policy)
binit = initial_belief_distribution(rpomdp)
sinit = rand(rng, initial_state_distribution(rpomdp))
sim = RolloutSimulator(max_steps = 100)
simval, simpercent = simulate(sim, rpomdp, policy, bu, binit, sinit)
```
To solve robust POMDP models, see [RobustValueIteration](https://github.com/ajkeith/RobustValueIteration).

## References
The robust POMDP environment is a direct extension of a fork of [POMDPToolbox.jl](https://github.com/JuliaPOMDP/POMDPToolbox.jl) to the robust setting. See [POMDPModelTools.jl](https://github.com/JuliaPOMDP/POMDPModelTools.jl) for the current version of the standard POMDP tools.

If this code is useful to you, please star this package and consider citing the following paper.

Egorov, M., Sunberg, Z. N., Balaban, E., Wheeler, T. A., Gupta, J. K., & Kochenderfer, M. J. (2017). POMDPs.jl: A framework for sequential decision making under uncertainty. Journal of Machine Learning Research, 18(26), 1–5.
