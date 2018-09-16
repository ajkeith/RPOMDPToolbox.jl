module BeliefUpdaters

using RPOMDPs
import RPOMDPs: Updater, update, initialize_belief, pdf, mode, updater, iterator
import Base: rand, rand!, mean, ==

using RPOMDPToolbox.ordered_states
using StatsBase

export
    VoidUpdater
include("void.jl")

export
    DiscreteBelief,
    DiscreteUpdater,
    uniform_belief,
    product                     # Remove because deprecated
include("discrete.jl")


export
    PreviousObservationUpdater,
    FastPreviousObservationUpdater,
    PrimedPreviousObservationUpdater

include("previous_observation.jl")

end
