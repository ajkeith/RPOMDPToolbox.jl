# Provides a structure for turning a mdp into a rpomdp where observations are the states of the original mdp

struct FullyObservablePOMDP{S, A} <: RPOMDP{S,A,S}
    mdp::MDP{S, A}
end

# observations are the state of the MDP
# The observation distribution is modeled by a SparseCat distribution with only one element
RPOMDPs.observations(rpomdp::FullyObservablePOMDP) = states(rpomdp.mdp)
RPOMDPs.n_observations(rpomdp::FullyObservablePOMDP) = n_states(rpomdp.mdp)
RPOMDPs.obs_index{S, A}(rpomdp::FullyObservablePOMDP{S, A}, o::S) = state_index(rpomdp.mdp, o)

RPOMDPs.convert_o(T::Type{V}, o, rpomdp::FullyObservablePOMDP) where {V<:AbstractArray} = convert_s(T, s, rpomdp.mdp)
RPOMDPs.convert_o(T::Type{S}, vec::V, rpomdp::FullyObservablePOMDP) where {S,V<:AbstractArray} = convert_s(T, vec, rpomdp.mdp)


function RPOMDPs.generate_o(rpomdp::FullyObservablePOMDP, s, a, rng::AbstractRNG)
    return s
end

function RPOMDPs.observation(rpomdp::FullyObservablePOMDP, s, a)
    return SparseCat((s,), (1.,))
end

function RPOMDPs.observation(rpomdp::FullyObservablePOMDP, s, a, sp)
    return SparseCat((sp,), (1.,))
end

RPOMDPs.isterminal_obs{S,A}(problem::FullyObservablePOMDP{S,A}, o::S) = isterminal(rpomdp.mdp, o)

# inherit other function from the MDP type

RPOMDPs.states(rpomdp::FullyObservablePOMDP) = states(rpomdp.mdp)
RPOMDPs.actions(rpomdp::FullyObservablePOMDP) = actions(rpomdp.mdp)
RPOMDPs.transition{S,A}(rpomdp::FullyObservablePOMDP{S,A}, s::S, a::A) = transition(rpomdp.mdp, s, a)
RPOMDPs.initial_state_distribution(rpomdp::FullyObservablePOMDP) = initial_state_distribution(rpomdp.mdp)
RPOMDPs.initial_state(rpomdp::FullyObservablePOMDP, rng::AbstractRNG) = initial_state(rpomdp.mdp, rng)
RPOMDPs.generate_s(rpomdp::FullyObservablePOMDP, s, a, rng::AbstractRNG) = generate_s(rpomdp.mdp, s, a, rng)
RPOMDPs.generate_sr(rpomdp::FullyObservablePOMDP, s, a, rng::AbstractRNG) = generate_sr(rpomdp.mdp, s, a, rng)
RPOMDPs.reward{S,A}(rpomdp::FullyObservablePOMDP{S, A}, s::S, a::A) = reward(rpomdp.mdp, s, a)
RPOMDPs.isterminal(rpomdp::FullyObservablePOMDP, s) = isterminal(rpomdp.mdp, s)
RPOMDPs.discount(rpomdp::FullyObservablePOMDP) = discount(rpomdp.mdp)
RPOMDPs.n_states(rpomdp::FullyObservablePOMDP) = n_states(rpomdp.mdp)
RPOMDPs.n_actions(rpomdp::FullyObservablePOMDP) = n_actions(rpomdp.mdp)
RPOMDPs.state_index{S,A}(rpomdp::FullyObservablePOMDP{S,A}, s::S) = state_index(rpomdp.mdp, s)
RPOMDPs.action_index{S, A}(rpomdp::FullyObservablePOMDP{S, A}, a::A) = action_index(rpomdp.mdp, a)
RPOMDPs.convert_s(T::Type{V}, s, rpomdp::FullyObservablePOMDP) where V<:AbstractArray = convert_s(T, s, rpomdp.mdp)
RPOMDPs.convert_s(T::Type{S}, vec::V, rpomdp::FullyObservablePOMDP) where {S,V<:AbstractArray} = convert_s(T, vec, rpomdp.mdp)
RPOMDPs.convert_a(T::Type{V}, a, rpomdp::FullyObservablePOMDP) where V<:AbstractArray = convert_a(T, a, rpomdp.mdp)
RPOMDPs.convert_a(T::Type{A}, vec::V, rpomdp::FullyObservablePOMDP) where {A,V<:AbstractArray} = convert_a(T, vec, rpomdp.mdp)
