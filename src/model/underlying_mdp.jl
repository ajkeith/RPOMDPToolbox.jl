# provide a structure to extract the underlying MDP of a RPOMDP

struct UnderlyingMDP{P <: RPOMDP, S, A} <: MDP{S, A}
    rpomdp::P
end

function UnderlyingMDP{S, A, O}(rpomdp::RPOMDP{S, A, O})
    P = typeof(rpomdp)
    return UnderlyingMDP{P, S, A}(rpomdp)
end

RPOMDPs.transition{P, S, A}(mdp::UnderlyingMDP{P, S, A}, s::S, a::A) = transition(mdp.rpomdp, s, a)
RPOMDPs.initial_state_distribution(mdp::UnderlyingMDP) = initial_state_distribution(mdp.rpomdp)
RPOMDPs.generate_s(mdp::UnderlyingMDP, s, a, rng::AbstractRNG) = generate_s(mdp.rpomdp, s, a, rng)
RPOMDPs.generate_sr(mdp::UnderlyingMDP, s, a, rng::AbstractRNG) = generate_sr(mdp.rpomdp, s, a, rng)
RPOMDPs.initial_state(mdp::UnderlyingMDP, rng::AbstractRNG) = initial_state(mdp.rpomdp, rng)
RPOMDPs.states(mdp::UnderlyingMDP) = states(mdp.rpomdp)
RPOMDPs.actions(mdp::UnderlyingMDP) = actions(mdp.rpomdp)
RPOMDPs.reward{P, S, A}(mdp::UnderlyingMDP{P, S, A}, s::S, a::A) = reward(mdp.rpomdp, s, a)
RPOMDPs.reward{P, S, A}(mdp::UnderlyingMDP{P, S, A}, s::S, a::A, sp::S) = reward(mdp.rpomdp, s, a, sp)
RPOMDPs.isterminal{P, S, A}(mdp ::UnderlyingMDP{P, S, A}, s::S) = isterminal(mdp.rpomdp, s)
RPOMDPs.discount(mdp::UnderlyingMDP) = discount(mdp.rpomdp)
RPOMDPs.n_actions(mdp::UnderlyingMDP) = n_actions(mdp.rpomdp)
RPOMDPs.n_states(mdp::UnderlyingMDP) = n_states(mdp.rpomdp)
RPOMDPs.state_index{P, S, A}(mdp::UnderlyingMDP{P, S, A}, s::S) = state_index(mdp.rpomdp, s)
RPOMDPs.state_index{P, A}(mdp::UnderlyingMDP{P, Int, A}, s::Int) = state_index(mdp.rpomdp, s) # fix ambiguity with src/convenience
RPOMDPs.state_index{P, A}(mdp::UnderlyingMDP{P, Bool, A}, s::Bool) = state_index(mdp.rpomdp, s)
RPOMDPs.action_index{P, S, A}(mdp::UnderlyingMDP{P, S, A}, a::A) = action_index(mdp.rpomdp, a)
RPOMDPs.action_index{P, S}(mdp::UnderlyingMDP{P,S, Int}, a::Int) = action_index(mdp.rpomdp, a)
RPOMDPs.action_index{P, S}(mdp::UnderlyingMDP{P,S, Bool}, a::Bool) = action_index(mdp.rpomdp, a)
