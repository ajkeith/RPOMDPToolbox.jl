# some implementations for convenience
# maintained by Zach Sunberg

actions{S}(mdp::MDP{S,Bool}) = (true, false)
actions{S}(mdp::MDP{S,Bool}, s::S, t::Tuple{Bool,Bool}=(true,false)) = (true,false)
actions{S,O}(mdp::POMDP{S,Bool,O}) = (true, false)
actions{S,O}(mdp::POMDP{S,Bool,O}, s::S, t::Tuple{Bool,Bool}=(true,false)) = (true,false)
actions{S,O}(mdp::RPOMDP{S,Bool,O}) = (true, false)
actions{S,O}(mdp::RPOMDP{S,Bool,O}, s::S, t::Tuple{Bool,Bool}=(true,false)) = (true,false)
n_actions{S}(mdp::MDP{S,Bool}) = 2
n_actions{S,O}(mdp::POMDP{S,Bool,O}) = 2
n_actions{S,O}(mdp::RPOMDP{S,Bool,O}) = 2

actions{S}(mdp::MDP{S,Int}, s::S, r::Range) = actions(mdp)

rand(rng::AbstractRNG, t::Tuple{Bool, Bool}) = rand(rng, Bool)
rand(t::Tuple{Bool, Bool}) = rand(Bool)

iterator(s::AbstractVector) = s
iterator(s::Tuple) = s
iterator(r::Range) = r
iterator(g::Base.Generator) = g

states(mdp::MDP{Bool}) = (true, false)
states(mdp::POMDP{Bool}) = (true, false)
states(mdp::RPOMDP{Bool}) = (true, false)
n_states(mdp::MDP{Bool}) = 2
n_states(mdp::POMDP{Bool}) = 2
n_states(mdp::RPOMDP{Bool}) = 2

observations{S,A}(::POMDP{S,A,Bool}) = (true,false)
observations{S,A}(::POMDP{S,A,Bool}, s::S) = (true,false)
observations{S,A}(::RPOMDP{S,A,Bool}) = (true,false)
observations{S,A}(::RPOMDP{S,A,Bool}, s::S) = (true,false)
n_observations{S,A}(::POMDP{S,A,Bool}) = 2
n_observations{S,A}(::RPOMDP{S,A,Bool}) = 2

state_index(mdp::Union{MDP, POMDP, RPOMDP}, s::Int) = s
action_index(mdp::Union{MDP, POMDP, RPOMDP}, a::Int) = a
obs_index(mdp::Union{MDP, POMDP, RPOMDP}, o::Int) = o

state_index(mdp::Union{MDP, POMDP, RPOMDP}, s::Bool) = s+1
action_index(mdp::Union{MDP, POMDP, RPOMDP}, a::Bool) = a+1
obs_index(mdp::Union{MDP, POMDP, RPOMDP}, o::Bool) = o+1
