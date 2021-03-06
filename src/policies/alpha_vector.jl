######################################################################
# alpha_vector.jl
#
# implements policy that is a set of alpha vectors
######################################################################

struct AlphaVectorPolicy{P<:Union{POMDP,IPOMDP,RPOMDP,RIPOMDP}, A} <: Policy
    pomdp::P
    alphas::Vector{Vector{Float64}}
    action_map::Vector{A}
end
function AlphaVectorPolicy(pomdp::Union{POMDP,IPOMDP,RPOMDP,RIPOMDP}, alphas)
    AlphaVectorPolicy(pomdp, alphas, ordered_actions(pomdp))
end
# assumes alphas is |S| x |A|
function AlphaVectorPolicy(p::Union{POMDP,IPOMDP,RPOMDP,RIPOMDP}, alphas::Matrix{Float64}, action_map)
    # turn alphas into vector of vectors
    num_actions = size(alphas, 2)
    alpha_vecs = Vector{Float64}[]
    for i = 1:num_actions
        push!(alpha_vecs, vec(alphas[:,i]))
    end

    AlphaVectorPolicy(p, alpha_vecs, action_map)
end



updater(p::AlphaVectorPolicy) = DiscreteUpdater(p.pomdp, p.alphas)

value(p::AlphaVectorPolicy, b::DiscreteBelief) = value(p, b.b)
function value(p::AlphaVectorPolicy, b::Vector{Float64})
    maximum(dot(b,a) for a in p.alphas)
end

function action(p::AlphaVectorPolicy, b::DiscreteBelief)
    num_vectors = length(p.alphas)
    best_idx = 1
    max_value = -Inf
    for i = 1:num_vectors
        temp_value = dot(b.b, p.alphas[i])
        if temp_value > max_value
            max_value = temp_value
            best_idx = i
        end
    end
    return p.action_map[best_idx]
end

function Base.push!(p::AlphaVectorPolicy, alpha::Vector{Float64}, a)
    push!(p.alphas, alpha)
    push!(p.action_map, a)
end
