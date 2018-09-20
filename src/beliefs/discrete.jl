# Goals: minimize calls to ordered_states (allocates memory)

# needs pomdp for state_index in pdf(b, s)
# needs list of ordered_states for rand(b)

# TO DO: P <: Union{POMDP,IPOMDP,RPOMDP,RIPOMDP} ?
"""
    DiscreteBelief
A belief specified by a probability vector.
Normalization of `b` is NOT enforced at all times, but the `DiscreteBeleif(pomdp, b)` constructor will warn, and `update(...)` always returns a belief with normalized `b`.
"""
struct DiscreteBelief{P, S}
    pomdp::P
    state_list::Vector{S}       # vector of ordered states
    b::Vector{Float64}
end

function DiscreteBelief(prob::Union{POMDP,IPOMDP,RPOMDP,RIPOMDP}, state_list::AbstractVector, b::AbstractVector{Float64}, check::Bool)
    if check
        if !isapprox(sum(b), 1.0, atol=0.001)
            warn("""
                 b in DiscreteBelief(pomdp, b) does not sum to 1.
                 To suppress this warning use `DiscreteBelief(pomdp, b, check=false)`
                 """)
        end
        if !all(0.0 <= p <= 1.0 for p in b)
            warn("""
                 b in DiscreteBelief(pomdp, b) contains entries outside [0,1].
                 To suppress this warning use `DiscreteBelief(pomdp, b, check=false)`
                 """)
        end
    end
    return DiscreteBelief(prob, state_list, b)
end

function DiscreteBelief(prob::Union{POMDP,IPOMDP,RPOMDP,RIPOMDP}, b::Vector{Float64}; check::Bool=true)
    return DiscreteBelief(prob, ordered_states(prob), b, check)
end

function DiscreteBelief(prob::Union{POMDP,IPOMDP,RPOMDP,RIPOMDP}, b; check::Bool=true)
    # convert b to a vector representation
    state_list = ordered_states(prob)
    bv = Vector{Float64}(n_states(prob))
    for (i, s) in enumerate(state_list)
        bv[i] = pdf(b, s)
    end
    return DiscreteBelief(prob, state_list, bv, check)
end


"""
Return a DiscreteBelief with equal probability for each state.
"""
function uniform_belief(prob::Union{POMDP,IPOMDP,RPOMDP,RIPOMDP})
    state_list = ordered_states(prob)
    ns = length(state_list)
    return DiscreteBelief(prob, state_list, ones(ns) / ns)
end

pdf(b::DiscreteBelief, s) = b.b[state_index(b.pomdp, s)]

function rand(rng::AbstractRNG, b::DiscreteBelief)
    i = sample(rng, Weights(b.b))
    return b.state_list[i]
end

function Base.fill!(b::DiscreteBelief, x::Float64)
    fill!(b.b, x)
    return b
end

Base.length(b::DiscreteBelief) = length(b.b)

iterator(b::DiscreteBelief) = b.state_list

==(b1::DiscreteBelief, b2::DiscreteBelief) = b1.state_list == b2.state_list && b1.b == b2.b

Base.hash(b::DiscreteBelief, h::UInt) = hash(b.b, hash(b.state_list, h))

# TO DO: P <: Union{POMDP,IPOMDP,RPOMDP,RIPOMDP} ?
mutable struct DiscreteUpdater{P} <: Updater
    pomdp::P
end

mutable struct RobustUpdater{R} <: Updater
    rpomdp::R
    alphas::Vector{Vector{Float64}}
end
DiscreteUpdater(pomdp::Union{POMDP,IPOMDP}, alphas::Vector{Vector{Float64}}) = DiscreteUpdater(pomdp)
DiscreteUpdater(rpomdp::Union{RPOMDP,RIPOMDP}, alphas::Vector{Vector{Float64}}) = RobustUpdater(rpomdp, alphas)

uniform_belief(up::DiscreteUpdater) = uniform_belief(up.pomdp)
uniform_belief(up::RobustUpdater) = uniform_belief(up.rpomdp)

function initialize_belief(bu::DiscreteUpdater, dist::Any)
    state_list = ordered_states(bu.pomdp)
    ns = length(state_list)
    b = zeros(ns)
    belief = DiscreteBelief(bu.pomdp, state_list, b)
    for s in iterator(dist)
        sidx = state_index(bu.pomdp, s)
        belief.b[sidx] = pdf(dist, s)
    end
    return belief
end

function initialize_belief(bu::RobustUpdater, dist::Any)
    state_list = ordered_states(bu.rpomdp)
    ns = length(state_list)
    b = zeros(ns)
    belief = DiscreteBelief(bu.rpomdp, state_list, b)
    for s in iterator(dist)
        sidx = state_index(bu.rpomdp, s)
        belief.b[sidx] = pdf(dist, s)
    end
    return belief
end

function update(bu::DiscreteUpdater, b::DiscreteBelief, a, o)
    pomdp = b.pomdp
    state_space = b.state_list
    bp = zeros(length(state_space))

    bp_sum = 0.0   # to normalize the distribution

    for (spi, sp) in enumerate(state_space)

        # po = O(a, sp, o)
        od = observation(pomdp, a, sp)
        po = pdf(od, o)

        if po == 0.0
            continue
        end

        b_sum = 0.0
        for (si, s) in enumerate(state_space)
            td = transition(pomdp, s, a)
            pp = pdf(td, sp)
            b_sum += pp * b.b[si]
        end

        bp[spi] = po * b_sum
        bp_sum += bp[spi]
    end

    if bp_sum == 0.0
        error("""
              Failed discrete belief update: new probabilities sum to zero.
              b = $b
              a = $a
              o = $o
              Failed discrete belief update: new probabilities sum to zero.
              """)
    else
        bp ./= bp_sum
    end

    return DiscreteBelief(pomdp, b.state_list, bp)
end

# Eq. (5) Osogami 2015
# from p[t,z,s,a] to p[t,z,s]
function minutil(prob::Union{RPOMDP,RIPOMDP}, b::Vector{Float64}, a, alphavecs::Vector{Vector{Float64}})
   nz = n_observations(prob)
   ns = n_states(prob)
   nα = length(alphavecs)
   aind = action_index(prob, a)
   plower, pupper = dynamics(prob)
   m = Model(solver = ClpSolver())
   @variable(m, u[1:nz])
   @variable(m, p[1:ns, 1:nz, 1:ns])
   @objective(m, Min, sum(u))
   for zind = 1:nz, αind = 1:nα
       @constraint(m, u[zind] >= sum(b[sind] * p[:,zind,sind]' * alphavecs[αind] for sind = 1:ns))
   end
   @constraint(m, p .<= pupper[:,:,:,aind])
   @constraint(m, p .>= plower[:,:,:,aind])
   for sind = 1:ns
       @constraint(m, sum(p[:,:,sind]) == 1)
   end
   JuMP.solve(m)
   getvalue(u), getvalue(p)
end

function update(bu::RobustUpdater, b::DiscreteBelief, a, o)
    rpomdp = b.pomdp
    state_space = b.state_list
    bp = zeros(length(state_space))
    oi = obs_index(rpomdp, o)
    bp_sum = 0.0   # to normalize the distribution
    umin, pmin = minutil(rpomdp, b.b, a, bu.alphas)
    for (spi, sp) in enumerate(state_space)
        po = sum(pmin[spi,oi,:]) / sum(pmin[spi,:,:])
        (po == 0.0) && continue
        b_sum = 0.0
        for (si, s) in enumerate(state_space)
            pp = sum(pmin[spi,:,si]) / sum(pmin[spi,:,:])
            b_sum += pp * b.b[si]
        end
        bp[spi] = po * b_sum
        bp_sum += bp[spi]
    end
    if bp_sum == 0.0
        error("""
              Failed discrete belief update: new probabilities sum to zero.
              b = $b
              a = $a
              o = $o
              Failed discrete belief update: new probabilities sum to zero.
              """)
    else
        bp ./= bp_sum
    end
    return DiscreteBelief(rpomdp, b.state_list, bp)
end

update(bu::Union{RobustUpdater,DiscreteUpdater}, b::Any, a, o) = update(bu, initialize_belief(bu, b), a, o)


# DEPRECATED
@generated function create_belief(bu::DiscreteUpdater)
    Core.println("WARNING: create_belief(up::DiscreteUpdater) is deprecated. Use uniform_belief(up) instead.")
    return :(uniform_belief(bu))
end

# alphas are |A|x|S|
# computes dot product of alpha vectors and belief
# util is array with utility of each alpha vecotr for belief b
@generated function product(alphas::Matrix{Float64}, b::DiscreteBelief)
    Core.println("WARNING: product(alphas, b::DiscreteBelief) is deprecated.")
    quote
        @assert size(alphas, 1) == length(b) "Alpha and belief sizes not equal"
        n = size(alphas, 2)
        util = zeros(n)
        for i = 1:n
            s = 0.0
            for j = 1:length(b)
                s += alphas[j,i]*b.b[j]
            end
            util[i] = s
        end
        return util
    end
end
