# HistoryRecorder
# maintained by @zsunberg

"""
A simulator that records the history for later examination

The simulation will be terminated when either
1) a terminal state is reached (as determined by `isterminal()` or
2) the discount factor is as small as `eps` or
3) max_steps have been executed

Keyword Arguments:
    - `rng`: The random number generator for the simulation
    - `capture_exception::Bool`: whether to capture an exception and store it in the history, or let it go uncaught, potentially killing the script
    - `show_progress::Bool`: show a progress bar for the simulation
    - `eps`
    - `max_steps`
    - `sizehint::Int`: the expected length of the simulation (for preallocation)

Usage (optional arguments in brackets):
    hr = HistoryRecorder()
    history = simulate(hr, pomdp, policy, [updater [, init_belief [, init_state]]])
"""
mutable struct HistoryRecorder <: Simulator
    rng::AbstractRNG

    # options
    capture_exception::Bool
    show_progress::Bool

    # optional: if these are null, they will be ignored
    max_steps::Nullable{Any}
    eps::Nullable{Any}
    sizehint::Nullable{Integer}

    # DEPRECATED
    initial_state::Nullable{Any}
end

# This is the only stable constructor
function HistoryRecorder(;rng=MersenneTwister(rand(UInt32)),
                          initial_state=Nullable{Any}(),
                          eps=Nullable{Any}(),
                          max_steps=Nullable{Any}(),
                          sizehint=Nullable{Integer}(),
                          capture_exception=false,
                          show_progress=false)
    if !isnull(initial_state)
        warn("The initial_state argument for HistoryRecorder is deprecated. The initial state should be specified as the last argument to simulate(...).")
    end
    return HistoryRecorder(rng, capture_exception, show_progress, max_steps, eps, sizehint, initial_state)
end

function simulate(sim::HistoryRecorder, pomdp::Union{POMDP,IPOMDP,RPOMDP,RIPOMDP}, policy::Policy, bu::Updater=updater(policy))
    dist = initial_state_distribution(pomdp)
    return simulate(sim, pomdp, policy, bu, dist)
end

function simulate{S,A,O}(sim::HistoryRecorder,
                           pomdp::Union{POMDP{S,A,O},IPOMDP{S,A,O},RPOMDP{S,A,O},RIPOMDP{S,A,O}},
                           policy::Policy,
                           bu::Updater,
                           initial_state_dist::Any,
                           initial_state::Any=get_initial_state(sim, initial_state_dist)
                          )

    initial_belief = initialize_belief(bu, initial_state_dist)
    max_steps = get(sim.max_steps, typemax(Int))
    if !isnull(sim.eps)
        max_steps = min(max_steps, ceil(Int,log(get(sim.eps))/log(discount(pomdp))))
    end
    sizehint = get(sim.sizehint, min(max_steps, 1000))

    # aliases for the histories to make the code more concise
    sh = sizehint!(Vector{S}(0), sizehint)
    ah = sizehint!(Vector{A}(0), sizehint)
    oh = sizehint!(Vector{O}(0), sizehint)
    bh = sizehint!(Vector{typeof(initial_belief)}(0), sizehint)
    rh = sizehint!(Vector{Float64}(0), sizehint)
    ih = sizehint!(Vector{Any}(0), sizehint)
    aih = sizehint!(Vector{Any}(0), sizehint)
    uih = sizehint!(Vector{Any}(0), sizehint)
    exception = Nullable{Exception}()
    backtrace = Nullable{Any}()

    push!(sh, initial_state)
    push!(bh, initial_belief)

    if sim.show_progress
        if isnull(sim.max_steps) && isnull(sim.eps)
            error("If show_progress=true in a HistoryRecorder, you must also specify max_steps or eps.")
        end
        prog = Progress(max_steps, "Simulating..." )
    end

    disc = 1.0
    step = 1

    try
        while !isterminal(pomdp, sh[step]) && step <= max_steps
            a, ai = action_info(policy, bh[step])
            push!(ah, a)
            push!(aih, ai)

            sp, o, r, i = generate_sori(pomdp, bh[step].b, sh[step], ah[step], sim.rng)

            push!(sh, sp)
            push!(oh, o)
            push!(rh, r)
            push!(ih, i)

            bp, ui = update_info(bu, bh[step], ah[step], oh[step])
            push!(bh, bp)
            push!(uih, ui)

            step += 1

            if sim.show_progress
                next!(prog)
            end
        end
    catch ex
        if sim.capture_exception
            exception = Nullable{Exception}(ex)
            backtrace = Nullable{Any}(catch_backtrace())
        else
            rethrow(ex)
        end
    end

    if sim.show_progress
        finish!(prog)
    end

    return POMDPHistory(sh, ah, oh, bh, rh, ih, aih, uih, discount(pomdp), exception, backtrace)
end

function simulate{S,A}(sim::HistoryRecorder,
                       mdp::MDP{S,A}, policy::Policy,
                       init_state::S=get_initial_state(sim, mdp))

    max_steps = get(sim.max_steps, typemax(Int))
    if !isnull(sim.eps)
        max_steps = min(max_steps, ceil(Int,log(get(sim.eps))/log(discount(mdp))))
    end
    sizehint = get(sim.sizehint, min(max_steps, 1000))

    # aliases for the histories to make the code more concise
    sh = sizehint!(Vector{S}(0), sizehint)
    ah = sizehint!(Vector{A}(0), sizehint)
    rh = sizehint!(Vector{Float64}(0), sizehint)
    ih = sizehint!(Vector{Any}(0), sizehint)
    aih = sizehint!(Vector{Any}(0), sizehint)
    exception = Nullable{Exception}()
    backtrace = Nullable{Any}()

    if sim.show_progress
        prog = Progress(max_steps, "Simulating..." )
    end

    push!(sh, init_state)

    disc = 1.0
    step = 1

    try
        while !isterminal(mdp, sh[step]) && step <= max_steps
            a, ai = action_info(policy, sh[step])
            push!(ah, a)
            push!(aih, ai)

            sp, r, i = generate_sri(mdp, sh[step], ah[step], sim.rng)

            push!(sh, sp)
            push!(rh, r)
            push!(ih, i)

            disc *= discount(mdp)
            step += 1

            if sim.show_progress
                next!(prog)
            end
        end
    catch ex
        if sim.capture_exception
            exception = Nullable{Exception}(ex)
            backtrace = Nullable{Any}(catch_backtrace())
        else
            rethrow(ex)
        end
    end

    if sim.show_progress
        finish!(prog)
    end

    return MDPHistory(sh, ah, rh, ih, aih, discount(mdp), exception, backtrace)
end

function get_initial_state(sim::Simulator, initial_state_dist)
    if isnull(sim.initial_state)
        return rand(sim.rng, initial_state_dist)
    else
        return get(sim.initial_state)
    end
end

function get_initial_state(sim::Simulator, mdp::Union{MDP,POMDP,IPOMDP,RPOMDP,RIPOMDP})
    if isnull(sim.initial_state)
        return initial_state(mdp, sim.rng)
    else
        return get(sim.initial_state)
    end
end
