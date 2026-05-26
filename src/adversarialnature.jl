mutable struct Sn{S,A}
    x::Any                      # The internal state representation for the policy (e.g. a belief)
    s::S                        # The underlying real state of the environment
    a::A
end

Base.:(==)(sn1::Sn, sn2::Sn) = (sn1.x == sn2.x) && (sn1.s == sn2.s) && (sn1.a == sn2.a)
Base.hash(sn::Sn, h::UInt) = hash(hash(hash(hash(sn.x), hash(sn.x)), hash(sn.a)), h)
Base.hash(sn::Sn) = hash(sn, UInt(0))

mutable struct An 
    sps#<:Vector{Any}
    probs#::Vector{Any}
end

Base.:(==)(a1::An, a2::An) = (a1.sps == a2.sps) && (a1.probs == a2.probs)
hash_alt(v::Vector) = foldr( (x,y) -> hash(x,y), v; init=UInt(0))
Base.hash(a::An, h::UInt) = hash(hash(hash_alt(a.sps), hash_alt(a.probs)), h)
Base.hash(a::An) = hash(a, UInt(0))

function get_model_adversary(M::X, π::Policy) where X<:RPOMDP
    
    # Get states & actions
    constants = get_constants(M)
    S_dict = Dict( zip(constants.S, 1:constants.ns)) 
    all_actions = get_all_adversarial_actions(M, constants) # note: this is kinda expensive to compute!
    actions(sn) = all_actions[S_dict[sn.s],actionindex(M, action(π, sn.x))]
    sn_isterminal(sn) = isterminal(M,sn.s)
    
    R(sn, _a) = -reward(M, sn.s, sn.a)
    
    # Get initial state
    b0 = initialstate(M)
    x_init = get_initial_memory(π)
    a_init = action_distr(π,b0)
    sns, probs = [], []
    for s in support(b0)
        for a in support(a_init)
            totprob = pdf(b0,s) * pdf(a_init,a)
            if totprob > 0.0
                push!(sns, Sn(x_init,s,a))
                push!(probs, pdf(b0,s) * pdf(a_init,a))
            end
        end
    end
    isempty(sns) && println("Error: empty initial nature belief! (b0 = $b0, x_init = $x_init, a_init = $a_init)")
    # probs = probs ./ sum(probs)
    sn_init = SparseCat(sns, probs)

    # Define transition function
    @memoize LRU(maxsize=10_000) function T(sn,an)
        x, s, a = sn.x, sn.s, sn.a
        snps, probs = [], []
        # Loop over all next states, memory states (<- observations) and agent actions and compute probabilities
        for (spidx, sp) in enumerate(an.sps)
            prob_sp = an.probs[spidx]
            for (o, prob_o) in weighted_iterator(observation(M,a,sp))
                xp = update_memory(π,x,a,o)
                Adistr = action_distr(π, xp)
                for (ap, prob_a) in weighted_iterator(Adistr)
                    totprob = prob_sp*prob_o*prob_a
                    if totprob > 0.0
                        push!(snps, Sn(xp,sp,ap))
                        push!(probs, prob_sp*prob_o*prob_a)
                    end
                end
            end
        end
        isempty(sns) && println("Error: No valid next states for sn = $sn and an = $an")
        return SparseCat(snps, probs)
    end

    # Combine into MDP:
    return QuickMDP(
        actions = actions,

        actiontype = An,

        transition = T,
        reward = R,

        initialstate = sn_init,
        isterminal=sn_isterminal,
        discount = discount(M)
    )
end

get_adversarial_actions(M, π, sn::Sn) = get_extreme_points(transition(M,sn.s, sn.a ))
get_adversarial_actions(M, s,a) = get_extreme_points(transition(M,s, a ))
function get_all_adversarial_actions(M, constants::C)
    all_actions = Array{Any}(undef,constants.ns, constants.na)
    for sidx in 1:constants.ns
        for aidx in 1:constants.na
            an = get_adversarial_actions(M,constants.S[sidx],constants.A[aidx])
            all_actions[sidx,aidx] = an
        end
    end
    return all_actions
end

function get_extreme_points(d; max_prob=1.0)
    # 'Unpack' distribution to arrays of states & minimal/maximal transition probabilities
    states, intervals = support(d), map(s->pdf(d,s), support(d))
    probs_min, probs_max = Float64[], Float64[]
    for int in intervals
        append!(probs_min, inf(int))
        append!(probs_max, sup(int))
    end
    max_rest_prob = 1 - sum(probs_min)

    # If minimal probabilities already sum to 1, we're done
    if max_rest_prob == 0.0
        This_Ans = [An(states, probs_min)]
    # Otherwise, rewrite as a problem of finding extreme points of a polytope and solve using LP
    else
        # include contraints given by probs_min into probs_max
        probs_max = map(idx -> min(probs_max[idx], max_rest_prob + probs_min[idx]), 1:length(states))
        # Call method
        _is_valid, extreme_points = get_extreme_points(states, probs_min, probs_max)
        This_Ans = map(ext -> An(states, ext), extreme_points)
    end
    return This_Ans
end

function get_extreme_points(states, probs_min, probs_max)
    nmbr_states = length(states)
    probs_diff = probs_max .- probs_min

    # use a solver that allows finding all feasible solutions (so Clp & HiGHS are not possible...)
    model = direct_generic_model(Float64, Gurobi.Optimizer(GRB_ENV[]))
    set_silent(model)
    set_string_names_on_creation(model, false)
    set_optimizer_attribute(model, "PoolSearchMode", 2)
    set_optimizer_attribute(model, "PoolSolutions", 1_000)

    # Define LP to find extreme points:
    @variable(model, is_maximized[1:nmbr_states], Bin)
    @variable(model, is_partially_maximized[1:nmbr_states], Bin)

    # A transition can be maximized, partially maximized, or minimized (= neither)
    @constraint(model, [idx in 1:nmbr_states], is_maximized[idx] + is_partially_maximized[idx] <= 1)

    # Only one partially maximized transition is allowed
    @constraint(model, sum(is_partially_maximized) == 1 )

    # If we maximized the partially maximized transition, total prob >= 1.0
    @constraint(model, sum(is_maximized .* probs_max) + sum( (1 .- is_maximized) .* probs_min) + sum(is_partially_maximized .* probs_diff) >= 1.0)

    # If we minimized the partial maximized transtion, total prob <= 1.0
    @constraint(model, sum(is_maximized .* probs_max) + sum( (1 .- is_maximized) .* probs_min) <= 1.0)
    optimize!(model)
    if termination_status(model) != MOI.OPTIMAL 
        println("Error: no valid extreme points found for the following setting")
        println("states = $states")
        println("probs_min = $probs_min")
        println("probs_max = $probs_max")
        println(model)
    end

    # Unpack results:
    all_extreme_points = []
    for i in 1:result_count(model)

        # get maximized and minimized transitions
        this_probs = zeros(nmbr_states)
        maximized_probs = map(x->isapprox(x,1.0), JuMP.value.(is_maximized; result=i))
        this_probs += probs_max .* maximized_probs
        this_probs += probs_min .* (.!maximized_probs)

        # Get partially maximized transition
        sum_probs = sum(this_probs)
        if sum_probs < 1.0
            nonextreme_prob = findfirst(map(x->isapprox(x,1.0),JuMP.value.(is_partially_maximized; result=i)))
            this_probs[nonextreme_prob] = 1.0 - (sum_probs - probs_min[nonextreme_prob])
        end

        push!(all_extreme_points, this_probs)
    end
    # println(unique(Ans))
    return (length(all_extreme_points)!=0, unique(all_extreme_points)) # Remove duplicates
end


function get_extreme_points_belief(env::IPOMDP, b)
    model = direct_generic_model(Float64, Gurobi.Optimizer(GRB_ENV[]))
    set_silent(model)
    set_string_names_on_creation(model, false)
    set_optimizer_attribute(model, "PoolSearchMode", 2)
    set_optimizer_attribute(model, "PoolSolutions", 1_000)
    
    @variable(model, is_maximized )
end