"""Given an RPOMDP m, returns a POMDP where all model uncertainty is decided by function f(model, s, a) -> distr."""
function to_POMDP(m::IPOMDP, f)
    S, O = states(m), observations(m)
    ns, na = length(states(m)), length(actions(m))
    S_dict = Dict( zip(states(m), 1:ns))
    A_dict = Dict( zip(actions(m), 1:na))
    O_dict = Dict( zip(observations(m), 1:length(observations(m))))
    T, O, R = Array{Any}(undef,na), Array{Any}(undef,na), Array{Any}(undef,ns, na)

    Os, Oo, Oprob = [], [], []

    for a in actions(m)
        aidx = actionindex(m,a)
   
        Ts, Tsp, Tprob = [], [], []
        Os, Oo, Oprob = [], [], []

        for s in states(m)
            sidx = stateindex(m,s)
            # Transitions:
            thisT_avg = f(m,s,a)
            append!(Ts, repeat([sidx], length(support(thisT_avg))))
            append!(Tsp, map(sp->stateindex(m,sp),support(thisT_avg)))
            append!(Tprob, map(x->pdf(thisT_avg, x), support(thisT_avg)))
                       
            # Rewards:
            R[sidx,aidx] = reward(m,s,a)
        end
        T[aidx] = sparse(Ts,Tsp,Tprob)
        O[aidx] = sparse(Os, Oo, Oprob)
    end

    # TODO: may be buggy...
    function Tf(s,a)
        sidx, aidx = S_dict[s], A_dict[a]
        sps, probs, = findnz(T[aidx][sidx,:])
        sps = map(sidx->S[sidx], sps)
        return SparseCat(sps, probs)
    end

    return QuickPOMDP(
        actions = actions(m),
        states = states(m),
        observations = observations(m),

        transition = Tf,
        observation = (a,sp) -> observation(m,a,sp),
        reward = (s,a) -> reward(m,s,a),

        initialstate = initialstate(m),
        isterminal = s->isterminal(m,s),
        discount=discount(m)
    )
end

to_mid_POMDP(m) = to_POMDP(m, (m,s,a)->find_avg_normalised_distr(m,s,a))
to_maxent_POMDP(m) = to_POMDP(m, (m,s,a)->find_entropy_maximized_distr(m,s,a))

function to_rmdp_POMDP(m::X) where X<:RPOMDP
    println("Unimplemented: requires code from RHSVI to run")
    # rqmdp_solver = RQMDPSolver()
    # rmdp_policy = solve(rqmdp_solver, m)
    # Qmax = map(s -> maximum(alpha -> alpha[s], rmdp_policy.alphas), states(m))
    # return to_POMDP(m, (m,s,a)->solvestep(RQMDPSolver(), m, s, a, Qmax;return_belief=true)[2])
end


to_sparsevec(m,d) = sparsevec(map(x->stateindex(m,x),d.vals), d.probs)

function find_avg_normalised_distr(m::X, s, a; f = i -> mid(i)) where X<:IPOMDP
    D = transition(m,s,a)
    vals, intervals =  D.vals, D.probs
    idxs = 1:length(vals)
    probs = map(x -> f(x), intervals)

    S = sum(probs)
    while !isapprox(S, 1.0)
        S < 1.0 ? (bound=i->sup(i); cond=(x,y)->min(x,y)) : (bound=i->inf(i); cond=(x,y)->max(x,y))

        changeable_idxs = findall( idx -> bound(intervals[idx]) != probs[idx], idxs)
        length(changeable_idxs) == 0 && (println("Error: cannot find normalized distribution for s=$s, a=$a, transition=$D"); return SparseCat(vals,probs))
        
        prob_mass_changeables = sum(probs[changeable_idxs])
        prob_mass_nonchangeables = S - prob_mass_changeables
        factor = (1-prob_mass_nonchangeables) / prob_mass_changeables
        probs = map( idx ->  cond(factor*probs[idx], bound(intervals[idx])) , idxs)
        S = sum(probs)
    end
    return SparseCat(vals, probs)
end


function find_entropy_maximized_distr(m::X, sprev, aprev) where X<:IPOMDP

    model = Model(Clp.Optimizer; add_bridges=false)
    set_silent(model)
    set_string_names_on_creation(model, false)
    
    T = transition(m,sprev,aprev)
    Sp = support(T)
    Sp_idxs = map(s->stateindex(m,s), Sp)

    # All probabilities fall within intervals and sum to 1.
    @variable(model, 0.0 <= prob_sp[1:length(Sp)] <= 1.0)
    @constraint(model, sum(prob_sp) == 1.0)
    for (sidx,s) in enumerate(Sp)
        @constraint(model, inf(pdf(T,s)) <= prob_sp[sidx] <= sup(pdf(T,s)))
    end


    @variable(model, 0.0 <= err[1:length(Sp), 1:length(Sp)])
    for (sidx, s) in enumerate(Sp)
        for (spidx, sp) in enumerate(Sp)
            # spidx <= sidx && continue
            @constraint(model, err[sidx, spidx] >= prob_sp[sidx] - prob_sp[spidx])
        end
    end
    
    @objective(model, Min, sum(err))
    optimize!(model)
    return SparseCat(Sp, JuMP.value.(prob_sp))
end