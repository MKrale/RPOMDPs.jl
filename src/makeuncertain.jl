abstract type IntervalType end
struct AdditiveAbs <: IntervalType end
struct AdditiveRel <: IntervalType end
struct Multiplicative <: IntervalType end

struct ConfidencePOMDP{S,A,O} <: IPOMDP{S,A,O} # Possibly the worst name I've ever given anything in my life...
    pomdp::X where X<:POMDP{S,A,O}
    d::Float64
    interval_type::I where I<:IntervalType
end

# Transitions are changed to intervals, with interval_type & d deciding how to construct them
function POMDPs.transition(M::ConfidencePOMDP, s,a)
    epsilon = 1e-5  # Ensures non-vanishing transitions
    d = M.d
    d <= 0.0 && return transition(M.pomdp,s,a)
    if M.interval_type isa AdditiveAbs
        f = prob -> (   prob == 1.0 && return interval(1.0);
                        prob == 0.0 && return interval(0.0);
                        pmin = max(epsilon,prob-d);
                        pmax = min(1.0, prob+d);
                        return interval(min(pmin,pmax), pmax) )
    elseif M.interval_type isa AdditiveRel
        f = prob -> (   prob == 1.0 && return interval(1.0);
                        prob == 0.0 && return interval(0.0);
                        pmin = max(epsilon, prob*(1-d));
                        pmax = min(1.0, prob*(1+d));
                        return interval(min(pmin,pmax), max(pmin,pmax)) )
    elseif M.interval_type isa Multiplicative
        f = prob -> (   prob == 1.0 && return interval(1.0);
                        prob == 0.0 && return interval(0.0);
                        pmin = epsilon;
                        pmax = min(1.0, 1/M.d * prob);
                        return interval(min(pmin,pmax), max(pmin,pmax)) )
    else
        throw("Error: interval type $(M.interval) not recongized!")
    end
    return general_robustified_transition(M,s,a,f)
end

function general_robustified_transition(M::X,s,a,f) where X<: ConfidencePOMDP
    T_init = POMDPs.transition(M.pomdp, s, a)
    sps, intervals = [], []
    for sp in support(T_init)
        if pdf(T_init, sp) > 0.0
            push!(sps, sp)
            push!(intervals, f(pdf(T_init, sp)))
        end
    end
    return SparseICat(sps, intervals)
end

# All other functions are simply redirected to the underlying POMDP
POMDPs.states(M::ConfidencePOMDP) = POMDPs.states(M.pomdp)
POMDPs.statetype(::Type{ConfidencePOMDP})  = POMDPs.statetype(M.pomdp)
POMDPs.stateindex(M::ConfidencePOMDP, s) = POMDPs.stateindex(M.pomdp, s)

POMDPs.actions(M::ConfidencePOMDP) = POMDPs.actions(M.pomdp)
POMDPs.actiontype(::Type{ConfidencePOMDP})  = POMDPs.actiontype(M.pomdp)
POMDPs.actionindex(M::ConfidencePOMDP, a) = POMDPs.actionindex(M.pomdp, a)

POMDPs.observations(M::ConfidencePOMDP) = POMDPs.observations(M.pomdp)
POMDPs.obstype(::Type{ConfidencePOMDP})  = POMDPs.obstype(M.pomdp)
POMDPs.obsindex(M::ConfidencePOMDP, o) = POMDPs.obsindex(M.pomdp, o)

POMDPs.discount(M::ConfidencePOMDP) = POMDPs.discount(M.pomdp)
POMDPs.initialstate(M::ConfidencePOMDP) = POMDPs.initialstate(M.pomdp)
POMDPs.isterminal(M::ConfidencePOMDP, s) = POMDPs.isterminal(M.pomdp, s)

POMDPs.observation(M::ConfidencePOMDP, a, sp) = POMDPs.observation(M.pomdp, a, sp)
POMDPs.reward(M::ConfidencePOMDP, s, a) = POMDPs.reward(M.pomdp, s, a)