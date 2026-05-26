using POMDPs, Distributions, IntervalArithmetic
export ChainInf, LinkInf 

@kwdef mutable struct ChainInf <: IPOMDP{Int, Int, Int} 
    discount::Float64    = 0.95
    # p1_interval::Interval  = interval(0.1, 0.1)
    # p2_interval::Interval  = interval(0.5,0.8)
    # p3_interval::Interval  = interval(0.1,0.4)
    # p1_interval::Interval  = interval(0.2, 0.4)
    # p2_interval::Interval  = interval(0.1,0.4)
    # p3_interval::Interval  = interval(0.3,0.6)
    p1_interval::Interval  = interval(0.2, 0.2)
    p2_interval::Interval  = interval(0.1,0.7)
    p3_interval::Interval  = interval(0.1,0.7)
end

struct LinkInf
    parity::Int
    steps::Int
end
Base.:(==)(l1::LinkInf, l2::LinkInf) = l1.parity == l2.parity && l1.steps == l2.steps
Base.isequal(l1::LinkInf, l2::LinkInf) = l1==l2
Base.:(<)(l1::LinkInf, l2::LinkInf) = l1.parity < l2.parity || (l1.parity == l2.parity && l1.steps < l2.steps)
Base.isless(l1::LinkInf, l2::LinkInf) = l1 < l2
Base.hash(l::LinkInf, h::UInt) = hash(hash(l.parity, hash(l.steps)), h)
Base.hash(l::LinkInf) = hash(l, UInt(0))

POMDPs.states(M::ChainInf) = [LinkInf(0,0), LinkInf(0,-2), LinkInf(0,1), LinkInf(0,2), LinkInf(0,3), LinkInf(1,-2), LinkInf(1,1), LinkInf(1,2), LinkInf(1,3)]
POMDPs.statetype(M::ChainInf) = LinkInf
POMDPs.stateindex(M::ChainInf, s) = findfirst(map(sp -> s==sp, states(M))) # TODO: this is inefficient I think...
POMDPs.actions(M::ChainInf) = [0,1,2,3] # odd, even, sodd, seven
POMDPs.actiontype(M::ChainInf) = Int
POMDPs.actionindex(M::ChainInf, a) = a+1
POMDPs.observations(M::ChainInf) = 1 #NullObs
POMDPs.obstype(M::ChainInf) = Int
POMDPs.obsindex(M::ChainInf, o) = 1
POMDPs.discount(M::ChainInf) = M.discount
POMDPs.initialstate(M::ChainInf) = SparseCat([LinkInf(0,0)], [1.0])
POMDPs.isterminal(M::ChainInf, s) = false

parity(s::LinkInf) = s.parity
notparity(s::LinkInf) = mod(s.parity+1,2)

function POMDPs.transition(M::ChainInf, s,a)
    spar = parity(s)
    if spar==a
        return SparseICat([LinkInf(notparity(s),1)], [1.0])
    elseif spar+2==a
        return SparseICat([LinkInf(notparity(s),1), LinkInf(parity(s),2), LinkInf(notparity(s),3)], [M.p1_interval, M.p2_interval, M.p3_interval])
    elseif (spar==0 && a==1) || (spar==1 && a==0)
        return SparseICat([LinkInf(parity(s),-2)],[1.0])
    elseif (spar==0 && a==3) || (spar==1 && a==2)
        return SparseICat([LinkInf(parity(s),-2)],[1.0])
    end
    println("Error: unrecognized transition ($s, $a)!")
    return SparseICat([0],interval(1))
end

function POMDPs.observation(M::ChainInf, a, sp)
    # return SparseICat([1],[interval(1)])
    return SparseCat([1],[1.0])
end

POMDPs.reward(M::ChainInf, s, a, sp) = reward(M, s, a)
function POMDPs.reward(M::ChainInf, s, a)
    # This is hacky: SparseTabularPOMDPs does not allow sp-based reward functions, so we cannot use those.
    # Instead we give the reward for the current state * 1/d, which is the same but looks weird...
    return 1/M.discount * s.steps
end

@kwdef mutable struct ChainN <: IPOMDP{Int, Int, Int}
    discount::Float64       = 0.95
    N::Int                  = 10
    p1_interval::Interval  = interval(0.1, 0.1)
    p2_interval::Interval  = interval(0.5,0.8)
    p3_interval::Interval  = interval(0.1,0.4)
    # p1_interval::Interval  = interval(0.2, 0.2)
    # p2_interval::Interval  = interval(0.1,0.7)
    # p3_interval::Interval  = interval(0.1,0.7)
end

POMDPs.states(M::ChainN) = 0:M.N+1 # Chain plus sink state 
POMDPs.statetype(M::ChainN) = Int
POMDPs.stateindex(M::ChainN, s) = findfirst(map(sp -> s==sp, states(M))) # TODO: this is inefficient I think...
POMDPs.actions(M::ChainN) = [0,1,2,3] # odd, even, sodd, seven
POMDPs.actiontype(M::ChainN) = Int
POMDPs.actionindex(M::ChainN, a) = a+1
POMDPs.observations(M::ChainN) = 1 #NullObs
POMDPs.obstype(M::ChainN) = Int
POMDPs.obsindex(M::ChainN, o) = 1
POMDPs.discount(M::ChainN) = M.discount
POMDPs.initialstate(M::ChainN) = SparseCat([0], [1.0])
POMDPs.isterminal(M::ChainN, s) = s==M.N+1

function POMDPs.transition(M::ChainN, s,a)
    s in [M.N, M.N+1] && return SparseICat([M.N+1], [1.0])
    this_parity = mod(s,2)

    sb = max(0, s-2)
    sp = min(s+1, M.N)
    spp = min(s+2, M.N)
    sppp = min(s+3, M.N)

    # Correct parity, normal
    if this_parity==a
        return SparseICat([sp], [1.0])

    # Correct parity, stochastic
    elseif this_parity+2==a
        # Boundary
        s == M.N-1 && return SparseICat([M.N], [1.0])
        s == M.N-2 && return SparseICat([M.N-1, M.N], [M.p1_interval, interval(0.001, 0.999)])
        # Normal
        return SparseICat([sp, spp, sppp], [M.p1_interval, M.p2_interval, M.p3_interval])

    # Incorrect parity, either
    else (this_parity==0 && a==1) || (this_parity==1 && a==0)
        return SparseICat([sb],[1.0])
    end
end

function POMDPs.observation(M::ChainN, a, sp)
    # return SparseICat([1],[interval(1)])
    return SparseCat([1],[1.0])
end

POMDPs.reward(M::ChainN, s, a) = s == M.N ? (return 100.0) : (return 0.0)
