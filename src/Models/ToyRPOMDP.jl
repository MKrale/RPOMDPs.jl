@kwdef mutable struct ToyRPOMDP <: IPOMDP{Int, Int, Int} 
    discount::Float64       = 0.9999
    p_interval::Interval    = interval(0.1,0.9)
    Rsafe::Float64          = 70.0
end
ToyRPOMDP_mid() = ToyRPOMDP(p_interval=interval(0.5, 5/6), Rsafe = 80.0)
ToyRPOMDP_rmdp() = ToyRPOMDP(p_interval=interval(0.2, 0.5), Rsafe = 80.0) #TODO: insert correct interval


POMDPs.states(M::ToyRPOMDP) = 0:8
POMDPs.statetype(M::ToyRPOMDP) = Int
POMDPs.stateindex(M::ToyRPOMDP, s::Int) = s+1
POMDPs.actions(M::ToyRPOMDP) = [1,2]
POMDPs.actiontype(M::ToyRPOMDP) = Int
POMDPs.actionindex(M::ToyRPOMDP, a::Int) = a
POMDPs.observations(M::ToyRPOMDP) = 1:2
POMDPs.obstype(M::ToyRPOMDP) = Int
POMDPs.obsindex(M::ToyRPOMDP, o::Int) = o
POMDPs.discount(M::ToyRPOMDP) = M.discount
POMDPs.initialstate(M::ToyRPOMDP) = SparseCat([0], [1.0])
POMDPs.isterminal(M::ToyRPOMDP, s::Int) = s==8

function POMDPs.transition(M::ToyRPOMDP, s,a)
    s==0 && return SparseICat([1,2], [interval(0.5), interval(0.5)])
    s==1 && return SparseICat([3,4], [interval(0.5), interval(0.5)])
    s==2 && return SparseICat([4], [interval(1.0)])
    s in [3,4] && a==2 && return SparseICat([8], [interval(1.0)])
    s==3 && return SparseICat([5], [interval(1.0)])
    s==4 && return SparseICat([6,7], [M.p_interval, M.p_interval])
    s in [5,6,7,8] && return SparseICat([8], [interval(1.0)])
    println("help! $s, $a")
end

function POMDPs.observation(M::ToyRPOMDP, a, sp)
    sp==2 && return SparseCat([2], [1.0])
    return SparseCat([1], [1.0])
end

POMDPs.reward(M::ToyRPOMDP, s, a, sp) = reward(M,s,a)
function POMDPs.reward(M::ToyRPOMDP, s, a)
    s in [3,4] && a==2 && return 70.0
    s==5 && a==1 && return 100.0
    s==6 && a==1 && return 100.0
    s==7 && a==2 && return 200.0
    return 0.0
end