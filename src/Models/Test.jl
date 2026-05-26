@kwdef mutable struct Test_Backup <: IPOMDP{Int, Int, Int} 
    discount::Float64       = 0.9999
    p1_interval::Interval    = interval(0.7,0.9)
    p2_interval::Interval    = interval(0.9, 0.9)
end

max_interval = interval(0.0001, 0.9999)

POMDPs.states(M::Test_Backup) = 0:5
POMDPs.statetype(M::Test_Backup) = Int
POMDPs.stateindex(M::Test_Backup, s::Int) = s+1
POMDPs.actions(M::Test_Backup) = [1,2]
POMDPs.actiontype(M::Test_Backup) = Int
POMDPs.actionindex(M::Test_Backup, a::Int) = a
POMDPs.observations(M::Test_Backup) = [1]
POMDPs.obstype(M::Test_Backup) = Int
POMDPs.obsindex(M::Test_Backup, o::Int) = o
POMDPs.discount(M::Test_Backup) = M.discount
POMDPs.initialstate(M::Test_Backup) = SparseCat([0], [1.0])
POMDPs.isterminal(M::Test_Backup, s::Int) = s==5

function POMDPs.transition(M::Test_Backup, s,a)
    s==0 && a==1 && return SparseICat([1], [interval(1.0)])
    s==0 && a==2 && return SparseICat([1,2], [interval(0.5), interval(0.5)])
    s==1 && return SparseICat([3,4], [M.p1_interval, max_interval])
    s==2 && return SparseICat([3,4], [max_interval, M.p2_interval,])
    s>2 && return SparseICat([5], [interval(1.0)])
    println("help! $s, $a")
end

POMDPs.observation(M::Test_Backup, a, sp) = SparseCat([1], [1.0])

POMDPs.reward(M::Test_Backup, s, a, sp) = reward(M,s,a)
function POMDPs.reward(M::Test_Backup, s, a)
    s==0 && a==1 && return -0.4 
    s==3 && a==1 && return 1.1
    s==4 && a==2 && return 1.0
    return 0.0
end

@kwdef mutable struct Test_Random <: IPOMDP{Int, Int, Int} 
    discount::Float64       = 0.9999
    p_interval::Interval    = interval(0.1,0.9)
end

POMDPs.states(M::Test_Random) = 0:5
POMDPs.statetype(M::Test_Random) = Int
POMDPs.stateindex(M::Test_Random, s::Int) = s+1
POMDPs.actions(M::Test_Random) = [1,2]
POMDPs.actiontype(M::Test_Random) = Int
POMDPs.actionindex(M::Test_Random, a::Int) = a
POMDPs.observations(M::Test_Random) = [1]
POMDPs.obstype(M::Test_Random) = Int
POMDPs.obsindex(M::Test_Random, o::Int) = o
POMDPs.discount(M::Test_Random) = M.discount
POMDPs.initialstate(M::Test_Random) = SparseCat([0], [1.0])
POMDPs.isterminal(M::Test_Random, s::Int) = s==5

function POMDPs.transition(M::Test_Random, s,a)
    s==0 && return SparseICat([1,2,3,4], [0.1, M.p_interval, M.p_interval, 0.1])
    return SparseICat([5], [1.0])
end

POMDPs.observation(M::Test_Random, a, sp) = SparseCat([1], [1.0])

POMDPs.reward(M::Test_Random, s, a, sp) = reward(M,s,a)
function POMDPs.reward(M::Test_Random, s, a)
    s == 1 && a == 1 && return 0.1
    s == 2 && a == 1 && return 1.0
    s == 3 && a == 2 && return 0.8
    s == 4 && a == 1 && return 0.9
    s == 5 && a == 2 && return 1.0
    return 0.0
end


@kwdef mutable struct Test_Random2 <: IPOMDP{Int, Int, Int} 
    discount::Float64       = 0.9999
    p_interval::Interval    = interval(0.1,0.9)
end

POMDPs.states(M::Test_Random2) = 0:5
POMDPs.statetype(M::Test_Random2) = Int
POMDPs.stateindex(M::Test_Random2, s::Int) = s+1
POMDPs.actions(M::Test_Random2) = [1,2]
POMDPs.actiontype(M::Test_Random2) = Int
POMDPs.actionindex(M::Test_Random2, a::Int) = a
POMDPs.observations(M::Test_Random2) = [1]
POMDPs.obstype(M::Test_Random2) = Int
POMDPs.obsindex(M::Test_Random2, o::Int) = o
POMDPs.discount(M::Test_Random2) = M.discount
POMDPs.initialstate(M::Test_Random2) = SparseCat([0], [1.0])
POMDPs.isterminal(M::Test_Random2, s::Int) = s==5

function POMDPs.transition(M::Test_Random2, s,a)
    s==0 && return SparseICat([1,2,3,4], [0.1, M.p_interval, M.p_interval, 0.1])
    return SparseICat([5], [1.0])
end

POMDPs.observation(M::Test_Random2, a, sp) = SparseCat([1], [1.0])

POMDPs.reward(M::Test_Random2, s, a, sp) = reward(M,s,a)
function POMDPs.reward(M::Test_Random2, s, a)
    s == 1 && a == 1 && return 0.1
    s == 2 && a == 1 && return 1.0
    s == 3 && a == 2 && return 0.8
    s == 4 && a == 1 && return 0.9
    s == 5 && a == 2 && return 1.0
    return 0.0
end

@kwdef mutable struct Test_Random3 <: IPOMDP{Int, Int, Int} 
    discount::Float64       = 0.9999
    p_interval::Interval    = interval(0.1,0.9)
end

POMDPs.states(M::Test_Random3) = 0:6
POMDPs.statetype(M::Test_Random3) = Int
POMDPs.stateindex(M::Test_Random3, s::Int) = s+1
POMDPs.actions(M::Test_Random3) = [1,2]
POMDPs.actiontype(M::Test_Random3) = Int
POMDPs.actionindex(M::Test_Random3, a::Int) = a
POMDPs.observations(M::Test_Random3) = [1,2]
POMDPs.obstype(M::Test_Random3) = Int
POMDPs.obsindex(M::Test_Random3, o::Int) = o
POMDPs.discount(M::Test_Random3) = M.discount
POMDPs.initialstate(M::Test_Random3) = SparseCat([0], [1.0])
POMDPs.isterminal(M::Test_Random3, s::Int) = s==6

function POMDPs.transition(M::Test_Random3, s,a)
    s==0 && return SparseICat([1,2], [0.5, 0.5])
    s==1 && return SparseICat([3,4,5], [0.1, M.p_interval, M.p_interval])
    s==2 && return SparseICat([3,4,5], [M.p_interval, 0.1, M.p_interval])
    return SparseICat([6], [1.0])
end

function POMDPs.observation(M::Test_Random3, a, sp)
    s==2 && return SparseCat([2], [1.0])
    return SparseCat([1], [1.0])
end

POMDPs.reward(M::Test_Random3, s, a, sp) = reward(M,s,a)
function POMDPs.reward(M::Test_Random3, s, a)
    s == 3 && a == 1 && return 5.0
    s == 4 && a == 1 && return 1.0
    s == 5 && a == 2 && return 1.0
    return 0.0
end