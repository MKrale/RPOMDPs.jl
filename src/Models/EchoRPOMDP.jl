using POMDPs, Distributions, IntervalArithmetic
export EchoRPOMDP 
mininterval = interval(0.001,0.999)


@kwdef mutable struct EchoRPOMDP <: IPOMDP{String, Int, Int} 
    discount::Float64               = 0.95
    breakchance::Interval           = interval(0.9,0.9)
end

# States
POMDPs.states(M::EchoRPOMDP) = ["x","y","nx","ny","bx","by","nb","sink"]
POMDPs.statetype(M::EchoRPOMDP) = String
POMDPs.stateindex(M::EchoRPOMDP, s) = findfirst(states(M) .== s)
POMDPs.actions(M::EchoRPOMDP) = 1:2 # Go, repair, flip
POMDPs.actiontype(M::EchoRPOMDP) = Int
POMDPs.actionindex(M::EchoRPOMDP, a) = a
POMDPs.observations(M::EchoRPOMDP) = 1:2 #x, y, z
POMDPs.obstype(M::EchoRPOMDP) = Int
POMDPs.obsindex(M::EchoRPOMDP, o) = o
POMDPs.discount(M::EchoRPOMDP) = M.discount
POMDPs.initialstate(M::EchoRPOMDP) = SparseCat(["nx", "ny"], [0.5, 0.5])
POMDPs.isterminal(M::EchoRPOMDP, s) = s=="sink"


function POMDPs.transition(M::EchoRPOMDP, s,a)
    # (a == 2 || s=="sink") && return SparseICat(["sink"], [interval(1.0)])
    if a == 2
        s == "x" && return SparseICat(["y"], [interval(1.0)])
        s == "y" && return SparseICat(["x"], [interval(1.0)])
        s == "bx" && return SparseICat(["by"], [interval(1.0)])
        s == "by" && return SparseICat(["bx"], [interval(1.0)])
        return SparseICat(["x", "y"], [0.5, 0.5])
    end
    s == "sink" && return SparseICat(["sink"], [interval(1.0)])
    s == "x" && return SparseICat(["nx"], [interval(1.0)])
    s == "nx" && return SparseICat(["x","bx"], [M.breakchance, mininterval])
    s == "y" && return SparseICat(["ny"], [interval(1.0)])
    s == "ny" && return SparseICat(["y","by"], [M.breakchance, mininterval])
    s == "nb" && return SparseICat(["bx", "by"], [mininterval, mininterval])
    s in ["bx","by"] && return SparseICat(["nb"], [interval(1.0)])
    println("transition not recognized! (s=$s, a=$a)")
end

function POMDPs.observation(M::EchoRPOMDP, a, sp)
    (sp=="x" || sp=="bx") && return Deterministic(1)
    (sp=="y" || sp=="by") && return Deterministic(2)
    sp in ["sink", "nx", "ny", "nb"] && return Deterministic(1)
    println("observation not recognized! (sp=$sp, a=$a)")
    # (sp=="z" || sp=="bz") && return Deterministic(3)
end

POMDPs.reward(M::EchoRPOMDP, s, a, sp) = reward(M,s,a)
function POMDPs.reward(M::EchoRPOMDP, s, a)::Float64
    if a == 2
        s in ["nx", "ny", "nb"] ? (return -1.0) : (return 0.0)
    end
    s in ["x","y"] && return 1.0
    s in ["bx","by"] && return 0.0
    s == "sink" && return 0.0
    # s in ["bx", "by"] && a==2 && return 100.0
    # s in ["bx", "by"] && return -1.0
    # return 0.0
    # return 1.0
end



# end