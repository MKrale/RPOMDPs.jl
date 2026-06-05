using RPOMDPs
using POMDPs, POMDPTools, IntervalArithmetic, Test

include("./tiger.jl")

@testset "RPOMDPs.jl" begin

    ### Testing models
    rpomdp = ToyRPOMDP()
    T, Tp = transition(rpomdp, 4, 1), SparseICat([6,7], [interval(0.1, 0.9), interval(0.1, 0.9)])
    for v in T.vals
        @test isequal_interval(pdf(T,v), pdf(Tp, v))
    end
    # TODO: add other models

    ### Testing Approximation functions
    rpomdp = ToyRPOMDP()
    mid_pomdp = to_mid_POMDP(rpomdp)
    # rmdp_pomdp = to_rmdp_POMDP(rpomdp)
    maxent_pomdp = to_maxent_POMDP(rpomdp)
    Tp = SparseCat([6,7], [0.5, 0.5])

    T = transition(mid_pomdp, 4, 1)
    @test(Tp.vals == T.vals && Tp.probs == T.probs)

    T = transition(maxent_pomdp, 4, 1)
    @test(Tp.vals == T.vals && Tp.probs == T.probs)

    ### Testing robustifying POMDPs (transition & observation function)
    pomdp = TigerPOMDP()
    rpomdp = ConfidencePOMDP(pomdp, 0.1, AdditiveAbs())

    T = transition(rpomdp, true, 1)
    Tp = SparseICat([true, false], [interval(0.4,0.6), interval(0.4, 0.6)])
    for v in support(T)
        @test isequal_interval(pdf(T,v), pdf(Tp, v))
    end
    O = observation(rpomdp, 0, true)
    Op = SparseCat([true, false], [0.85, 0.15])
    for v in support(O)
        @test isapprox(pdf(O,v), pdf(Op, v))
    end

    ### Testing ExplicitRPOMDPs (transition & observation function)
    rpomdp = Index_IPOMDP(rpomdp)
    C = ModelSizes(rpomdp)
    @test (C.ns == 2); @test(C.na == 3); @test(C.no == 2)
    T = transition(rpomdp, 1, 2)
    Tp = SparseICat([1, 2], [interval(0.4,0.6), interval(0.4, 0.6)])
    for v in T.vals
        @test isequal_interval(pdf(T,v), pdf(Tp, v))
    end
    O = observation(rpomdp, 1, 1)
    Op = SparseCat([1, 2], [0.85, 0.15])
    for v in support(O)
        @test isapprox(pdf(O,v), pdf(Op, v))
    end
    @test isapprox(reward(rpomdp,1,3), 10.0)


    ### Testing Nature MDP
    policy = POMDPTools.Policies.FunctionPolicy(x->1)
    rpomdp = ToyRPOMDP()
    naturemdp = get_model_adversary(rpomdp, policy)
    s0 = Sn(nothing, 0, 1)
    @test (pdf(initialstate(naturemdp), s0) == 1.0)
    sp = Sn(nothing, 4, 1)
    @test (actions(naturemdp, sp) == [An([6,7], [0.1, 0.9]), An([6,7], [0.9, 0.1])])
    @test (pdf( transition(naturemdp, sp, An([6,7], [0.1, 0.9])), Sn(nothing, 6,1)) == 0.1)
    @test (pdf( transition(naturemdp, sp, An([6,7], [0.1, 0.9])), Sn(nothing, 7,1)) == 0.9)
end