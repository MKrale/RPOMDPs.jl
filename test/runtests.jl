using RPOMDPs
using POMDPs, POMDPTools, IntervalArithmetic, Test

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