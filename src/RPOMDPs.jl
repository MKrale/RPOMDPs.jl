module RPOMDPs
    using POMDPs, POMDPTools, QuickPOMDPs, Random, IntervalArithmetic, Distributions, JuMP, Clp, Gurobi, SparseArrays, Memoize, LRUCache, Combinatorics

    # Surpressing Gurobis printing
 
    const GRB_ENV = Ref{Gurobi.Env}()
    function __init__()
        oldstd = stdout
        redirect_stdout(devnull)
        GRB_ENV[] = Gurobi.Env()
        redirect_stdout(oldstd)
        return
    end
    
    include("Utils.jl")
    export C, get_constants, add_to_dict!
    include("rpomdps.jl")
    export RPOMDP, IPOMDP, SparseICat, SafeSparseCat, IDeterministic
    include("RobustPolicies.jl")
    export get_action_probs, get_memory_type, get_initial_memory, update_memory
    include("approximate.jl")
    export to_mid_POMDP, to_rmdp_POMDP, to_maxent_POMDP
    include("adversarialnature.jl")
    export get_model_adversary, An, Sn
    include("makeuncertain.jl")
    export RobustifiedPOMDP, ConfidencePOMDP,
    IntervalType, AdditiveAbs, AdditiveRel, Multiplicative 
    include("Models/RPOMDP_Models.jl")
end

