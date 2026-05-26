function action_distr(policy::X, b) where X<:Policy
    a = POMDPs.action(policy, b)
    return SparseCat([a], [1.0])
end

function get_memory_type(policy::X) where X<:Policy
    return Nothing
end

function get_initial_memory(policy::X) where X<:Policy
    return nothing
end

function update_memory(policy::X, x, a, o) where X<:Policy
    return nothing
end

