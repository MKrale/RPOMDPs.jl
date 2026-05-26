"""Struct containing parameter vectors & sizes, to prevent calling (possibly expensive) POMDP functions"""
struct C
    S; A; O 
    ns; na; no
    S_dict
end

get_constants(model) = C( states(model), actions(model), observations(model),
                         length(states(model)), length(actions(model)), length(observations(model)),
                         Dict( zip(states(model), 1:length(states(model)))))

function add_to_dict!(dict, key, value; func=+, minvalue=0)
    if haskey(dict, key)
        dict[key] = func(dict[key], value)
    elseif isnothing(minvalue) || value > minvalue
        dict[key] = value
    end
end