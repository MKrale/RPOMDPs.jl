# Models:

"""RPOMDP: most generic framework: not used."""
abstract type RPOMDP{S,A,O} <: POMDP{S,A,O} end
"""IPOMDP: Assumes transitions & observations are given as (independent) intervals"""
abstract type IPOMDP{S,A,O} <: RPOMDP{S,A,O} end 

# Policies

# Distribution over intervals

struct SparseICat{V,P} <: DiscreteUnivariateDistribution
    vals::AbstractVector{V} 
    probs::AbstractVector{P} 
end

Distributions.support(d::SparseICat) = d.vals

Distributions.pdf(d::SparseICat,x::Real) = thispdf(d,x)
Distributions.pdf(d::SparseICat,x) = thispdf(d,x)
function thispdf(d::SparseICat, x)
    idx = findfirst(map(el->el==x, d.vals))
    (!(idx isa Nothing) && idx <= length(d.probs) && d.vals[idx] == x) ? (return d.probs[idx]) : (return 0.0) #should be zero(eltype(probs))...
end

IDeterministic(s) = SparseICat([s],[interval(1)])

# TODO: add sparsetabular functions for R/IPOMDPs so that we only have to work with numbers!

function SafeSparseCat(vals, probs)
    vals_new = unique(vals)
    probs_new = []
    for v in vals_new
        push!(probs_new, sum(probs[findall(isequal(v), vals )]))
    end
    return SparseCat(vals_new, probs_new)
end