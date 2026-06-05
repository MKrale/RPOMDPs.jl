struct Index_IPOMDP <: IPOMDP{Int64,Int64,Int64}
    T::Vector{SparseMatrixCSC{Interval{Float64}, Int}} # T[a][sp, s] as intervals
    R::Array{Float64,2} # R[s,a]
    O::Vector{SparseMatrixCSC{Float64, Int}} # O[a][o, sp] as probabilities
    isterminal::BitVector
    initialstate::Any # typically SparseICat or similar distribution over states with intervals
    discount::Float64
    Svals::Vector{Int64}
    Avals::Vector{Int64}
    Ovals::Vector{Int64}
end

function Index_IPOMDP(pomdp::IPOMDP)
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    O = ordered_observations(pomdp)

    terminal = _vectorized_terminal(pomdp, S)
    T = transition_matrix_a_sp_s(pomdp)
    R = _tabular_rewards(pomdp, S, A, terminal)
    Omat = observation_matrix_a_sp_o(pomdp)
    b0 = _vectorized_initialstate(pomdp, S)
    return Index_IPOMDP(T, R, Omat, terminal, b0, discount(pomdp), collect(1:length(S)), collect(1:length(A)), collect(1:length(O)))
end

# Explicit concrete constructor for index-based IPOMDPs (states/actions/obs are Int)
Index_IPOMDP(T::Vector{<:SparseMatrixCSC{Interval{Float64}, Int}}, R::AbstractMatrix{Float64}, Omat::Vector{<:SparseMatrixCSC{Float64, Int}}, isterm::BitVector, b0, disc::Float64, Svals::Vector{Int64}=Int64[], Avals::Vector{Int64}=Int64[], Ovals::Vector{Int64}=Int64[]) =
    Index_IPOMDP(T, Matrix{Float64}(R), Omat, isterm, b0, disc, Svals, Avals, Ovals)

function transition_matrix_a_sp_s(mdp::IPOMDP)
    S = ordered_states(mdp)
    A = ordered_actions(mdp)

    ns = length(S)
    na = length(A)
    transmat_row_A = [Int[] for _ in 1:na]
    transmat_col_A = [Int[] for _ in 1:na]
    transmat_data_A = [Vector{Interval{Float64}}() for _ in 1:na]

    for (si,s) in enumerate(S)
        for (ai,a) in enumerate(A)
            if isterminal(mdp, s)
                push!(transmat_row_A[ai], si)
                push!(transmat_col_A[ai], si)
                push!(transmat_data_A[ai], 1.0)
            else
                td = transition(mdp, s, a)
                for (sp, p) in weighted_iterator(td)
                    # convert to interval if needed
                    if isa(p, Number)
                        pint = interval(p)
                    else
                        pint = p
                    end
                    # include even degenerate-zero intervals if nonzero bounds
                    if sup(pint) > 0.0
                        spi = stateindex(mdp, sp)
                        push!(transmat_row_A[ai], spi)
                        push!(transmat_col_A[ai], si)
                        push!(transmat_data_A[ai], pint)
                    end
                end
            end
        end
    end
    return [sparse(transmat_row_A[a], transmat_col_A[a], transmat_data_A[a], ns, ns) for a in 1:na]
end

function observation_matrix_a_sp_o(mdp::IPOMDP)
    S = ordered_states(mdp)
    A = ordered_actions(mdp)
    O = ordered_observations(mdp)

    ns = length(S)
    na = length(A)
    no = length(O)

    obs_row_A = [Int[] for _ in 1:na]
    obs_col_A = [Int[] for _ in 1:na]
    obs_data_A = [Float64[] for _ in 1:na]

    for (sp_i, sp) in enumerate(S)
        for (ai, a) in enumerate(A)
            od = observation(mdp, a, sp)
            for (o, p) in weighted_iterator(od)
                prob = isa(p, Number) ? Float64(p) : mid(p)
                if prob > 0.0
                    oi = obsindex(mdp, o)
                    push!(obs_row_A[ai], oi)
                    push!(obs_col_A[ai], sp_i)
                    push!(obs_data_A[ai], prob)
                end
            end
        end
    end
    return [sparse(obs_row_A[a], obs_col_A[a], obs_data_A[a], no, ns) for a in 1:na]
end

function _tabular_rewards(pomdp, S, A, terminal)
    R = Matrix{Float64}(undef, length(S), length(A))
    for (s_idx, s) in enumerate(S)
        if terminal[s_idx]
            R[s_idx, :] .= 0.0
            continue
        end
        for (a_idx, a) in enumerate(A)
            R[s_idx, a_idx] = reward(pomdp, s, a)
        end
    end
    return R
end

function _vectorized_terminal(pomdp, S)
    term = BitVector(undef, length(S))
    @inbounds for i in eachindex(term, S)
        term[i] = isterminal(pomdp, S[i])
    end
    return term
end

function _vectorized_initialstate(pomdp, S)
    b0 = initialstate(pomdp)
    vals = support(b0)
    idxs = map(x -> stateindex(pomdp, x), vals)
    probs = map(x -> pdf(b0, x), vals)
    return SparseCat(idxs, probs)
end

# POMDPs interface for Index_IPOMDP
POMDPTools.ordered_states(pomdp::Index_IPOMDP) = pomdp.Svals
POMDPs.states(pomdp::Index_IPOMDP) = ordered_states(pomdp)
POMDPTools.ordered_actions(pomdp::Index_IPOMDP) = pomdp.Avals
POMDPs.actions(pomdp::Index_IPOMDP) = ordered_actions(pomdp)
POMDPTools.ordered_observations(pomdp::Index_IPOMDP) = pomdp.Ovals
POMDPs.observations(pomdp::Index_IPOMDP) = ordered_observations(pomdp)

POMDPs.discount(pomdp::Index_IPOMDP) = pomdp.discount
POMDPs.initialstate(pomdp::Index_IPOMDP) = pomdp.initialstate
POMDPs.isterminal(pomdp::Index_IPOMDP, s) = begin
    si = POMDPs.stateindex(pomdp, s)
    return pomdp.isterminal[si]
end

function POMDPs.transition(pomdp::Index_IPOMDP, s, a)
    si = POMDPs.stateindex(pomdp, s)
    ai = POMDPs.actionindex(pomdp, a)
    # column si of T[ai] (rows are sp indices)
    col = pomdp.T[ai][:, si]
    nz = findnz(col)
    if length(nz) == 2
        inds, vals = nz[1], nz[2]
    else
        inds, vals = nz[1], nz[3]
    end
    if isempty(inds)
        return SparseICat(Int[], Vector{Interval{Float64}}())
    end
    return SparseICat(inds, vals)
end

function POMDPs.observation(pomdp::Index_IPOMDP, a, sp)
    ai = POMDPs.actionindex(pomdp, a)
    spi = POMDPs.stateindex(pomdp, sp)
    col = pomdp.O[ai][:, spi]
    nz = findnz(col)
    if length(nz) == 2
        inds, vals = nz[1], nz[2]
    else
        inds, vals = nz[1], nz[3]
    end
    if isempty(inds)
        return SparseCat(Int[], Float64[])
    end
    return SparseCat(inds, Float64.(vals))
end

POMDPs.reward(pomdp::Index_IPOMDP, s::Int, a::Int) = pomdp.R[s, a]

function POMDPs.stateindex(pomdp::Index_IPOMDP, s)
    if isa(s, Integer)
        si = Int(s)
        if 1 <= si <= length(pomdp.Svals)
            return si
        end
    end
    idx = findfirst(x->x==s, pomdp.Svals)
    idx === nothing && throw(ErrorException("state not found in Index_IPOMDP"))
    return idx
end

function POMDPs.actionindex(pomdp::Index_IPOMDP, a)
    if isa(a, Integer)
        ai = Int(a)
        if 1 <= ai <= length(pomdp.Avals)
            return ai
        end
    end
    idx = findfirst(x->x==a, pomdp.Avals)
    idx === nothing && throw(ErrorException("action not found in Index_IPOMDP"))
    return idx
end

function POMDPs.obsindex(pomdp::Index_IPOMDP, o)
    if isa(o, Integer)
        oi = Int(o)
        if 1 <= oi <= length(pomdp.Ovals)
            return oi
        end
    end
    idx = findfirst(x->x==o, pomdp.Ovals)
    idx === nothing && throw(ErrorException("observation not found in Index_IPOMDP"))
    return idx
end

n_states(pomdp::Index_IPOMDP) = length(states(pomdp))
n_actions(pomdp::Index_IPOMDP) = length(actions(pomdp))
n_observations(pomdp::Index_IPOMDP) = length(observations(pomdp))