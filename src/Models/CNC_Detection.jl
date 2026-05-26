HEALTHY, SMALL_POL, LARGE_POL, LOC_CRC_Y1, LOC_CRC_Y2, REG_CRC_Y1, REG_CRC_Y2, D_CRC, SINK = 1:9
HIDDEN, DETECTED = 1:2
NOOP, COLO, FIT = 1:3
AGES = 50:70

@kwdef struct CNC_Detection <: IPOMDP{Tuple{Int,Int}, Int, Int}
    discount = 0.99
end

POMDPs.states(M::CNC_Detection) = vec([(h,d,y) for h in 1:SINK, d in [HIDDEN, DETECTED], y in AGES])
# POMDPs.states(M::CNC_Detection) = vec([(h,d) for h in 1:SINK, d in [HIDDEN, DETECTED]])
POMDPs.statetype(M::CNC_Detection) = Tuple{Int, Int, Int}
POMDPs.stateindex(M::CNC_Detection, s) = findfirst(isequal(s), states(M))
# POMDPs.stateindex(M::HeavenOrHell, s) = (first(s)+1) + (maxstate(M)+1) * (last(s)-1)
POMDPs.actions(M::CNC_Detection) = [NOOP, COLO, FIT] # LEFT, RIGHT
POMDPs.actiontype(M::CNC_Detection) = Int 
POMDPs.actionindex(M::CNC_Detection, a) = findfirst(isequal(a), actions(M))
POMDPs.observations(M::CNC_Detection) = 1:SINK
POMDPs.obstype(M::CNC_Detection) = Int
POMDPs.obsindex(M::CNC_Detection, o) = o
POMDPs.discount(M::CNC_Detection) = M.discount
POMDPs.initialstate(M::CNC_Detection) = SparseCat([(HEALTHY, HIDDEN, 50)], [1.0])
# POMDPs.initialstate(M::CNC_Detection) = SparseCat([(HEALTHY, DETECTED)], [1.0])
POMDPs.isterminal(M::CNC_Detection, s) = s[1] in [SINK] || s[3] >= maximum(AGES)
# POMDPs.isterminal(M::CNC_Detection, s) = s[1] in [SINK]


function POMDPs.transition(M::CNC_Detection, s, a)
    healthstate, detected, age = s

    detected == DETECTED && (healthstate = Detected_trans[healthstate])
    h_distr = Rates[healthstate]
    age = min(age + 1, maximum(AGES))

    ss, probs = [], []  
    for h in support(h_distr)
        hp = pdf(h_distr, h)
        dpmin, dpmax = inf(Detection_rates[a][h]), sup(Detection_rates[a][h])

        push!(ss, (h, DETECTED, age))
        push!(probs, interval(hp * dpmin, hp * dpmax))

        push!(ss, (h, HIDDEN, age))
        push!(probs, interval(hp * (1-dpmax), hp * (1-dpmin)))

    end
    return SparseICat(ss, probs)
end

function POMDPs.observation(M::CNC_Detection, a,sp)
    detected = sp[2]
    sp[2] == DETECTED && return Deterministic(sp[1])
    return Deterministic(HEALTHY)
end

function POMDPs.reward(M::CNC_Detection, s, a)
    r = 0.0
    a == FIT && (r -= 22.0)
    a == COLO && (r -= 920.0)
    h, d = s[1], s[2]
    h in [LOC_CRC_Y1, LOC_CRC_Y2] && d==DETECTED && (r -= 51_000.0)
    h in [REG_CRC_Y1, REG_CRC_Y2] && d==DETECTED && (r -= 98_000.0)
    h == D_CRC && (r -= 200_000.0)
    h == SINK && (r=0.0)
    return r
end

### DATA ###

Rates = Dict(
    HEALTHY => SparseCat([HEALTHY, SMALL_POL, LOC_CRC_Y1], [0.97, 0.02, 0.01]),
    SMALL_POL => SparseCat([SMALL_POL, LARGE_POL],[0.98,0.02]),
    LARGE_POL => SparseCat([LARGE_POL, LOC_CRC_Y1], [0.95, 0.05]),
    LOC_CRC_Y1 => SparseCat([LOC_CRC_Y2], [1.0]),
    LOC_CRC_Y2 => SparseCat([REG_CRC_Y1], [1.0]),
    REG_CRC_Y1 => SparseCat([REG_CRC_Y2], [1.0]),
    REG_CRC_Y2 => SparseCat([D_CRC], [1.0]),

    # LOC_CRC_TRT => SparseCat([LOC_CRC_TRT, SINK], [0.98, 0.02]),
    # REG_CRC_TRT => SparseCat([REG_CRC_TRT, SINK], [0.91, 0.09]),
    D_CRC => SparseCat([D_CRC, SINK], [0.5, 0.5]),
    SINK => SparseCat([SINK], [1.0])
)

Detected_trans = Dict(
    HEALTHY => HEALTHY,
    SMALL_POL => HEALTHY,
    LARGE_POL => HEALTHY,
    LOC_CRC_Y1 => SINK,
    LOC_CRC_Y2 => SINK,
    REG_CRC_Y1 => SINK,
    REG_CRC_Y2 => SINK,
    D_CRC => D_CRC,

    # LOC_CRC_TRT => SINK,
    # REG_CRC_TRT => SINK,
    SINK => SINK,
)

cnc_interval_fit = interval(0.55, 0.76)
cnc_interval_colon = interval(0.9, 0.97)

Detection_rates = Dict(
    NOOP => Dict(
        HEALTHY => interval(0.0),
        SMALL_POL => interval(0.0),
        LARGE_POL => interval(0.0),
        LOC_CRC_Y1 => interval(0.22),
        LOC_CRC_Y2 => interval(0.22),
        REG_CRC_Y1 => interval(0.4),
        REG_CRC_Y2 => interval(0.4),

        D_CRC => interval(1.0),
        # LOC_CRC_TRT => interval(1.0),
        # REG_CRC_TRT => interval(1.0),
        SINK => interval(1.0),
    ),
    FIT => Dict(
        HEALTHY => interval(0.0),
        SMALL_POL => interval(0.0),
        LARGE_POL => interval(0.17, 0.23),
        LOC_CRC_Y1 => cnc_interval_fit,
        LOC_CRC_Y2 => cnc_interval_fit,
        REG_CRC_Y1 => cnc_interval_fit,
        REG_CRC_Y2 => cnc_interval_fit,
        D_CRC => cnc_interval_fit,
    
        # LOC_CRC_TRT => interval(1.0),
        # REG_CRC_TRT => interval(1.0),
        SINK => interval(1.0),
    ),
    COLO => Dict(
        HEALTHY => interval(0.0),
        SMALL_POL => interval(0.8, 0.9),
        LARGE_POL => interval(0.85, 0.95),
        LOC_CRC_Y1 => cnc_interval_colon,
        LOC_CRC_Y2 => cnc_interval_colon,
        REG_CRC_Y1 => cnc_interval_colon,
        REG_CRC_Y2 => cnc_interval_colon,
        D_CRC => cnc_interval_colon,
    
        # LOC_CRC_TRT => interval(1.0),
        # REG_CRC_TRT => interval(1.0),
        SINK => interval(1.0),
    )
)

