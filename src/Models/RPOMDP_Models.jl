"""
Collection of a number of RPOMDP models, including the benchmark models Toy, Echo and Parity from Krale et. al. (2025)
"""

include("ToyRPOMDP.jl")
include("ChainRPOMDP.jl")
include("EchoRPOMDP.jl")
include("Test.jl")
include("CNC_Detection.jl")

export ToyRPOMDP, ToyRPOMDP_mid, ToyRPOMDP_rmdp
export ChainRPoMDPInf, ChainRPOMDPN
export EchoRPOMDP
export Test_Backup, Test_Random, Test_Random3
export CNC_Detection