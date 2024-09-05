using Cersyve
using Random
using PyCall

task = Unicycle
save_path = joinpath(@__DIR__, "../data/dynamic_data.jld2")
seed = 1

Random.seed!(seed)

collect_python_data(
    "/home/jiaxing/projects/Cersyve.jl/data/dynamic_data.h5",
    9,
    7,
    1000000,
    0.01,
    save_path,
)

# collect_data(
#     task.x_low,
#     task.x_high,
#     task.u_low,
#     task.u_high,
#     task.dynamics,
#     task.terminated;
#     save_path=save_path,
# )
