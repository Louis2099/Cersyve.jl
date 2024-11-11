using Cersyve
using Random
using PyCall

task = Unicycle
save_path = joinpath(@__DIR__, "../data/tilted_pendulum_data_1.5m.jld2")
#save_path = joinpath(@__DIR__, "../data/unicycle_data.jld2")
seed = 1

Random.seed!(seed)

collect_python_data(
    "/home/jiaxing/projects/Cersyve.jl/data/tilted_pendulum_data.h5",
    2,
    1,
    1500000,
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
