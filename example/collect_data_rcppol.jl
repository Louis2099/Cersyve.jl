using Cersyve
using Random
using PyCall

task = Unicycle
# save_path = joinpath(@__DIR__, "../model/simple_pendulum/tilted_pendulum_data_1.5m.jld2")
save_path = "/home/jiaxingl/project/Cersyve.jl/model/simple_pendulum/dynamic_data.jld2"
seed = 1

Random.seed!(seed)

collect_python_data(
    "/home/jiaxingl/project/Cersyve.jl/model/simple_pendulum/dynamic_data.h5",
    3,
    1,
    800000,
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
