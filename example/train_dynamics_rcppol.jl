using Cersyve
using JLD2
using Random


hidden_sizes = [32, 32]
seed = 1

# pendulum environment
# data_path = joinpath(@__DIR__, "../data/tilted_pendulum_data_1.5m.jld2")
data_path = "/home/jiaxingl/project/Cersyve.jl/model/simple_pendulum/dynamic_data.jld2"
# log_dir = joinpath(@__DIR__, "../log/tilted_pendulum_1.5m/")
log_dir = "/home/jiaxingl/project/Cersyve.jl/model/simple_pendulum"


x_high = Float32[1, 1, 5.0]
x_low = Float32[-1, -1, -5.0]
u_high = Float32[3.0]
u_low = Float32[-3.0]

obs_dim = 3
act_dim = 1
# other environment


Random.seed!(seed)
data = JLD2.load(data_path)["data"]
f_model = Cersyve.create_mlp(obs_dim + act_dim, obs_dim, hidden_sizes)
train_dynamics(
    data,
    f_model;
    penalty="APA",
    space_size=[x_high; u_high] - [x_low; u_low],
    apa_coef=0.01,
    epoch_num = 400,
    log_dir=log_dir,
)
