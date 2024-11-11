using Cersyve
using JLD2
using Random

task = Unicycle
hidden_sizes = [32, 32]
#data_path = joinpath(@__DIR__, "../data/dynamic_data.jld2")
data_path = joinpath(@__DIR__, "../data/tilted_pendulum_data_1.5m.jld2")
log_dir = joinpath(@__DIR__, "../log/tilted_pendulum_1.5m/")
seed = 1

Random.seed!(seed)

data = JLD2.load(data_path)["data"]

obs_dim = 2
act_dim = 1
f_model = Cersyve.create_mlp(obs_dim + act_dim, obs_dim, hidden_sizes)
#f_model = Cersyve.create_mlp(task.x_dim + task.u_dim, task.x_dim, hidden_sizes)
# x[end, goal, hazard]
# u[joint_pos*7]
# x_high = Float32[0.9091, 0.9091, 0.9091, 0.9091, 0.9091, 0.2, 0.9091, 0.9091, 0.2]
# x_low = Float32[0, 0, 0, 0.3, 0.3, 0, 0.3, 0.3, 0]
# u_high = Float32[3.14159265359, 2.2497294058206907, 3.14159265359, 2.5795966344476193, 3.14159265359, 2.0996310901491784, 3.14159265359]
# u_low = Float32[-3.14159265359, -2.2497294058206907, -3.14159265359, -2.5795966344476193, -3.14159265359, -2.0996310901491784, -3.14159265359]
x_high = Float32[pi, 8.0]
x_low = Float32[-pi, -8.0]

u_high = Float32[2.0]
u_low = Float32[-2.0]

train_dynamics(
    data,
    f_model;
    penalty="APA",
    space_size=[x_high; u_high] - [x_low; u_low],
    apa_coef=0.01,
    epoch_num = 400,
    log_dir=log_dir,
)
