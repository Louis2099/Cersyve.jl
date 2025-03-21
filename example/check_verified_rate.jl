using Revise
using Cersyve
using Flux
using JLD2
using Random

# task = Unicycle
# task = LaneKeep
# task = DoubleIntegrator
# task= Pendulum
# task = CartPole
# task = Quadrotor
# task = PointMass
# task = DoubleIntegrator2D
# task = Unicycle4D
# task = RobotArm
task = TwoLinkRobotArm


value_hidden_sizes = [32, 32]
dynamics_hidden_sizes = [32, 32]
constraint_hidden_sizes = [16]



# data_path = joinpath(@__DIR__, "../data/lane_keep_data.jld2")
# model_dir = joinpath(@__DIR__, "../model/lane_keep/")
# log_dir = joinpath(@__DIR__, "../log/lane_keep/")

# data_path = joinpath(@__DIR__, "../data/double_intergrator_data.jld2")
# model_dir = joinpath(@__DIR__, "../model/double_integrator/")
# log_dir = joinpath(@__DIR__, "../log/double_integrator/")

# data_path = joinpath(@__DIR__, "../data/unicycle_data.jld2")
# model_dir = joinpath(@__DIR__, "../model/unicycle/")
# log_dir = joinpath(@__DIR__, "../log/unicycle/")

# data_path = joinpath(@__DIR__, "../data/pendulum_data.jld2")
# model_dir = joinpath(@__DIR__, "../model/pendulum/")
# log_dir = joinpath(@__DIR__, "../log/pendulum/")

# data_path = joinpath(@__DIR__, "../data/cart_pole_data.jld2")
# model_dir = joinpath(@__DIR__, "../model/cart_pole/")
# log_dir = joinpath(@__DIR__, "../log/cart_pole/")

# data_path = joinpath(@__DIR__, "../data/quadrotor_data.jld2")
# model_dir = joinpath(@__DIR__, "../model/quadrotor/")
# log_dir = joinpath(@__DIR__, "../log/quadrotor/")

# data_path = joinpath(@__DIR__, "../data/point_mass_data.jld2")
# model_dir = joinpath(@__DIR__, "../model/point_mass/")
# log_dir = joinpath(@__DIR__, "../log/point_mass/")

# data_path = joinpath(@__DIR__, "../data/2D_double_integrator_data.jld2")
# model_dir = joinpath(@__DIR__, "../model/2D_double_integrator/")
# log_dir = joinpath(@__DIR__, "../log/2D_double_integrator/")

# data_path = joinpath(@__DIR__, "../data/unicycle4D_data.jld2")
# model_dir = joinpath(@__DIR__, "../model/unicycle4D/")
# log_dir = joinpath(@__DIR__, "../log/unicycle4D/")

# data_path = joinpath(@__DIR__, "../data/robot_arm_data.jld2")
# model_dir = joinpath(@__DIR__, "../model/robot_arm/")
# log_dir = joinpath(@__DIR__, "../log/robot_arm/")

data_path = joinpath(@__DIR__, "../data/2link_robot_arm_data.jld2")
model_dir = joinpath(@__DIR__, "../model/2link_robot_arm/")
log_dir = joinpath(@__DIR__, "../log/2link_robot_arm/")

# if not exist create folder
if !isdir(log_dir)
    mkdir(log_dir)
end

if !isdir(model_dir)
    mkdir(model_dir)
end

seed = 1

Random.seed!(seed)

# if !isfile(data_path)
#     collect_data(
#         task.x_low,
#         task.x_high,
#         task.u_low,
#         task.u_high,
#         task.dynamics,
#         task.terminated;
#         save_path=data_path,
#     )
# end


# non linear dynamic
# data = JLD2.load(data_path)["data"]
# f_model = Cersyve.create_mlp(task.x_dim + task.u_dim, task.x_dim, dynamics_hidden_sizes)
# # train_dynamics(data, f_model, log_dir=log_dir)
# Flux.loadmodel!(f_model, JLD2.load(joinpath(model_dir, "f.jld2"), "state"))
# f_pi_model = Cersyve.create_closed_loop_dynamics_model(
#     f_model, task.pi_model, data, task.x_low, task.x_high, task.u_dim)
# dynamics_model = Cersyve.create_non_linear_dynamics_model(f_model, data, task.x_low, task.x_high, task.u_dim, task.x_dim)


h_model = Cersyve.create_mlp(task.x_dim, 1, constraint_hidden_sizes)
# train_constraint(task.x_low, task.x_high, h_model, task.constraint, log_dir=log_dir)
# Flux.loadmodel!(h_model, JLD2.load(joinpath(model_dir, "h_double_circle.jld2"), "state"))
Flux.loadmodel!(h_model, JLD2.load(joinpath(model_dir, "h.jld2"), "state"))




x_a_low =  [task.x_low; task.u_low]
x_a_high = [task.x_high; task.u_high]


# affine_Q = create_parallel_affine_Q(task.x_dim, task.u_dim)
# affine_Q = create_mul_Q(task.x_dim, task.u_dim)
# affine_Q = create_lim_X_affine_Q(task.x_dim, task.u_dim)
# affine_Q = create_baseline_affine_Q(task.x_dim, task.u_dim)
affine_Q = create_x_mul_xu_Q(task.x_dim, task.u_dim)
# affine_Q = create_x_add_xu_Q(task.x_dim, task.u_dim)

model_path = "/home/jiaxingl/project/Cersyve.jl/log/2link_robot_arm/finetune_20250218_160454_x_mul_xu_tol1e-4_verified/Q_finetune_final.jld2"
Flux.loadmodel!(affine_Q, JLD2.load(model_path, "state"))
Q_h_model = create_Q_constraint_model(affine_Q, h_model, task)
Q_Q_prime, Q_max = create_Q_Q_max_prime(affine_Q, task.f_pi_model, task.f_model, task)
res = 4
# verify_rate(
#     x_a_low,
#     x_a_high,
#     task.x_dim,
#     resolution,
#     Q_h_model,
#     Q_Q_prime;
#     con_start_values = nothing,
#     inv_start_values = nothing,
#     save_file = "/home/jiaxingl/project/Cersyve.jl/log/2link_robot_arm/finetune_20250218_160454_x_mul_xu_tol1e-4_verified/verified_bounds_$(resolution).jld2"
# )

verify_safeset_rate(
    x_a_low,
    x_a_high,
    task.x_dim + task.u_dim,
    res,
    Q_h_model,
    Q_Q_prime;
    con_start_values = nothing,
    inv_start_values = nothing,
    save_file = "/home/jiaxingl/project/Cersyve.jl/log/2D_double_integrator/finetune_20250211_192942_hold/verified_safe_bounds_$(res).jld2"
)