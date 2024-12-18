using Revise
using Cersyve
using Flux
using JLD2
using Random
using ModelVerification





task = Unicycle
value_hidden_sizes = [32, 32]
dynamics_hidden_sizes = [32, 32]
constraint_hidden_sizes = [16]
data_path = joinpath(@__DIR__, "../data/unicycle_data.jld2")
model_dir = joinpath(@__DIR__, "../model/unicycle/")
log_dir = joinpath(@__DIR__, "../log/unicycle/")
seed = 1

Random.seed!(seed)


data = JLD2.load(data_path)["data"]
f_model = Cersyve.create_mlp(task.x_dim + task.u_dim, task.x_dim, dynamics_hidden_sizes)
Flux.loadmodel!(f_model, JLD2.load(joinpath(model_dir, "f.jld2"), "state"))
f_pi_model = Cersyve.create_closed_loop_dynamics_model(
    f_model, task.pi_model, data, task.x_low, task.x_high, task.u_dim)

h_model = Cersyve.create_mlp(task.x_dim, 1, constraint_hidden_sizes)
Flux.loadmodel!(h_model, JLD2.load(joinpath(model_dir, "h.jld2"), "state"))

x_a_low =  [task.x_low; task.u_low]
x_a_high = [task.x_high; task.u_high]

fun_affine_Q = create_func_parallel_affine_Q(task.x_dim, task.u_dim)
affine_Q = create_parallel_affine_Q(task.x_dim, task.u_dim)
Flux.loadmodel!(fun_affine_Q, JLD2.load(joinpath(model_dir, "Q_pretrain.jld2"), "state"))

affine_Q[1].layers[1].layers[2].weight .= fun_affine_Q[1].layers[1].layers[2].weight
affine_Q[1].layers[1].layers[2].bias .= fun_affine_Q[1].layers[1].layers[2].bias
affine_Q[1].layers[1].layers[3].weight .= fun_affine_Q[1].layers[1].layers[3].weight
affine_Q[1].layers[1].layers[3].bias .= fun_affine_Q[1].layers[1].layers[3].bias
affine_Q[2].layers[1].weight .= fun_affine_Q[2].layers[1].weight
affine_Q[2].layers[1].bias .= fun_affine_Q[2].layers[1].bias

finetune_Q(
    task, 
    affine_Q,
    f_pi_model,
    h_model,
    x_a_low,
    x_a_high;
    log_dir=log_dir,
)
