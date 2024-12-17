using Revise
using Cersyve
using Flux
using JLD2
using Random
using ModelVerification





function create_parallel_affine_Q(x_dim, u_dim)
    # Assume the input has 13 elements: x (0–7), u (8–13)
    # x_w = create_filter_matrix(1, x_dim, x_dim+u_dim)
    # u_w = create_filter_matrix(x_dim+1, x_dim+u_dim, x_dim+u_dim)
    # [input_dim, batch_size]
    function filter_x(input)
        return input[1:x_dim, :]
    end
    
    function filter_u(input)
        return input[x_dim+1:end, :]
    end

    #Branch1
    b1 = Chain(
        filter_x,  
        Dense(x_dim, 32, relu),  
        Dense(32, 32, relu)  
    )

    # Define the final output layer (scalar output)
    final_layer = Chain(Dense(32 + u_dim, 1))  # Concatenation of x (32) and u (6)

    # Complete model
    model = Chain(
        Parallel(
            vcat, 
            b1,
            filter_u
        ),
        final_layer                       # Compute scalar output
    )
    return model
end

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


affine_Q = create_parallel_affine_Q(task.x_dim, task.u_dim)
Flux.loadmodel!(affine_Q, JLD2.load(joinpath(model_dir, "Q_pretrain.jld2"), "state"))


finetune_Q(
    task, 
    affine_Q,
    f_pi_model,
    h_model,
    x_a_low,
    x_a_high;
    log_dir=log_dir,
)
