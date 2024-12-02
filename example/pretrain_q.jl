using Cersyve
using Flux
using JLD2
using Random

struct FilterX
    W::Matrix  # Weight matrix
end

struct FilterU
    W::Matrix  # Weight matrix
end

function (layer::FilterX)(input::Matrix{Float32})
    return layer.W * input
end

Flux.@functor FilterX  # Make the layer compatible with Flux
function Flux.params(layer::FilterX)
    return Flux.Params([])  # Exclude weights from being trainable
end

# Define a filtering layer for extracting u (indices 9 to 14)


function (layer::FilterU)(input::Matrix{Float32})
    return layer.W * input
end

Flux.@functor FilterU  # Make the layer compatible with Flux
function Flux.params(layer::FilterU)
    return Flux.Params([])  # Exclude weights from being trainable
end

# Initialize the fixed weight matrices for filtering
function create_filter_matrix(start_idx, end_idx, total_len)
    W = zeros(end_idx - start_idx + 1, total_len)
    for i in start_idx:end_idx
        W[i - start_idx + 1, i] = 1.0
    end
    return W
end

function create_affine_Q(x_dim, u_dim)
    # Assume the input has 13 elements: x (0–7), u (8–13)
    x_filter = FilterX(create_filter_matrix(1, x_dim, x_dim+u_dim))  # Extract x
    u_filter = FilterU(create_filter_matrix(x_dim, x_dim+u_dim-1, x_dim+u_dim))  # Extract u

    # Define the branch1 network (process x)
    branch1 = Chain(
        Dense(x_dim, 32, relu),  # First hidden layer (32 neurons, input size is 8 for x)
        Dense(32, 32, relu)  # Second hidden layer (32 neurons)
    )

    # Define the final output layer (scalar output)
    final_layer = Chain(Dense(32 + u_dim, 1))  # Concatenation of x (32) and u (6)

    # Complete model
    model = Chain(
        x -> (x_filter(x), u_filter(x)),  # Apply the filters to extract x and u
        x -> (branch1(x[1]), x[2]),       # Process x through branch1, keep u unchanged
        x -> vcat(x[1], x[2]),            # Concatenate outputs of branch1 and u
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

# V_model = Cersyve.create_mlp(task.x_dim, 1, value_hidden_sizes)
Q_model = Cersyve.create_mlp(task.x_dim + task.u_dim, 1, value_hidden_sizes)


data = JLD2.load(data_path)["data"]
f_model = Cersyve.create_mlp(task.x_dim + task.u_dim, task.x_dim, dynamics_hidden_sizes)
Flux.loadmodel!(f_model, JLD2.load(joinpath(model_dir, "f.jld2"), "state"))
f_pi_model = Cersyve.create_closed_loop_dynamics_model(
    f_model, task.pi_model, data, task.x_low, task.x_high, task.u_dim)

h_model = Cersyve.create_mlp(task.x_dim, 1, constraint_hidden_sizes)
Flux.loadmodel!(h_model, JLD2.load(joinpath(model_dir, "h.jld2"), "state"))

x_a_low =  [task.x_low; task.u_low]
x_a_high = [task.x_high; task.u_high]


affine_Q = create_affine_Q(task.x_dim, task.u_dim)

pretrain_Q(
    affine_Q,
    f_pi_model,
    task.pi_model,
    h_model,
    task.x_low,
    task.x_high;
    penalty="APA",
    space_size=x_a_high - x_a_low,
    apa_coef=1e-4,
    log_dir=log_dir,
)


