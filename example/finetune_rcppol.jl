using Cersyve
using Flux
using JLD2
using Random
using ModelVerification

task = Unicycle
seed = 1

value_hidden_sizes = [64, 64]
dynamics_hidden_sizes = [32, 32]
constraint_hidden_sizes = [16]

# data_path = joinpath(@__DIR__, "../data/tilted_pendulum_data_1.5m.jld2")
# model_dir = joinpath(@__DIR__, "../model/tilted_pendulum/")
# log_dir = joinpath(@__DIR__, "../log/tilted_pendulum_1.5m/")

data_path = "/home/jiaxingl/project/Cersyve.jl/model/simple_pendulum/dynamic_data.jld2"
model_dir = "/home/jiaxingl/project/Cersyve.jl/model/simple_pendulum"
log_dir = "/home/jiaxingl/project/Cersyve.jl/model/simple_pendulum"


x_high = Float32[1, 1, 5.0]
x_low = Float32[-1, -1, -5.0]
u_high = Float32[3.0]
u_low = Float32[-3.0]
x_dim = 3
u_dim = 1

# x_high = Float32[0.9091, 0.9091, 0.9091, 0.9091, 0.9091, 0.2, 0.9091, 0.9091, 0.2]
# x_low = Float32[0, 0, 0, 0.3, 0.3, 0, 0.3, 0.3, 0]
# u_high = Float32[3.14159265359, 2.2497294058206907, 3.14159265359, 2.5795966344476193, 3.14159265359, 2.0996310901491784, 3.14159265359]
# u_low = Float32[-3.14159265359, -2.2497294058206907, -3.14159265359, -2.5795966344476193, -3.14159265359, -2.0996310901491784, -3.14159265359]
# x_dim = 9
# u_dim = 7

Random.seed!(seed)

function check_model(model)
    # Get all parameters of the model
    params = Flux.params(model)
    
    # Check for NaN values
    for (i, p) in enumerate(params)
        if any(isnan, p)
            println("Parameter $i contains NaN values.")
            println(size(p))
            return false
        end
    end
    
    println("All parameters are valid (no NaN values).")
    return true
end

# Load models
V_model = ModelVerification.build_flux_model(joinpath(model_dir, "Vc_net.onnx"))
pi_model = ModelVerification.build_flux_model(joinpath(model_dir, "pi_net.onnx"))


######################################################
# Structure of EnsembleModel, cannot fit into verification pipeline
# struct EnsembleModel
#     policy_network::Chain
#     value_network::Chain
# end

# function EnsembleModel(policy_network, value_network)    
#     model = EnsembleModel(policy_network, value_network)
#     return model
# end

# function (model::EnsembleModel)(observation)
#     # Pass observation through policy network to get action vector
#     action_vector = model.policy_network(observation)
    
#     # Concatenate observation and action vector
#     combined_input = cat(observation, action_vector; dims=1)
    
#     # Pass the concatenated input to the value network
#     value = model.value_network(combined_input)
    
#     return value
# end
# ensemble_V = EnsembleModel(pi_model, V_model)
######################################################

# Chain EnsembleModel
#ensemble_V = Chain(ob->cat(pi_model(ob), ob, dims=1), V_model)

# Parallel EnsembleModel
# x_expend_matrix = Matrix{Float32}(zeros(x_dim+u_dim, x_dim))
# for i in 1:x_dim
#     x_expend_matrix[i, i] = 1
# end
# for row in eachrow(x_expend_matrix)
#     println(row)
# end


# u_expend_matrix = Matrix{Float32}(zeros(x_dim+u_dim, u_dim))
# for i in x_dim+1:x_dim+u_dim
#     u_expend_matrix[i, i-x_dim] = 1
# end
# for row in eachrow(u_expend_matrix)
#     println(row)
# end

# ensemble_V = Chain(
#     Parallel(+, 
#         Dense(x_expend_matrix, zeros(Float32, x_dim+u_dim)),
#         Chain(pi_model, 
#             Dense(u_expend_matrix, zeros(Float32, x_dim+u_dim))
#             )
#         ), 
#     V_model)
# Function to check for NaN in model weights


data = JLD2.load(data_path)["data"]
# println("Checking data")    
# println("Checking x_std")    
# println(data["x_std"])
# println("Checking u_std")    
# println(data["u_std"])
# println("Checking dx_std")
# println(data["dx_std"])
# println("Checking x_mean")
# println(data["x_mean"])
# println("Checking u_mean")
# println(data["u_mean"])
# println("Checking dx_mean")
# println(data["dx_mean"])    

f_model = Cersyve.create_mlp(x_dim + u_dim, x_dim, dynamics_hidden_sizes)
Flux.loadmodel!(f_model, JLD2.load(joinpath(model_dir, "f.jld2"), "state"))
# println("Checking f_model")
# check_model(f_model)


f_pi_model = Cersyve.create_closed_loop_dynamics_model(
    f_model, pi_model, data, x_low, x_high, u_dim)
# println("Checking f_pi_model")
# check_model(f_pi_model)


h_model = Cersyve.create_mlp(x_dim, 1, constraint_hidden_sizes)
Flux.loadmodel!(h_model, JLD2.load(joinpath(model_dir, "h.jld2"), "state"))



finetune_value(
    V_model,
    #ensemble_V,
    f_pi_model,
    h_model,
    x_low,
    x_high;
    log_dir=log_dir,
)
