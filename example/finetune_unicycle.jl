using Cersyve
using Flux
using JLD2
using Random
using ModelVerification

task = Unicycle
value_hidden_sizes = [32, 32]
dynamics_hidden_sizes = [32, 32]
constraint_hidden_sizes = [16]
data_path = joinpath(@__DIR__, "../data/dynamic_data.jld2")
model_dir = joinpath(@__DIR__, "../model/kinova/")
log_dir = joinpath(@__DIR__, "../log/kinova/")
seed = 1
x_high = Float32[0.9091, 0.9091, 0.9091, 0.9091, 0.9091, 0.2, 0.9091, 0.9091, 0.2]
x_low = Float32[0, 0, 0, 0.3, 0.3, 0, 0.3, 0.3, 0]
u_high = Float32[3.14159265359, 2.2497294058206907, 3.14159265359, 2.5795966344476193, 3.14159265359, 2.0996310901491784, 3.14159265359]
u_low = Float32[-3.14159265359, -2.2497294058206907, -3.14159265359, -2.5795966344476193, -3.14159265359, -2.0996310901491784, -3.14159265359]
x_dim = 9
u_dim = 7

Random.seed!(seed)

# V_model = Cersyve.create_mlp(task.x_dim, 1, value_hidden_sizes)
# Flux.loadmodel!(V_model, JLD2.load(joinpath(model_dir, "V_pretrain.jld2"), "state"))
V_model = ModelVerification.build_flux_model(joinpath(model_dir, "vc_net_b.onnx"))
pi_model = ModelVerification.build_flux_model(joinpath(model_dir, "pi_net.onnx"))

struct EnsembleModel
    policy_network::Chain
    value_network::Chain
end

function EnsembleModel(policy_network, value_network)    
    model = EnsembleModel(policy_network, value_network)
    return model
end

function (model::EnsembleModel)(observation)
    # Pass observation through policy network to get action vector
    action_vector = model.policy_network(observation)
    
    # Concatenate observation and action vector
    combined_input = cat(observation, action_vector; dims=1)
    
    # Pass the concatenated input to the value network
    value = model.value_network(combined_input)
    
    return value
end

ensemble_V = EnsembleModel(pi_model, V_model)

data = JLD2.load(data_path)["data"]
#f_model = Cersyve.create_mlp(task.x_dim + task.u_dim, task.x_dim, dynamics_hidden_sizes)
f_model = Cersyve.create_mlp(x_dim + u_dim, x_dim, dynamics_hidden_sizes)
Flux.loadmodel!(f_model, JLD2.load(joinpath(model_dir, "f.jld2"), "state"))

# f_pi_model = Cersyve.create_closed_loop_dynamics_model(
#     f_model, task.pi_model, data, task.x_low, task.x_high, task.u_dim)

f_pi_model = Cersyve.create_closed_loop_dynamics_model(
    f_model, pi_model, data, x_low, x_high, u_dim)

h_model = Cersyve.create_mlp(x_dim, 1, constraint_hidden_sizes)
Flux.loadmodel!(h_model, JLD2.load(joinpath(model_dir, "h.jld2"), "state"))

finetune_value(
    ensemble_V,
    f_pi_model,
    h_model,
    x_low,
    x_high;
    log_dir=log_dir,
)
