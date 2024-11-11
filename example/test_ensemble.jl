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

######################################################
# Structure of EnsembleModel, cannot fit into verification pipeline
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

struct_ensemble_V = EnsembleModel(pi_model, V_model)
######################################################

# Chain EnsembleModel
chain_ensemble_V = Chain(ob->cat(pi_model(ob), ob, dims=1), V_model)

# Parallel EnsembleModel
x_expend_matrix = Matrix{Float32}(zeros(x_dim+u_dim, x_dim))
for i in 1:x_dim
    x_expend_matrix[i, i] = 1
end
for row in eachrow(x_expend_matrix)
    println(row)
end


u_expend_matrix = Matrix{Float32}(zeros(x_dim+u_dim, u_dim))
for i in x_dim+1:x_dim+u_dim
    u_expend_matrix[i, i-x_dim] = 1
end
for row in eachrow(u_expend_matrix)
    println(row)
end

Parallel_ensemble_V = Chain(
    Parallel(+, 
        Dense(x_expend_matrix, zeros(Float32, x_dim+u_dim)),
        Chain(pi_model, 
            Dense(u_expend_matrix, zeros(Float32, x_dim+u_dim))
            )
        ), 
    V_model)

println(typeof(struct_ensemble_V.value_network))
println(typeof(chain_ensemble_V[2]))
println(typeof(Parallel_ensemble_V[2]))


function value_loss_fn(V_model)
    loss = sum(V_model(x_high))
    return loss
end
lr = 1e-4
struct_opt_state = Flux.setup(Adam(lr), struct_ensemble_V.value_network)
chain_opt_state = Flux.setup(Adam(lr), chain_ensemble_V[2])
parallel_opt_state = Flux.setup(Adam(lr), Parallel_ensemble_V[2])

s_loss, s_grad = Flux.withgradient(value_loss_fn, struct_ensemble_V)
c_loss, c_grad = Flux.withgradient(value_loss_fn, chain_ensemble_V)
p_loss, p_grad = Flux.withgradient(value_loss_fn, Parallel_ensemble_V)

# println(typeof(c_grad))
# println(length(c_grad[1][1][2]))

println(typeof(c_grad[1][1][2]))
println(size(c_grad[1][1][2][1][1][1]))

println(typeof(p_grad[1][1][2]))
println(size(p_grad[1][1][2][1][1][1]))

# function get_tuple_inside(tuple)
#     count = 0
#     current_tuple = tuple
#     while true
#         if length(current_tuple) != 1
#             println(length(current_tuple))
#             println(typeof(current_tuple))
#             break
#         else
#             print("[1]")
#             current_tuple = current_tuple[1]
#         end
#         count += 1
#     end
# end
# get_tuple_inside(c_grad)


# value_grad = grad[1]
# if isnothing(value_grad)
#     println(grad)
#     continue
# end
# value_grad = value_grad[:value_network]
# Flux.update!(opt_state, V_model.value_network, value_grad)