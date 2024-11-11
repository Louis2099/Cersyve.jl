using JLD2
using Flux
using ONNX
using PyCall
pickle = pyimport("pickle")



jld_path = "/home/jiaxing/projects/Cersyve.jl/log/tilted_pendulum/finetune_20241014_154924/V_finetune.jld2"
onnx_model_path = "/home/jiaxing/projects/Cersyve.jl/log/tilted_pendulum/finetune_20241014_154924/model.onnx"
pkl_path = "/home/jiaxing/projects/Cersyve.jl/log/tilted_pendulum/finetune_20241014_154924/V_finetune.pkl"
input_dim = 2
output_dim = 1
hidden_size = 64
# Load the model from the JLD2 file

model_data = JLD2.load(jld_path)

# Assuming the model's weights or the full model are saved under a specific key in the JLD2 file
model_weights = model_data["state"][1]  # Replace with the actual key where your model or weights are stored
println(typeof(model_weights))
# Define your model architecture: V
weights_dict = Dict(
    "Dense_1" => (model_weights[1][1], model_weights[1][2]),
    "Dense_2" => (model_weights[2][1], model_weights[2][2]),
    "Dense_3" => (model_weights[3][1], model_weights[3][2])
)


open(pkl_path, "w") do f
    pickle.dump(weights_dict, f)
end
# model = Chain(
#     # Dense(input_dim, hidden_size, relu, model_weights[1][1], model_weights[1][2]),
#     # Dense(hidden_size, hidden_size, relu, model_weights[2][1], model_weights[2][2]),
#     # Dense(hidden_size, output_dim, model_weights[3][1], model_weights[3][2])
#     Dense(model_weights[1][1], model_weights[1][2], relu),
#     Dense(model_weights[2][1], model_weights[2][2], relu),
#     Dense(model_weights[3][1], model_weights[3][2])
# )

# Load the weights into the model (you need to adjust this depending on how your weights are structured)
# model.layers[1].weight = model_weights[1][1]
# model.layers[1].bias = model_weights[1][2]
# model.layers[2].weight = model_weights[2][1]
# model.layers[2].bias = model_weights[2][2]
# model.layers[3].weight = model_weights[3][1]
# model.layers[3].bias = model_weights[3][2]

# dummy_input = randn(Float32, (input_dim,))


# ONNX.save(onnx_model_path, model)