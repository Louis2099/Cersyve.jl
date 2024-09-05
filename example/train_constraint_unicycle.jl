using Cersyve
using JLD2
using Random

task = Unicycle
hidden_sizes = [16]
log_dir = joinpath(@__DIR__, "../log/dynamic/")
seed = 1


x_high = Float32[0.9091, 0.9091, 0.9091, 0.9091, 0.9091, 0.2, 0.9091, 0.9091, 0.2]
x_low = Float32[0, 0, 0, 0.3, 0.3, 0, 0.3, 0.3, 0]
x_dim = 9

Random.seed!(seed)

h_model = Cersyve.create_mlp(x_dim, 1, hidden_sizes)

function constraint(x::Array{Float32})::Array{Float32}
    hazard_radius = 0.1
    batch_size = size(x, 2)
    end_pos = x[1:3, :]
    hazard_pos = x[7:9, :]
    dist = sqrt.(sum((end_pos - hazard_pos) .^ 2, dims=1))
    hazard_size = fill(Float32(hazard_radius), 1, batch_size)
    cost = zeros(Float32, 1, batch_size)
    for i in 1:batch_size
        if dist[1, i] < hazard_size[1, i]
            cost[1, i] = (hazard_size[1, i] - dist[1, i])
        end
    end
    return cost
end

# train_constraint(
#     task.x_low,
#     task.x_high,
#     h_model,
#     task.constraint;
#     log_dir=log_dir,
# )

train_constraint(
    x_low,
    x_high,
    h_model,
    constraint;
    log_dir=log_dir,
)