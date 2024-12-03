using Pkg
Pkg.activate("/home/jiaxingl/project/verify_julia_env")
Pkg.status()

using Cersyve
using Flux
using JLD2
using Random


# Initialize the fixed weight matrices for filtering
function create_filter_matrix(start_idx, end_idx, total_len)
    W = zeros(end_idx - start_idx + 1, total_len)
    for i in start_idx:end_idx
        W[i - start_idx + 1, i] = 1.0
    end
    return W
end

function create_parallel_affine_Q(x_dim, u_dim)
    # Assume the input has 13 elements: x (0–7), u (8–13)
    x_w = create_filter_matrix(1, x_dim, x_dim+u_dim)
    x_b = zeros(x_dim)
    filter_x = Dense(x_w, x_b)

    u_w = create_filter_matrix(x_dim, x_dim+u_dim-1, x_dim+u_dim)
    u_b = zeros(u_dim)
    filter_u = Dense(u_w, u_b)
    
    #Branch1
    b1 = Chain(
        filter_x,  # First hidden layer (32 neurons, input size is 8 for x)
        Dense(x_dim, 32, relu),  # First hidden layer (32 neurons, input size is 8 for x)
        Dense(32, 32, relu)  # Second hidden layer (32 neurons)
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

struct ExpandXToXU
    x_dim::Int
    u_dim::Int
end

# Define the forward pass for the layer
function (layer::ExpandXToXU)(x)
    # `x` is the input vector of size `x_dim`
    u = rand(layer.u_dim)  # Generate random `u` part
    return vcat(x, u)      # Concatenate `x` and `u`
end

# Example usage
function create_expand_xu_layer(x_dim, u_dim)
    return ExpandXToXU(x_dim, u_dim)
end

# Custom Dense Layer for Interval Arithmetic
struct DenseInterval
    W::AbstractMatrix
    b::AbstractVector
    x_dim::Int
    u_low::AbstractVector
    u_up::AbstractVector
end

function DenseInterval(W, b, x_dim, u_low, u_up)
    DenseInterval(W, b, x_dim, u_low, u_up)
end

# Forward pass for interval arithmetic
function (layer::DenseInterval)(xu::AbstractVector)
    # Compute lower and upper bounds for each neuron
    x = xu[layer.x_dim+1:end]
    W_u = layer.W[:, layer.x_dim+1:end]
    W_up = max.(W_u, 0.0)
    W_un = min.(W_u, 0.0)
    
    W_p = layer.W
    W_p[:, layer.x_dim+1:end] = W_up
    W_n = layer.W
    W_n[:, layer.x_dim+1:end] = W_un

    l = W_p * layer.u_low .+ W_n * layer.u_up .+ layer.b
    return l
end

# Create the model
function create_parallel_affine_Q_interval(x_dim, u_dim, u_low, u_up)
    # Assume the input has 13 elements: x (0–7), u (8–13)
    x_w = create_filter_matrix(1, x_dim, x_dim+u_dim)
    x_b = zeros(x_dim)
    filter_x = Dense(x_w, x_b)

    u_w = create_filter_matrix(x_dim, x_dim+u_dim-1, x_dim+u_dim)
    u_b = zeros(u_dim)
    filter_u = Dense(u_w, u_b)
    
    # Branch 1
    b1 = Chain(
        filter_x,
        Dense(x_dim, 32, relu),
        Dense(32, 32, relu)
    )

    # Define the interval arithmetic layer for the final output
    final_layer_interval = DenseInterval(randn(1, 32 + u_dim), randn(1), task.x_dim, task.u_low, task.u_high)

    # Complete model
    model = Chain(
        Parallel(
            vcat, 
            b1,
            filter_u
        ),
        final_layer_interval  # Perform interval arithmetic here
    )
    return model
end

function create_Q_Q_prime(affine_Q, task)
    # trainable parameters
    # println(affine_Q[1][1][2])
    # println(affine_Q[1][1][3])
    # println(affine_Q[2])
    
    # creating Q_prime
    affine_Q_interval = create_parallel_affine_Q_interval(task.x_dim, task.u_dim, task.u_low, task.u_high)
    println(affine_Q_interval)
    # Copy weights and biases from affine_Q to affine_Q_interval
    affine_Q_interval[1].layers[1].layers[2].weight .= affine_Q[1].layers[1].layers[2].weight
    affine_Q_interval[1].layers[1].layers[2].bias .= affine_Q[1].layers[1].layers[2].bias
    affine_Q_interval[1].layers[1].layers[3].weight .= affine_Q[1].layers[1].layers[3].weight
    affine_Q_interval[1].layers[1].layers[3].bias .= affine_Q[1].layers[1].layers[3].bias
    println("PASS 0")
    # Map the final layer to DenseInterval
    affine_Q_interval[2].W .= affine_Q[2].layers[1].weight
    affine_Q_interval[2].b .= affine_Q[2].layers[1].bias
    println("PASS 1")
    x_w = create_filter_matrix(1, task.x_dim, task.x_dim + task.u_dim)
    x_b = zeros(task.x_dim)
    filter_x = Dense(x_w, x_b)
    expand_layer = create_expand_xu_layer(task.x_dim, task.u_dim)
    println("PASS 2")
    return Chain(Parallel(+,
        Chain(affine_Q, Dense(Float32[1; 0;;])),
        Chain(filter_x, f_pi_model, expand_layer, affine_Q_interval, Dense(Float32[0; 1;;])),
    ))
end

task = Unicycle
affine_Q = create_parallel_affine_Q(task.x_dim, task.u_dim)

Q_Q_prime = create_Q_Q_prime(affine_Q, task)
# println(Q_Q_prime)

input = Float32.(rand(task.x_dim + task.u_dim))
Q_Q_prime(input)


