function create_mlp(
    input_dim::Int64,
    output_dim::Int64,
    hidden_sizes::Vector{Int64},
)::Chain
    layers = []
    push!(layers, Dense(input_dim => hidden_sizes[1], relu))
    for i in 1:length(hidden_sizes) - 1
        push!(layers, Dense(hidden_sizes[i] => hidden_sizes[i + 1], relu))
    end
    push!(layers, Dense(hidden_sizes[end] => output_dim))
    return Chain(layers...)
end

function create_closed_loop_dynamics_model(
    f_model::Chain,
    pi_model::Any,
    data::Dict{String, Array{Float32}},
    x_low::Vector{Float32},
    x_high::Vector{Float32},
    u_dim::Int64,
)::Chain
    x_dim = length(x_low)
    return Chain(
        Parallel(+,
            Dense(Matrix{Float32}(I(x_dim))),
            Chain(
                Parallel(+,
                    Chain(
                        Dense(diagm(1 ./ data["x_std"]), -data["x_mean"] ./ data["x_std"]),
                        Dense(vcat(Matrix{Float32}(I(x_dim)), zeros(Float32, u_dim, x_dim))),
                    ),
                    Chain(
                        pi_model,
                        Dense(diagm(1 ./ data["u_std"]), -data["u_mean"] ./ data["u_std"]),
                        Dense(vcat(zeros(Float32, x_dim, u_dim), Matrix{Float32}(I(u_dim)))),
                    ),
                ),
                f_model,
                Dense(diagm(data["dx_std"]), data["dx_mean"]),
            ),
        ),
        # max(x, x_low) = relu(x - x_low) + x_low
        Dense(Matrix{Float32}(I(x_dim)), -x_low, relu),
        Dense(Matrix{Float32}(I(x_dim)), x_low),
        # min(x, x_high) = -max(-x, -x_high) = -relu(-x + x_high) + x_high
        Dense(-Matrix{Float32}(I(x_dim)), x_high, relu),
        Dense(-Matrix{Float32}(I(x_dim)), x_high),
    )
end

function create_value_constraint_model(V_model::Any, h_model::Any)::Chain
    return Chain(Parallel(+,
        Chain(V_model, Dense(Float32[1; 0;;])),
        Chain(h_model, Dense(Float32[0; 1;;])),
    ))
end

function create_value_next_value_model(V_model::Any, f_pi_model::Any)::Chain
    return Chain(Parallel(+,
        Chain(V_model, Dense(Float32[1; 0;;])),
        Chain(f_pi_model, V_model, Dense(Float32[0; 1;;])),
    ))
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
    u_high::AbstractVector
end

function DenseInterval(W, b, x_dim, u_low, u_high)
    DenseInterval(W, b, x_dim, u_low, u_high)
end

# Forward pass for interval arithmetic
function (layer::DenseInterval)(xu::AbstractMatrix)
    # Compute lower and upper bounds for each neuron
    # xu are embeddings of x(32) and u
    x_dim = 32
    x = xu[1:x_dim, :]

    W_x = layer.W[:, 1:x_dim]
    z = W_x * x .+ layer.b
    # println(z)

    W_u = layer.W[:, x_dim+1:end]
    W_up = max.(W_u, 0.0)
    W_un = min.(W_u, 0.0)
    # println(W_u)

    l = W_up * layer.u_low .+ W_un * layer.u_high .+ z
    return l
end

# Create the model
function create_parallel_affine_Q_interval(x_dim, u_dim, u_low, u_high)

    
    function filter_x(input)
        return input[1:x_dim, :]
    end
    
    function filter_u(input)
        if size(input, 1) == x_dim
            return input
        end
        return input[x_dim+1:end, :]
    end
    
    # Branch 1
    b1 = Chain(
        filter_x,
        Dense(x_dim, 32, relu),
        Dense(32, 32, relu)
    )

    # Define the interval arithmetic layer for the final output
    final_layer_interval = DenseInterval(randn(1, 32 + u_dim), randn(1), 32, u_low, u_high)

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


function create_Q_constraint_model(Q_model, h_model, task)
    function filter_x(input)
        return input[1:task.x_dim, :]
    end
    return Chain(Parallel(+,
        Chain(filter_x, h_model, Dense(Float32[1; 0;;])),
        Chain(Q_model,  Dense(Float32[0; 1;;]))
    ))
end

function create_parallel_affine_Q(x_dim, u_dim)
    # can take both x and xu as input
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

function create_Q_Q_prime(affine_Q, f_pi_model, task)
    # trainable parameters
    # println(affine_Q[1][1][2])
    # println(affine_Q[1][1][3])
    # println(affine_Q[2])
    
    # creating Q_prime
    affine_Q_interval = create_parallel_affine_Q_interval(task.x_dim, task.u_dim, task.u_low, task.u_high)
    # println(affine_Q_interval)
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
    function filter_x(input)
        return input[1:task.x_dim, :]
    end
    expand_layer = create_expand_xu_layer(task.x_dim, task.u_dim)
    println("PASS 2")
    return Chain(Parallel(+,
        Chain(affine_Q, Dense(Float32[1; 0;;])),
        Chain(filter_x, f_pi_model, expand_layer, affine_Q_interval, Dense(Float32[0; 1;;])),
    )), affine_Q_interval
end
