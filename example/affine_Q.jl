using Flux
struct FilterX
    W::Matrix  # Weight matrix
end

struct FilterU
    W::Matrix  # Weight matrix
end

function (layer::FilterX)(input::AbstractVector)
    return layer.W * input
end

Flux.@functor FilterX  # Make the layer compatible with Flux
function Flux.params(layer::FilterX)
    return Flux.Params([])  # Exclude weights from being trainable
end

# Define a filtering layer for extracting u (indices 9 to 14)


function (layer::FilterU)(input::AbstractVector)
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

# Test the model with a sample input
model = create_affine_Q(8, 5)
x_u = rand(13)  # Example input [x, u] with 13 elements
output = model(x_u)

println("Output: ", output)
