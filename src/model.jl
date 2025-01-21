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


function create_filter_matrix(start_idx, end_idx, total_len)
    W = zeros(end_idx - start_idx + 1, total_len)
    for i in start_idx:end_idx
        # println(i - start_idx + 1)
        W[i - start_idx + 1, i] = 1.0
    end
    return W
end

function create_expand_matrix(start_idx, end_idx, expand_dim)
    input_dim = end_idx - start_idx + 1

    W = zeros(expand_dim, input_dim)
    for i in start_idx:end_idx
        # println(i - start_idx + 1)
        W[i, i - start_idx + 1] = 1.0
    end
    return W
end




# Create the model
# function create_parallel_affine_Q_interval(x_dim, u_dim, u_low, u_high)

    
#     # function filter_x(input)
#     #     return input[1:x_dim, :]
#     # end
    
#     # function filter_u(input)
#     #     if size(input, 1) == x_dim
#     #         return input
#     #     end
#     #     return input[x_dim+1:end, :]
#     # end
#     W_x = create_filter_matrix(1, x_dim, x_dim + u_dim)
#     b_x = zeros(x_dim)
#     filter_x = Dense(W_x, b_x)
#     W_u = create_filter_matrix(x_dim + 1, x_dim + u_dim, x_dim + u_dim)
#     u_b = zeros(u_dim)
#     filter_u = Dense(W_u, u_b)

#     # start_idx, end_idx, expand_dim
#     b1_embedding_dim = 32
#     W_expand_emb = create_expand_matrix(1, b1_embedding_dim, b1_embedding_dim + u_dim)
#     b_expand_emb = zeros(b1_embedding_dim + u_dim)
#     expand_emb = Dense(W_expand_emb, b_expand_emb)

#     W_expand_u = create_expand_matrix(b1_embedding_dim + 1, b1_embedding_dim + u_dim, b1_embedding_dim + u_dim)
#     b_expand_u = zeros(b1_embedding_dim + u_dim)
#     expand_u = Dense(W_expand_u, b_expand_u)


#     W_replace_u = zeros(u_dim, u_dim)
#     b_replace_low = u_low
#     b_replace_high = u_high
#     replace_u_low = Dense(W_replace_u, b_replace_low)
#     replace_u_high = Dense(W_replace_u, b_replace_high)

#     # Branch 1
#     b1 = Chain(
#         filter_x,
#         Dense(x_dim, 32, relu),
#         Dense(32, 32, relu),
#         Dense(32, 1)
#         # expand_emb
#     )
    
#     b_low = Chain(
#         filter_u,
#         replace_u_low,
#         Dense(u_dim, 1)
#         # expand_u
#     )
#     b_high = Chain(
#         filter_u,
#         replace_u_high,
#         Dense(u_dim, 1)
#     )

#     model = Chain(
#         Parallel(
#             +, 
#             b1,
#             b_low,
#             b_high
#         ),
#         # final_layer_interval  # Perform interval arithmetic here
#     )


#     return model
# end


function weight_luul(W::AbstractMatrix)
    # Split the weight matrix into positive and negative components
    W_plus = max.(W, 0)  # Element-wise maximum
    W_minus = min.(W, 0) # Element-wise minimum
    
    # Construct the new weight matrix
    # new_W = [
    #     W_plus  zeros(size(W))  W_minus  zeros(size(W));
    #     W_minus zeros(size(W))  W_plus   zeros(size(W));
    #     zeros(size(W)) W_plus  zeros(size(W))  W_minus;
    #     zeros(size(W)) W_minus zeros(size(W))  W_plus
    # ]
    new_W = [W_plus W_minus zeros(size(W)) zeros(size(W));
             zeros(size(W)) zeros(size(W)) W_plus W_minus;
             W_minus W_plus zeros(size(W)) zeros(size(W));
             zeros(size(W)) zeros(size(W)) W_minus W_plus]
    return new_W
end

function bias_luul(b::AbstractVector)
    new_b = [b; b; b; b]
    
    return new_b
end

function weight_l(W::AbstractMatrix)
    # Split the weight matrix into positive and negative components
    W_plus = max.(W, 0)  # Element-wise maximum
    W_minus = min.(W, 0) # Element-wise minimum
    
    # Construct the new weight matrix
    new_W = [W_plus W_minus zeros(size(W)) zeros(size(W))]
    println("new_W shape", size(new_W))
    return new_W
end



function create_parallel_affine_Q_interval(affine_Q, x_dim, u_dim, u_low, u_high)
    println("UPDATED MIN Q")
    W_x = create_filter_matrix(1, x_dim, x_dim + u_dim)
    b_x = zeros(x_dim)
    filter_x = Dense(W_x, b_x)
    W_u = create_filter_matrix(x_dim + 1, x_dim + u_dim, x_dim + u_dim)
    u_b = zeros(u_dim)
    filter_u = Dense(W_u, u_b)

    x_w1 = affine_Q[1][1][2].weight
    x_b1 = affine_Q[1][1][2].bias
    x_w2 = affine_Q[1][1][3].weight
    x_b2 = affine_Q[1][1][3].bias
    x_w3 = affine_Q[1][1][4].weight
    x_b3 = affine_Q[1][1][4].bias

    x_2_x_luul = Dense(u_dim, 4*u_dim; bias = [u_low; u_high; u_high; u_low])
    x_2_x_luul.weight .= 0.0

    h1_w = affine_Q[1][2][2].weight
    h1_b = affine_Q[1][2][2].bias
    x_luul_2_h1_luul = Dense(weight_luul(h1_w), bias_luul(h1_b), relu)

    h2_w = affine_Q[1][2][3].weight
    h2_b = affine_Q[1][2][3].bias   
    h1_luul_2_h2_luul = Dense(weight_luul(h2_w), bias_luul(h2_b), relu)


    Q_w = affine_Q[1][2][4].weight
    Q_b = affine_Q[1][2][4].bias
    println("shape of Q_w", size(Q_w))
    h2_luul_2_Q_l = Dense(weight_l(Q_w), Q_b)

    # Branch 1
    b1 = Chain(
        filter_x,
        Dense(x_w1, x_b1, relu),
        Dense(x_w2, x_b2, relu),
        Dense(x_w3, x_b3)
        # expand_emb
    )

    b_low = Chain(
        filter_u,
        x_2_x_luul,
        x_luul_2_h1_luul,
        h1_luul_2_h2_luul,
        h2_luul_2_Q_l
    )
    model = Chain(
        Parallel(
            +, 
            b1,
            b_low
            
        )
    )


    return model
end


function create_Q_constraint_model(Q_model, h_model, task)
    # function filter_x(input)
    #     return input[1:task.x_dim, :]
    # end

    W_x = create_filter_matrix(1, task.x_dim, task.x_dim + task.u_dim)
    b_x = zeros(task.x_dim)
    filter_x = Dense(W_x, b_x)
    return Chain(Parallel(+,
        Chain(Q_model,  Dense(Float32[1; 0;;])),
        Chain(filter_x, h_model, Dense(Float32[0; 1;;]))
    ))
end

function create_parallel_affine_Q(x_dim, u_dim, hidden_sizes=32)
    # can take both x and xu as input
    # function filter_x(input)
    #     return input[1:x_dim, :]
    # end
    
    # function filter_u(input)
    #     return input[x_dim+1:end, :]
    # end
    W_x = create_filter_matrix(1, x_dim, x_dim + u_dim)
    b_x = zeros(x_dim)
    filter_x = Dense(W_x, b_x)
    W_u = create_filter_matrix(x_dim + 1, x_dim + u_dim, x_dim + u_dim)
    u_b = zeros(u_dim)
    filter_u = Dense(W_u, u_b)

    # start_idx, end_idx, expand_dim
    b1_embedding_dim = 32
    W_expand_emb = create_expand_matrix(1, b1_embedding_dim, b1_embedding_dim + u_dim)
    b_expand_emb = zeros(b1_embedding_dim + u_dim)
    expand_emb = Dense(W_expand_emb, b_expand_emb)

    W_expand_u = create_expand_matrix(b1_embedding_dim + 1, b1_embedding_dim + u_dim, b1_embedding_dim + u_dim)
    b_expand_u = zeros(b1_embedding_dim + u_dim)
    expand_u = Dense(W_expand_u, b_expand_u)

    # Branch 1
    b1 = Chain(
        filter_x,
        Dense(x_dim, 32, relu),
        Dense(32, 32, relu),
        Dense(32, 1)
    )
    # b2 = Chain(
    #     filter_u,
    #     # expand_u
    #     Dense(u_dim, 1)
    # )

    b2 = Chain(
        filter_u,
        # expand_u
        Dense(u_dim, hidden_sizes, relu),
        Dense(hidden_sizes, hidden_sizes, relu),
        Dense(hidden_sizes, 1)
    )
    # final_layer = Chain(Dense(32 + u_dim, 1))

    # Complete model
    model = Chain(
        Parallel(
            +, 
            b1,
            b2
        ),
        # final_layer  
    )
    return model
end



function create_Q_Q_prime(affine_Q, f_pi_model, f_model, task)
    # creating Q_prime
    affine_Q_interval = create_parallel_affine_Q_interval(affine_Q, task.x_dim, task.u_dim, task.u_low, task.u_high)

    W_x = create_filter_matrix(1, task.x_dim, task.x_dim + task.u_dim)
    b_x = zeros(task.x_dim)
    filter_x = Dense(W_x, b_x)
    
    # expand_layer = create_expand_xu_layer(task.x_dim, task.u_dim)
    expand_W = create_expand_matrix(1, task.x_dim, task.x_dim + task.u_dim)
    expand_b = zeros(task.x_dim + task.u_dim)
    expand_layer = Dense(expand_W, expand_b)
    # println("PASS 2")
    return Chain(Parallel(+,
        Chain(affine_Q, Dense(Float32[1; 0;;])),
        # Chain(filter_x, f_pi_model, expand_layer, affine_Q_interval, Dense(Float32[0; 1;;])),
        Chain(f_model, expand_layer, affine_Q_interval, Dense(Float32[0; 1;;])),
    )), affine_Q_interval
end


# function create_Q_Q_prime(affine_Q, f_pi_model, f_model, task)
#     # trainable parameters
#     # println(affine_Q[1][1][2])
#     # println(affine_Q[1][1][3])
#     # println(affine_Q[2])
    
#     # creating Q_prime
#     affine_Q_interval = create_parallel_affine_Q_interval(task.x_dim, task.u_dim, task.u_low, task.u_high)
#     # println(affine_Q_interval)
#     # Copy weights and biases from affine_Q to affine_Q_interval
#     affine_Q_interval[1].layers[1].layers[2].weight .= affine_Q[1].layers[1].layers[2].weight
#     affine_Q_interval[1].layers[1].layers[2].bias .= affine_Q[1].layers[1].layers[2].bias
#     affine_Q_interval[1].layers[1].layers[3].weight .= affine_Q[1].layers[1].layers[3].weight
#     affine_Q_interval[1].layers[1].layers[3].bias .= affine_Q[1].layers[1].layers[3].bias
#     affine_Q_interval[1].layers[1].layers[4].weight .= affine_Q[1].layers[1].layers[4].weight
#     affine_Q_interval[1].layers[1].layers[4].bias .= affine_Q[1].layers[1].layers[4].bias
    

#     # println("PASS 0")
#     # Map the final layer to DenseInterval
#     # affine_Q_interval[2].W .= affine_Q[2].layers[1].weight
#     # func_affine_Q_interval[2].W .= affine_Q[2].layers[1].weight

#     # affine_Q_interval[2].b .= affine_Q[2].layers[1].bias
#     # func_affine_Q_interval[2].b .= affine_Q[2].layers[1].bias

#     # b_low
#     # println(affine_Q[1].layers[2].layers[2].weight)
#     # println(affine_Q[1].layers[2].layers[2].bias)
#     # println(max.(affine_Q[1].layers[2].layers[2].weight, 0.0))
#     # println(min.(affine_Q[1].layers[2].layers[2].weight, 0.0))
#     affine_Q_interval[1].layers[2].layers[3].weight .= max.(affine_Q[1].layers[2].layers[2].weight, 0.0)
#     affine_Q_interval[1].layers[2].layers[3].bias .= affine_Q[1].layers[2].layers[2].bias
    
#     # b_high
#     affine_Q_interval[1].layers[3].layers[3].weight .= min.(affine_Q[1].layers[2].layers[2].weight, 0.0)
#     affine_Q_interval[1].layers[3].layers[3].bias .= 0.0



#     # println("PASS 1")
#     # function filter_x(input)
#     #     return input[1:task.x_dim, :]
#     # end
#     W_x = create_filter_matrix(1, task.x_dim, task.x_dim + task.u_dim)
#     b_x = zeros(task.x_dim)
#     filter_x = Dense(W_x, b_x)
    
#     # expand_layer = create_expand_xu_layer(task.x_dim, task.u_dim)
#     expand_W = create_expand_matrix(1, task.x_dim, task.x_dim + task.u_dim)
#     expand_b = zeros(task.x_dim + task.u_dim)
#     expand_layer = Dense(expand_W, expand_b)
#     # println("PASS 2")
#     return Chain(Parallel(+,
#         Chain(affine_Q, Dense(Float32[1; 0;;])),
#         # Chain(filter_x, f_pi_model, expand_layer, affine_Q_interval, Dense(Float32[0; 1;;])),
#         Chain(f_model, expand_layer, affine_Q_interval, Dense(Float32[0; 1;;])),
#     )), affine_Q_interval
# end

# function create_Q_Q_minmax(affine_Q, f_pi_model, f_model, task)
#     # trainable parameters
#     # println(affine_Q[1][1][2])
#     # println(affine_Q[1][1][3])
#     # println(affine_Q[2])
    
#     # creating Q_prime
#     affine_Q_interval = create_parallel_affine_Q_interval(task.x_dim, task.u_dim, task.u_low, task.u_high)
#     affine_Q_max = create_parallel_affine_Q_interval(task.x_dim, task.u_dim, task.u_low, task.u_high)
#     # println(affine_Q_interval)
#     # Copy weights and biases from affine_Q to affine_Q_interval
#     affine_Q_interval[1].layers[1].layers[2].weight .= affine_Q[1].layers[1].layers[2].weight
#     affine_Q_interval[1].layers[1].layers[2].bias .= affine_Q[1].layers[1].layers[2].bias
#     affine_Q_interval[1].layers[1].layers[3].weight .= affine_Q[1].layers[1].layers[3].weight
#     affine_Q_interval[1].layers[1].layers[3].bias .= affine_Q[1].layers[1].layers[3].bias
#     affine_Q_interval[1].layers[1].layers[4].weight .= affine_Q[1].layers[1].layers[4].weight
#     affine_Q_interval[1].layers[1].layers[4].bias .= affine_Q[1].layers[1].layers[4].bias

#     affine_Q_max[1].layers[1].layers[2].weight .= affine_Q[1].layers[1].layers[2].weight
#     affine_Q_max[1].layers[1].layers[2].bias .= affine_Q[1].layers[1].layers[2].bias
#     affine_Q_max[1].layers[1].layers[3].weight .= affine_Q[1].layers[1].layers[3].weight
#     affine_Q_max[1].layers[1].layers[3].bias .= affine_Q[1].layers[1].layers[3].bias
#     affine_Q_max[1].layers[1].layers[4].weight .= affine_Q[1].layers[1].layers[4].weight
#     affine_Q_max[1].layers[1].layers[4].bias .= affine_Q[1].layers[1].layers[4].bias
    

#     # println("PASS 0")
#     # Map the final layer to DenseInterval
#     # affine_Q_interval[2].W .= affine_Q[2].layers[1].weight
#     # func_affine_Q_interval[2].W .= affine_Q[2].layers[1].weight

#     # affine_Q_interval[2].b .= affine_Q[2].layers[1].bias
#     # func_affine_Q_interval[2].b .= affine_Q[2].layers[1].bias

#     # b_low
#     # println(affine_Q[1].layers[2].layers[2].weight)
#     # println(affine_Q[1].layers[2].layers[2].bias)
#     # println(max.(affine_Q[1].layers[2].layers[2].weight, 0.0))
#     # println(min.(affine_Q[1].layers[2].layers[2].weight, 0.0))
#     affine_Q_interval[1].layers[2].layers[3].weight .= max.(affine_Q[1].layers[2].layers[2].weight, 0.0)
#     affine_Q_interval[1].layers[2].layers[3].bias .= affine_Q[1].layers[2].layers[2].bias
    
#     affine_Q_max[1].layers[2].layers[3].weight .= min.(affine_Q[1].layers[2].layers[2].weight, 0.0)
#     affine_Q_max[1].layers[2].layers[3].bias .= affine_Q[1].layers[2].layers[2].bias

#     # b_high
#     affine_Q_interval[1].layers[3].layers[3].weight .= min.(affine_Q[1].layers[2].layers[2].weight, 0.0)
#     affine_Q_interval[1].layers[3].layers[3].bias .= 0.0

#     affine_Q_max[1].layers[3].layers[3].weight .= max.(affine_Q[1].layers[2].layers[2].weight, 0.0)
#     affine_Q_max[1].layers[3].layers[3].bias .= 0.0


#     # println("PASS 1")
#     # function filter_x(input)
#     #     return input[1:task.x_dim, :]
#     # end
#     W_x = create_filter_matrix(1, task.x_dim, task.x_dim + task.u_dim)
#     b_x = zeros(task.x_dim)
#     filter_x = Dense(W_x, b_x)
    
#     # expand_layer = create_expand_xu_layer(task.x_dim, task.u_dim)
#     expand_W = create_expand_matrix(1, task.x_dim, task.x_dim + task.u_dim)
#     expand_b = zeros(task.x_dim + task.u_dim)
#     expand_layer = Dense(expand_W, expand_b)
#     # println("PASS 2")
#     return Chain(Parallel(+,
#         Chain(affine_Q, Dense(Float32[1; 0;;])),
#         # Chain(filter_x, f_pi_model, expand_layer, affine_Q_interval, Dense(Float32[0; 1;;])),
#         Chain(f_model, expand_layer, affine_Q_interval, Dense(Float32[0; 1;;])),
#     )), affine_Q_interval, affine_Q_max
# end