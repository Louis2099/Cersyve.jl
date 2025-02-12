using Optimisers




function backtrack_GD_U(
    task::Any,
    Q_model::Any,
    max_step::Int64,
    lr::Float64,
    eps::Float64,
    xu::Matrix{Float32},
    xu_low::Vector{Float32},
    xu_high::Vector{Float32},
    h_model::Any,
)   
"""
1. Perform gradient descent on Q_model to obtain a surrogate argmin(u)_Q, with backtracking
""" 
    #TODO: use Adam optimizer
    # opt = Optimisers.setup(Optimisers.Adam(lr), xu[task.x_dim+1:end, :])
    opt = Optimisers.setup(Optimisers.Adam(lr), xu)
    converged = zeros(Bool, size(xu, 2))
    count = 0
    for _ in 1:max_step
        v = Q_model(xu)
        g = Flux.gradient(xu -> sum(Q_model(xu)), xu)[1]
        # println("g: ", g)
        # only update u
        temp_xu = copy(xu)
        if !isnothing(g)
            # only update unconverged u
            # temp_xu[task.x_dim+1:end, .!vec(converged)] = xu[task.x_dim+1:end, .!vec(converged)] - lr * g[task.x_dim+1:end, .!vec(converged)]
            
            # Optimisers.update!(opt, temp_xu[task.x_dim+1:end, .~converged], g[task.x_dim+1:end, .~converged])
            g[1:task.x_dim, :] .= 0
            g[:, .!vec(converged)] .= 0
            print("original xu: ", xu)
            update, opt = Optimisers.update!(opt, xu, g)
            println("type of update: ", typeof(update))
            println(update)
            println("updated xu: ", xu)
            xu .= update
            count += 1
        end
        # temp_xu = min.(max.(temp_xu, xu_low), xu_high)
        xu .= min.(max.(xu, xu_low), xu_high)
        # backtracking
        # if Q_model(temp_xu)[1, :] < v[1, :]
        # if Q_model(xu)[1, :] < v[1, :]
        #     count += 1
        #     # xu = temp_xu
        # else
        #     xu = temp_xu
        #     lr = lr/2
        # end
        # check the convergence, stop update the converged u
        # converged = abs.(v .- Q_model(temp_xu)) .< eps
        converged = abs.(v .- Q_model(xu)) .< eps
        if all(converged)
            break
        end
    end
    println("count: ", count)
    return xu
end