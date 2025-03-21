function train_policy(
    pi_model::Any,
    f_model::Any,
    h_model::Any,
    Q_model::Any,
    x_low::Vector{Float32},
    x_high::Vector{Float32},
    u_low::Vector{Float32},
    u_high::Vector{Float32},
    task::Any;
    pi_ready = false,
    gamma::Float64 = 0.9,
    lr::Float64 = 3e-4,
    batch_size::Int64 = 256,
    iter_num::Int64 = 100000,
    weight_decay::Float64 = 0.0,
    log_dir::Union{String, Nothing} = nothing,
    q_step::Int64 = 10000,
)

"""
pi_model
loss: Minimizing Q value
co: Whether to cotrain Q
"""

    rng = MersenneTwister(1)
    # trainable_params = Flux.params(Q_model[1][1][2], Q_model[1][1][3], Q_model[2])
    # optim = Flux.setup(AdamW(lr, (0.9, 0.999), weight_decay), trainable_params)

    # optim = Flux.setup(AdamW(lr, (0.9, 0.999), weight_decay), Q_model)

    #TODO: for mul model
    pi_optim = Optimisers.setup(Optimisers.AdamW(lr, (0.9, 0.999), weight_decay), pi_model)
    println("layers of model:", length(pi_model))
    Optimisers.freeze!(pi_optim.layers[4])
    Optimisers.freeze!(pi_optim.layers[5])
    Optimisers.freeze!(pi_optim.layers[6])
    Optimisers.freeze!(pi_optim.layers[7])

    # Optimisers.freeze!(optim.layers[1].layers[1].layers[1])
    # Optimisers.freeze!(optim.layers[1].layers[2].layers[1])
    # Optimisers.freeze!(optim.layers[2])



    if isnothing(log_dir)
        log_dir = joinpath(@__DIR__, "../log/")
    end
    log_path = joinpath(log_dir, "policy_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS"))
    log_path = joinpath(log_dir, "policy_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS"))
    logger = TBLogger(log_path)

    if pi_ready != true
        for _ in ProgressBar(1:iter_num)
            
            x = uniform(x_low, x_high, batch_size)
            # v_targ = (1 - gamma) * max.(c, c_prime) + gamma * max.(max.(c, c_prime), argmin_Q(state_action_prime))

            function loss_fn(m)
                u = m(x)
                xu = vcat(x, u)
                return mean(Q_model(xu))
            end
            
            
            u = pi_model(x)
            
            state_action = vcat(x, u)
            

            # x_prime = f_model(state_action)
            # u_prime = pi_model(x_prime)
            c = h_model(x)
            # c_prime = h_model(x_prime)
            # state_action = vcat(x, u)
            # state_action_prime = vcat(x_prime, u_prime)
            argmin_Q = create_x_mul_xu_Q_interval_old(Q_model, task.x_dim, task.u_dim, task.u_low, task.u_high)
            d_Q = mean(Q_model(state_action) .- argmin_Q(state_action))


            # loss, grad = Flux.withgradient(loss_fn, Q_model[1].layers[1])
            # Flux.update!(optim, Q_model, grad[1])
            
            loss, grad = Flux.withgradient(loss_fn, pi_model)
            
            Optimisers.update!(pi_optim, pi_model, grad[1])

            # Flux.update!(optim, Q_model, grad[1])
            with_logger(logger) do
                @info "pretrain" loss=loss
                @info "pretrain" d_Q=d_Q
                @info "pretrain" constraint_satisfying_rate=mean(c .<= 0) log_step_increment=0
                @info "pretrain" predicted_feasible_rate=mean(Q_model(state_action) .<= 0) log_step_increment=0
                @info "pretrain" min_Q=mean(argmin_Q(state_action)) log_step_increment=0
                @info "pretrain" pi_Q=mean(Q_model(state_action)) log_step_increment=0
            end
        end
        jldsave(joinpath(log_path, "pi_pretrained.jld2"); state=Flux.state(pi_model))
    end

    if q_step >0
        q_optim = Optimisers.setup(Optimisers.AdamW(lr, (0.9, 0.999), weight_decay), Q_model)
        Optimisers.freeze!(q_optim.layers[1].layers[1].layers[1])
        Optimisers.freeze!(q_optim.layers[2])
        for i in ProgressBar(1:q_step)
            x = uniform(x_low, x_high, batch_size)
            u = uniform(u_low, u_high, batch_size)
            state_action = vcat(x, u)

            x_prime = f_model(state_action)
            u_prime = pi_model(x_prime)

            c = h_model(x)
            c_prime = h_model(x_prime)            
            state_action_prime = vcat(x_prime, u_prime)
            v_targ = (1 - gamma) * max.(c, c_prime) + gamma * max.(max.(c, c_prime), Q_model(state_action_prime))
            function loss_fn(m)
                v_pred = m(state_action)
                return mean((v_pred - v_targ) .^ 2)
            end
            loss, grad = Flux.withgradient(loss_fn, Q_model)
        
            Optimisers.update!(q_optim, Q_model, grad[1])
            
            with_logger(logger) do    
                @info "finetune_Q" loss=loss
                @info "finetune_Q" constraint_satisfying_rate=mean(c .<= 0) log_step_increment=0
                @info "finetune_Q" predicted_feasible_rate=mean(Q_model(state_action) .<= 0) log_step_increment=0
            end
        end
        jldsave(joinpath(log_path, "Q_finetunes.jld2"); state=Flux.state(Q_model))
    end
end

function cotrain_Q_pi(
    Q_model::Chain,
    f_model::Any,
    pi_model::Any,
    h_model::Any,
    x_low::Vector{Float32},
    x_high::Vector{Float32},
    u_low::Vector{Float32},
    u_high::Vector{Float32},
    task::Any;
    gamma::Float64 = 0.9,
    lr::Float64 = 3e-4,
    batch_size::Int64 = 256,
    iter_num::Int64 = 100000,
    weight_decay::Float64 = 0.0,
    penalty::Union{String, Nothing} = nothing,  # "APA" / "SNR" / nothing
    noise_scale::Float64 = 0.1,
    space_size::Union{Vector{Float32}, Nothing} = nothing,
    apa_coef::Float64 = 0.01,
    snr_coef::Tuple{Float64, Float64} = (1e-3, 5e-4),
    log_dir::Union{String, Nothing} = nothing,
    bl_strength::Float64 = 1e-2,
    pi_steps::Int64 = 1,
    freq_target_Q::Int64 = 100,
)
    println("Corrected L2 Model PRETRAIN")
    rng = MersenneTwister(1)
    # trainable_params = Flux.params(Q_model[1][1][2], Q_model[1][1][3], Q_model[2])
    # optim = Flux.setup(AdamW(lr, (0.9, 0.999), weight_decay), trainable_params)

    # optim = Flux.setup(AdamW(lr, (0.9, 0.999), weight_decay), Q_model)

    #TODO: for mul model
    optim = Optimisers.setup(Optimisers.AdamW(lr, (0.9, 0.999), weight_decay), Q_model)
    Optimisers.freeze!(optim.layers[1].layers[1].layers[1])
    # Optimisers.freeze!(optim.layers[1].layers[2].layers[1])
    Optimisers.freeze!(optim.layers[2])

    pi_optim = Optimisers.setup(Optimisers.AdamW(lr, (0.9, 0.999), weight_decay), pi_model)

    if isnothing(log_dir)
        log_dir = joinpath(@__DIR__, "../log/")
    end
    log_path = joinpath(log_dir, "pretrain_Q_pi" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS"))
    logger = TBLogger(log_path)

    count = 0
    x = uniform(x_low, x_high, batch_size)
    pi_loss = 0
    target_Q = Flux.deepcopy(Q_model)
    for _ in ProgressBar(1:iter_num)
        # if count % 50 == 0
        #     x = uniform(x_low, x_high, batch_size)
        # end
        count += 1
        x = uniform(x_low, x_high, batch_size)
        u = uniform(u_low, u_high, batch_size)
        
        state_action = vcat(x, u)

        x_prime = f_model(state_action)
        u_prime = pi_model(x_prime)
        
        c = h_model(x)
        c_prime = h_model(x_prime)

        state_action = vcat(x, u)
        state_action_prime = vcat(x_prime, u_prime)
        

        # v_targ = (1 - gamma) * c + gamma * max.(c, Q_model(state_action_prime))
        
        
        v_targ = (1 - gamma) * max.(c, c_prime) + gamma * max.(max.(c, c_prime), Q_model(state_action_prime))

        function loss_fn(m)
            if isnothing(penalty)
                return mean((m(state_action) - v_targ) .^ 2)
            elseif penalty == "SNR"
                noise = Float32(noise_scale) * space_size / 2 .* randn(rng, Float32, size(state_action))
                v_pred, snr_loss = forward_with_snr(m, [x x + noise]; alpha=snr_coef[1], beta=snr_coef[2])
                return mean((v_pred - v_targ) .^ 2) + snr_loss
            elseif penalty == "APA"
                apa_loss = 0
                v_pred = m(state_action)
                return mean((v_pred - v_targ) .^ 2) + apa_loss
            end
        end

        function pi_loss_fn(m)
            u = m(x)
            xu = vcat(x, u)
            return mean(Q_model(xu))
        end
        
        # loss, grad = Flux.withgradient(loss_fn, Q_model[1].layers[1])
        # Flux.update!(optim, Q_model, grad[1])
        
        loss, grad = Flux.withgradient(loss_fn, Q_model)
        Optimisers.update!(optim, Q_model, grad[1])

        for i in 1:pi_steps
            pi_loss, pi_grad = Flux.withgradient(pi_loss_fn, pi_model)
            Optimisers.update!(pi_optim, pi_model, pi_grad[1])
        end
        if count % freq_target_Q == 0
            target_Q = Flux.deepcopy(Q_model)
        end
        # Flux.update!(optim, Q_model, grad[1])
        with_logger(logger) do
            
            @info "pretrain" loss=loss
            @info "pretrain" constraint_satisfying_rate=mean(c .<= 0) log_step_increment=0
            @info "pretrain" predicted_feasible_rate=mean(Q_model(state_action) .<= 0) log_step_increment=0
            @info "pretrain" pi_loss=pi_loss log_step_increment=0
        end
    end
    jldsave(joinpath(log_path, "pi_pretrain.jld2"); state=Flux.state(pi_model))
    jldsave(joinpath(log_path, "Q_pretrain.jld2"); state=Flux.state(Q_model))
end