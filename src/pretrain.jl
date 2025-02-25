using Optimisers
function pretrain_value(
    V_model::Chain,
    f_pi_model::Any,
    h_model::Any,
    x_low::Vector{Float32},
    x_high::Vector{Float32};
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
)
    rng = MersenneTwister(1)
    optim = Flux.setup(AdamW(lr, (0.9, 0.999), weight_decay), V_model)
    if isnothing(log_dir)
        log_dir = joinpath(@__DIR__, "../log/")
    end
    log_path = joinpath(log_dir, "pretrain_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS"))
    logger = TBLogger(log_path)

    for _ in ProgressBar(1:iter_num)
        x = uniform(x_low, x_high, batch_size)
        x_prime = f_pi_model(x)
        c = h_model(x)
        v_targ = (1 - gamma) * c + gamma * max.(c, V_model(x_prime))

        function loss_fn(m)
            if isnothing(penalty)
                return mean((m(x) - v_targ) .^ 2)
            elseif penalty == "SNR"
                noise = Float32(noise_scale) * space_size / 2 .* randn(rng, Float32, size(x))
                v_pred, snr_loss = forward_with_snr(m, [x x + noise]; alpha=snr_coef[1], beta=snr_coef[2])
                return mean((v_pred - v_targ) .^ 2) + snr_loss
            elseif penalty == "APA"
                noise = Float32(noise_scale) * space_size / 2 .* randn(rng, Float32, size(x))
                v_pred, apa_loss = forward_with_apa(m, [x x + noise]; alpha=apa_coef)
                return mean((v_pred - v_targ) .^ 2) + apa_loss
            end
        end

        loss, grad = Flux.withgradient(loss_fn, V_model)
        Flux.update!(optim, V_model, grad[1])

        with_logger(logger) do
            @info "pretrain" loss=loss
            @info "pretrain" constraint_satisfying_rate=mean(c .<= 0) log_step_increment=0
            @info "pretrain" predicted_feasible_rate=mean(V_model(x) .<= 0) log_step_increment=0
        end
    end

    jldsave(joinpath(log_path, "/V_pretrain.jld2"); state=Flux.state(V_model))
end

function print_keys(nt, prefix="")
    for key in fieldnames(typeof(nt))
        new_prefix = prefix * string(key) * "."
        println(new_prefix)
        value = getfield(nt, key)
        if isa(value, NamedTuple)
            print_keys(value, new_prefix)
        end
    end
end

function pretrain_Q(
    Q_model::Chain,
    f_model::Any,
    f_pi_model::Any,
    pi_model::Any,
    h_model::Any,
    x_low::Vector{Float32},
    x_high::Vector{Float32};
    u_low::Vector{Float32},
    u_high::Vector{Float32},
    task::Any,
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
)
    println("Corrected L2 Model PRETRAIN")
    rng = MersenneTwister(1)
    # trainable_params = Flux.params(Q_model[1][1][2], Q_model[1][1][3], Q_model[2])
    # optim = Flux.setup(AdamW(lr, (0.9, 0.999), weight_decay), trainable_params)

    # optim = Flux.setup(AdamW(lr, (0.9, 0.999), weight_decay), Q_model)

    #TODO: for mul model
    optim = Optimisers.setup(Optimisers.AdamW(lr, (0.9, 0.999), weight_decay), Q_model)
    # Optimisers.freeze!(optim.layers[1].layers[1].layers[1])
    # Optimisers.freeze!(optim.layers[1].layers[2].layers[1])
    # Optimisers.freeze!(optim.layers[2])



    if isnothing(log_dir)
        log_dir = joinpath(@__DIR__, "../log/")
    end
    log_path = joinpath(log_dir, "pretrain_Q_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS"))
    log_path = joinpath(log_dir, "pretrain_Q_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS"))
    logger = TBLogger(log_path)

    count = 0
    x = uniform(x_low, x_high, batch_size)
    for _ in ProgressBar(1:iter_num)
        # if count % 50 == 0
        #     x = uniform(x_low, x_high, batch_size)
        # end
        count += 1
        x = uniform(x_low, x_high, batch_size)
        u = uniform(u_low, u_high, batch_size)
        # u = pi_model(x)
        
        state_action = vcat(x, u)

        x_prime = f_model(state_action)
        u_prime = pi_model(x_prime)
        
        c = h_model(x)
        c_prime = h_model(x_prime)

        state_action = vcat(x, u)
        state_action_prime = vcat(x_prime, u_prime)

        # v_targ = (1 - gamma) * c + gamma * max.(c, Q_model(state_action_prime))
        
        #TODO: learning target can be pi or argmin
        # v_targ = (1 - gamma) * max.(c, c_prime) + gamma * max.(max.(c, c_prime), Q_model(state_action_prime))
        # v_targ = (1 - gamma) * c + gamma * max.(c, Q_model(state_action_prime))
        argmin_Q = create_baseline_affine_Q_interval(Q_model, task.x_dim, task.u_dim, task.u_low, task.u_high)
        v_targ = (1 - gamma) * max.(c, c_prime) + gamma * max.(max.(c, c_prime), argmin_Q(state_action_prime))

        #TODO: use f and pi, get MC learning target
        # T = 100
        # traj = Array{Float32}(undef, size(x)..., T)
        # traj[:, :, 1] = x
        # x_temp = copy(x_prime)
        # for i in 2:T
        #     traj[:, :, i] = x_temp
        #     x_temp = f_pi_model(x_temp)
        # end

        # v_targ = maximum(h_model(traj)[1, :, :], dims=2)
        # v_targ = transpose(v_targ)
        # argmin_Q = create_mul_Q_interval(Q_model, task.x_dim, task.u_dim, task.u_low, task.u_high)
        # v_targ = (1 - gamma) * v_targ + gamma * max.(max.(c, c_prime), argmin_Q(state_action_prime))
        # branch_scale_idx = mean(norm(Q_model[1][1](state_action))./norm(Q_model[1][2](state_action)))
        

        function loss_fn(m)
            if isnothing(penalty)
                return mean((m(state_action) - v_targ) .^ 2)
            elseif penalty == "SNR"
                noise = Float32(noise_scale) * space_size / 2 .* randn(rng, Float32, size(state_action))
                v_pred, snr_loss = forward_with_snr(m, [x x + noise]; alpha=snr_coef[1], beta=snr_coef[2])
                return mean((v_pred - v_targ) .^ 2) + snr_loss
            elseif penalty == "APA"
                # noise = Float32(noise_scale) * space_size / 2 .* randn(rng, Float32, size(x))
                # v_pred, apa_loss = forward_with_apa(m, [x x + noise]; alpha=apa_coef)

                # disable apa loss for now
                # L2_loss = sum(norm(w)^2 for w in Flux.params(m)) + sum(norm(b)^2 for b in Flux.params(m))
                # branch_balance_penalty = bl_strength * scale_target * (norm(m[1][2][2].weight)^2 + norm(m[1][2][3].weight)^2 + norm(m[1][2][4].weight)^2 +norm(m[1][2][2].bias)^2 + norm(m[1][2][3].bias)^2 + norm(m[1][2][4].bias)^2) 
                
                # branch_balance_penalty = bl_strength * (branch_scale_idx-1) * norm(m[1][2](state_action))^2
                apa_loss = 0
                v_pred = m(state_action)
                # return mean((v_pred - v_targ) .^ 2) + apa_loss + L2_strength * L2_loss
                return mean((v_pred - v_targ) .^ 2) + apa_loss
            end
        end

        
        # loss, grad = Flux.withgradient(loss_fn, Q_model[1].layers[1])
        # Flux.update!(optim, Q_model, grad[1])
        
        loss, grad = Flux.withgradient(loss_fn, Q_model)
        
        Optimisers.update!(optim, Q_model, grad[1])

        # Flux.update!(optim, Q_model, grad[1])
        with_logger(logger) do
            
            # @info "pretrain" x_W=Q_model[1][1][3].weight[1] log_step_increment=0
            # @info "pretrain" x_b=Q_model[1][1][3].bias[1] log_step_increment=0

            # @info "pretrain" u_W=Q_model[1][2][2].weight[1] log_step_increment=0
            # @info "pretrain" u_b=Q_model[1][2][2].bias[1] log_step_increment=0
            
            # @info "pretrain" x_W=Q_model[1][1][2].weight[1] log_step_increment=0
            # @info "pretrain" x_b=Q_model[1][1][2].bias[1] log_step_increment=0

            # @info "pretrain" u_W=Q_model[1][2][2].weight[1] log_step_increment=0
            # @info "pretrain" u_b=Q_model[1][2][2].bias[1] log_step_increment=0

            # @info "pretrain" scale_idx=branch_scale_idx log_step_increment=0
            @info "pretrain" loss=loss
            @info "pretrain" constraint_satisfying_rate=mean(c .<= 0) log_step_increment=0
            @info "pretrain" predicted_feasible_rate=mean(Q_model(state_action) .<= 0) log_step_increment=0
        end
    end

    jldsave(joinpath(log_path, "Q_pretrain.jld2"); state=Flux.state(Q_model))
end