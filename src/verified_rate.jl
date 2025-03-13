function split_hyperrectangle(u::Vector, l::Vector, res::Int, n::Int)
    dims = length(u)  # Number of dimensions
    split_dims = min(n, dims)  # Ensure we do not exceed the number of dimensions
    
    # Create ranges for split dimensions, keep others unchanged
    steps = [i <= split_dims ? range(l[i], u[i], length=res+1) : [l[i]] for i in 1:dims]
    
    # Generate grid points for lower corners of sub-hyperrectangles
    grid_points = Iterators.product(steps...)
    
    # Calculate bounds for each sub-hyperrectangle
    grid_bounds = []
    for point in grid_points
        lower = collect(point)
        upper = copy(lower)
        
        # For split dimensions, calculate upper bounds
        for i in 1:split_dims
            if i < length(lower)  # Ensure we're within bounds
                step_size = (u[i] - l[i]) / res
                upper[i] = lower[i] + step_size
            end
        end
        
        # For unsplit dimensions, use original upper bounds
        for i in (split_dims+1):dims
            upper[i] = u[i]
        end
        
        push!(grid_bounds, (lower, upper))
    end
    
    return grid_bounds
end


function verify_rate(
    x_low::Vector{Float32},
    x_high::Vector{Float32},
    x_dim::Int64,
    res::Int64,
    V_h_model::Any,
    V_V_prime_model::Any;
    con_start_values::Union{Nothing, Vector{Float64}} = nothing,
    inv_start_values::Union{Nothing, Vector{Float64}} = nothing,
    save_file::String = "verified_bounds.jld2"
)::Tuple{ModelVerification.ResultInfo, ModelVerification.ResultInfo}

    search_method = BFS(max_iter=1000, batch_size=1)
    split_method = Bisect(1)
    solver = MIPVerify(pre_bound_method=Crown())

    # Split the entire hyperrectangle state space into grids according to the res(resolution)
    girds_bound = split_hyperrectangle(x_high, x_low, res, x_dim)
    Y = Complement(HPolyhedron([1 0; 0 -1], [0, 0]))
    verified = 0
    total = length(girds_bound)
    verify_t = 0
    sofar = 0
    
    # Create arrays to store verified bounds and results
    verified_bounds = []
    verification_results = Dict{Int, Dict{String, Any}}()
    
    for i in ProgressBar(1:total)
        sofar += 1
        l, u = girds_bound[i]
        println(u, l)
        X = Hyperrectangle(low=l, high=u)
        
        # Verify constraint property
        con_problem = Problem(V_h_model, X, Y)
        con_t = @elapsed con_res = verify(search_method, split_method, solver, con_problem;
            collect_bound=true, start_values=con_start_values)
        @printf "Constraint property %s! Verification time: %.3fs\n" con_res.status con_t

        # Verify invariance property
        inv_problem = Problem(V_V_prime_model, X, Y)
        inv_t = @elapsed inv_res = verify(search_method, split_method, solver, inv_problem;
            collect_bound=true, start_values=inv_start_values)
        @printf "Invariance property %s! Verification time: %.3fs\n" inv_res.status inv_t
        
        verify_t += con_t + inv_t

        # Store results for this grid
        result_info = Dict{String, Any}(
            "lower_bound" => l,
            "upper_bound" => u,
            "constraint_status" => String(con_res.status),
            "invariance_status" => String(inv_res.status),
            "constraint_time" => con_t,
            "invariance_time" => inv_t
        )
        verification_results[i] = result_info
        
        # Save bounds if both properties hold
        if (con_res.status == :holds) && (inv_res.status == :holds)
            verified += 1
            push!(verified_bounds, (l, u))
        end
        
        # Save intermediate results every 100 iterations
        if i % 100 == 0
            println("Verified Rate: ", (verified/sofar))
            println("Verified Time: ", (verify_t))
            
            # Save intermediate results
            jldsave(save_file; 
                verified_bounds = verified_bounds,
                verification_results = verification_results,
                metadata = Dict(
                    "x_low" => x_low,
                    "x_high" => x_high,
                    "x_dim" => x_dim,
                    "resolution" => res,
                    "verified_count" => verified,
                    "total_count" => total,
                    "verified_rate" => verified/sofar,
                    "verification_time" => verify_t,
                    "timestamp" => now()
                )
            )
            println("Saved intermediate results to $(save_file)")
        end
    end

    # Final save
    jldsave(save_file; 
        verified_bounds = verified_bounds,
        verification_results = verification_results,
        metadata = Dict(
            "x_low" => x_low,
            "x_high" => x_high,
            "x_dim" => x_dim,
            "resolution" => res,
            "verified_count" => verified,
            "total_count" => total,
            "verified_rate" => verified/total,
            "verification_time" => verify_t,
            "timestamp" => now()
        )
    )
    
    println("Verified Rate: ", (verified/total))
    println("Verified Time: ", (verify_t))
    println("Saved final results to $(save_file)")
    
    # Return the last verification results
    # return (con_res, inv_res)
end

function load_verification_results(file::String)
    println("updated")
    # data = jldopen(file, "r") do f
    #     return Dict(
    #         "verified_bounds" => read(f, "verified_bounds"),
    #         "verification_results" => read(f, "verification_results"),
    #         "metadata" => read(f, "metadata")
    #     )
    # end
    data = jldopen(file, "r")
    # Print all metadata
    println("=== Verification Metadata ===")
    for (key, value) in data["metadata"]
        println("$key: $value")
        
    end
    println("===========================")
    return data["metadata"]
end