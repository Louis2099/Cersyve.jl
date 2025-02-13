module Unicycle4D

using Flux
using LinearAlgebra
using EllipsisNotation

# state: [xo, yo, v, theta]
x_dim = 4
x_low = Float32[-1, -1, -1, -pi]
x_high = Float32[1, 1, 1, pi]

# action: [a, w]
u_dim = 2
u_low = Float32[-1, -1]
u_high = Float32[1, 1]

dt = 0.1
ro = 0.4

# two square convex hull at 0.5 and -0.5
convex_hull1 = [-0.4; 0.6; -0.4; 0.6]
convex_hull2 = [0.6; -0.4; 0.6; -0.4]


function dynamics(x::Array{Float32}, u::Array{Float32})::Array{Float32}
    xo = x[1, ..]
    yo = x[2, ..]
    v = x[3, ..]
    theta = x[4, ..]

    a = u[1, ..]
    w = u[2, ..]

    # xo_tmp = xo - v * dt
    # dth = w * dt
    x_prime = Float32.(stack((
        xo + v .* cos.(theta) .* dt,
        yo + v .* sin.(theta) .* dt,
        v + a .* dt,
        theta + w .* dt,
    ); dims=1))
    return clamp.(x_prime, x_low, x_high)
end

function constraint(x::Array{Float32})::Array{Float32}
    return ro .- sqrt.(sum(x[1:2, ..] .^ 2; dims=1))
end

function convex_hull_constraint(x::Array{Float32})::Array{Float32}
    X = x[1:2, ..]
    A = [1 0; -1 0; 0 1; 0 -1]
    cost1 = -(A*X - convex_hull1)
    cost2 = -(A*X - convex_hull2)
    return max.(cost1, cost2)
end

function terminated(x::Array{Float32})::Array{Float32}
    xo = x[1, ..]
    yo = x[2, ..]
    return (xo .== x_low[2]) .| (xo .== x_high[2]) .| (
        yo .== x_low[3]) .| (yo .== x_high[3])
end

pi_model = Dense(Float32[0 0 0 0; 0 0 0 -1], Float32[1, 0])

end
