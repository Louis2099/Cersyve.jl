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
c1 = 0.5
c2 = -0.5
ro = 0.2

# two square convex hull at 0.5 and -0.5
convex_hull1 = [-0.4; 0.6; -0.4; 0.6]
hull_center1 = [-0.5; -0.5] 
convex_hull2 = [0.6; -0.4; 0.6; -0.4]
hull_center2 = [0.5; 0.5]
# half width of the hull
hull_width = 0.1

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

# function constraint(x::Array{Float32})::Array{Float32}

#     return ro .- sqrt.(sum(x[1:2, ..] .^ 2; dims=1))
# end

function constraint(x::Array{Float32})::Array{Float32}
    # calculate the distance between 2 circle with c1 and c2
    dist1 = sqrt.(sum((x[1:2, ..] .- c1) .^ 2; dims=1)) .- ro
    dist2 = sqrt.(sum((x[1:2, ..] .- c2) .^ 2; dims=1)) .- ro
    dist = min.(dist1, dist2)
    cost = -dist
    return cost
end

# function constraint(x::AbstractArray{Float32})::AbstractArray{Float32}
#     X = x[1:2, ..]
#     A = [1 0; -1 0; 0 1; 0 -1]
#     # cost1 = -(A*X - convex_hull1)
#     # cost2 = -(A*X - convex_hull2)
#     x_dist1 = abs.(X[1, ..] .- hull_center1[1]) .- hull_width
#     y_dist1 = abs.(X[2, ..] .- hull_center1[2]) .- hull_width
#     x_dist2 = abs.(X[1, ..] .- hull_center2[1]) .- hull_width
#     y_dist2 = abs.(X[2, ..] .- hull_center2[2]) .- hull_width
#     # println("x_dist1: $(x_dist1)")
#     # println("y_dist1: $(y_dist1)")
#     # println("x_dist2: $(x_dist2)")
#     # println("y_dist2: $(y_dist2)")
#     dist1 = min.(max.(x_dist1, y_dist1), 0) + max.(x_dist1, y_dist1, 0)
#     dist2 = min.(max.(x_dist2, y_dist2), 0) + max.(x_dist2, y_dist2, 0)
#     # println("dist1: $(dist1)")
#     # println("dist2: $(dist2)")
#     cost = -min.(dist1, dist2)
#     # println("cost: $(cost)")

#     return reshape(cost, (1, size(cost)[1]))
# end

function terminated(x::Array{Float32})::Array{Float32}
    xo = x[1, ..]
    yo = x[2, ..]
    return (xo .== x_low[2]) .| (xo .== x_high[2]) .| (
        yo .== x_low[3]) .| (yo .== x_high[3])
end

pi_model = Dense(Float32[0 0 0 0; 0 0 0 -1], Float32[1, 0])

end
