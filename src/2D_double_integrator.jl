module DoubleIntegrator2D

using Flux
using LinearAlgebra
using MatrixEquations

# Dimensions
x_dim = 4  # 2D position (x, y) and velocity (vx, vy)
u_dim = 2  # acceleration in x and y

# State boundaries (x, y, vx, vy)
x_low = Float32[-1, -1, -1, -1]
x_high = Float32[1, 1, 1, 1]

# Control boundaries (acceleration in x and y)
u_low = Float32[-1, -1]
u_high = Float32[1, 1]

# Hazard (circle) with center at (0, 0) and radius 0.4
hazard_center = [0.0, 0.0]
hazard_radius = 0.4

dt = 0.1

# Linear dynamics model
A = Float32[1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1]  # State transition matrix
B = Float32[0 0; 0 0; dt 0; 0 dt]  # Control input matrix
Q = diagm(Float32[1, 1, 0.1, 0.1])  # State cost
R = diagm(Float32[0.1, 0.1])  # Control cost
P, _, _ = ared(A, B, R, Q)  # Solution to Riccati equation
K = inv(R + B' * P * B) * (B' * P * A)  # LQR gain
AB = [A B]

# Nonlinear constraint function based on distance from hazard (circle)
function constraint(x::Array{Float32})::Array{Float32}
    # Compute distance from hazard center (0, 0)
    dist_to_hazard = sqrt.(x[1,:].^2 + x[2,:].^2)
    # dist_to_hazard = norm(x[1:2, :] - hazard_center, dims=1)
    # Constraint: radius - dist(hazard_center, point_loc)
    result = hazard_radius .- dist_to_hazard
    return reshape(result, 1, length(result))
end

# Policy model (LQR control)
pi_model = Chain(
    Dense(-K),  # LQR controller
    Dense(Matrix{Float32}(I(u_dim)), -u_low, relu),
    Dense(Matrix{Float32}(I(u_dim)), u_low),
    Dense(Matrix{Float32}(-I(u_dim)), u_high, relu),
    Dense(Matrix{Float32}(-I(u_dim)), u_high),
)

# System dynamics model (f(x) = Ax + Bu)
f_model = Chain(
    Dense(AB),  # Dynamics: x' = Ax + Bu
    Dense(Matrix{Float32}(I(x_dim)), -x_low, relu),
    Dense(Matrix{Float32}(I(x_dim)), x_low),
    Dense(-Matrix{Float32}(I(x_dim)), x_high, relu),
    Dense(-Matrix{Float32}(I(x_dim)), x_high),
)

# Combined dynamics and control model
f_pi_model = Chain(
    Parallel(+,
        Dense(A),
        Chain(pi_model, Dense(B)),
    ),
    Dense(Matrix{Float32}(I(x_dim)), -x_low, relu),
    Dense(Matrix{Float32}(I(x_dim)), x_low),
    Dense(-Matrix{Float32}(I(x_dim)), x_high, relu),
    Dense(-Matrix{Float32}(I(x_dim)), x_high),
)

end
