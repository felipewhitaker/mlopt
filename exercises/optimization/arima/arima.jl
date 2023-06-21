using JuMP
using Ipopt, Juniper
optimizer = optimizer_with_attributes(
    Juniper.Optimizer, 
    "nl_solver" => Ipopt.Optimizer,
    # "set_silent" => true # doesn't exist
)

using Statistics: std

aic(K, T, epsilon) = 2K + T*log(sum((epsilon - mean(epsilon)).^2))

function problem(y::AbstractArray, K::Integer, p::Integer)

    dy = diff(y)
    y = y[2:end]
    T = length(y)

    global optimizer
    m = Model(optimizer)

    @variable(m, alpha)
    @variable(m, beta)
    @variable(m, gamma)
    @variable(m, theta[1:p])
    @variable(m, epsilon[p:T])

    @constraint(
        m, 
        [t in (1 + p):T], 
        dy[t] == alpha + beta*t + gamma*y[t-1] + theta'*dy[t-p:t-1] + epsilon[t]
    )

    # # true zero norm to be satisfied
    # @constraint(m, sum(alpha != 0, beta != 0, (theta .!= 0)...) <= K)

    # limit parameters to be different from zero
    @variable(m, b[1:(p + 3)], Bin)
    @constraint(m, [alpha beta gamma theta...] .* (1 .- b) .== 0)
    @constraint(m, sum(b) <= K)

    @objective(m, Min, sum(epsilon.^2))
    return m
end

using CSV
using DataFrames

y = CSV.read("./exercises/optimization/arima/AirPassengers.csv", DataFrame)[!, "value"]

d = Dict()
for k in 1:5:40
    model = problem(y, k, 25)
    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        println("Optimal solution found with K = $k")
        println("Objective value: ", objective_value(model))
        kaic = aic(k, length(y), value.(collect(model[:epsilon])))
        println("AIC: ", kaic)
        d[k] = kaic
    else
        println("No optimal solution found with K = $k")
    end
end
d

using Plots

plot(y)


# # CoPilot is creative

# function arima(y, p, d, q)
#     T = length(y)
#     K = p + q + 1
#     epsilon = zeros(T)
#     for t = K:T
#         epsilon[t] = y[t] - sum([y[t-i] for i = 1:p]) - sum([epsilon[t-i] for i = 1:q])
#     end
#     return aic(K, T, epsilon)
# end

# function arima_opt(y, p, d, q)
#     T = length(y)
#     K = p + q + 1
#     m = Model(with_optimizer(Ipopt.Optimizer))
#     @variable(m, -1 <= theta[1:p] <= 1)
#     @variable(m, -1 <= phi[1:q] <= 1)
#     @variable(m, -1 <= mu <= 1)
#     @variable(m, -1 <= sigma <= 1)
#     @variable(m, epsilon[1:T])
#     @constraint(m, epsilon .== y .- mu .- sum([y[i] for i = 1:p]) .- sum([epsilon[i] for i = 1:q]))
#     @NLobjective(m, Min, aic(K, T, epsilon))
#     optimize!(m)
#     return value.(theta), value.(phi), value(mu), value(sigma)
# end