using Plots
using CSV
using DataFrames
using Statistics: std

using JuMP
using Ipopt, Juniper
optimizer = optimizer_with_attributes(
    Juniper.Optimizer, 
    "nl_solver" => Ipopt.Optimizer,
    # "set_silent" => true # doesn't exist
)

aic(K, T, epsilon) = 2K + T*log(std(epsilon)^2)

function optarima(y, alpha, beta, gamma, theta, p)

    # FIXME repeated
    dy = diff(y)
    y = y[2:end]
    T = length(y)
    ps = 1 + p # only starts at the p-th element    

    return [-(alpha + beta * t + gamma * y[t - 1] + theta'*dy[t-p:t-1]) for t in ps:T]
end

function problem(y::AbstractArray, K::Integer, p::Integer, M::Float64 = 1.)

    dy = diff(y)
    y = y[2:end]
    T = length(y)

    ps = 1 + p # only starts at the p-th element

    global optimizer
    m = Model(optimizer)

    @variable(m, alpha)
    @variable(m, beta)
    @variable(m, gamma)
    @variable(m, theta[1:p])
    @variable(m, epsilon[ps:T])

    @constraint(
        m, 
        [t in ps:T], 
        dy[t] == alpha + beta*t + gamma*y[t-1] + theta'*dy[t-p:t-1] + epsilon[t]
    )

    # # true zero norm to be satisfied
    # @constraint(m, sum(alpha != 0, beta != 0, (theta .!= 0)...) <= K)

    # constraint for the zero norm
    @variable(m, b[1:(p + 3)], Bin)
    @constraint(m, [alpha beta gamma theta...]' .<= M .* b)
    @constraint(m, [alpha beta gamma theta...]' .>= - M .* b)
    @constraint(m, sum(b) <= K)

    @objective(m, Min, sum(epsilon.^2))
    return m
end

y = CSV.read("./exercises/optimization/arima/AirPassengers.csv", DataFrame)[!, "value"]

d = Dict()
for k in 1:5:40
    model = problem(y, k, 25)
    optimize!(model)

    if termination_status(model) in (MOI.LOCALLY_SOLVED, MOI.OPTIMAL)
        println("Solution found for K = $k")
        println("Objective value: ", objective_value(model))
        kaic = aic(k, length(y), value.(collect(model[:epsilon])))
        println("AIC: ", kaic)
        d[k] = kaic
    else
        println("No solution found with K = $k")
    end
end

best_k = argmin(d)
best_k = 10

pplots = []
for p in (3, 6, 13, 25)

    model = problem(y, best_k, p)
    optimize!(model)

    alpha, beta, gamma = value.(model[s] for s in (:alpha, :beta, :gamma))
    theta = value.(model[:theta])
    preds = optarima(y, alpha, beta, gamma, theta, p)

    plt = plot(y, label="y")
    plot!(plt, 1:(length(y) - p - 1), preds, label="preds")
    push!(pplots, plt)
end

plot(pplots...)