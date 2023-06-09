using Plots
using CSV
using DataFrames
using Statistics: mean, std
using Distributions, StatsPlots

using JuMP
using Ipopt, HiGHS, Juniper

# set global variables
optimizer = optimizer_with_attributes(
    Juniper.Optimizer, 
    "nl_solver" => optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0),
    "mip_solver" => optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => true)
)

FIND_BEST_K = false

aic(K, T, epsilon) = 2K + T*log(std(epsilon)^2)

function optarima(y, alpha, beta, gamma, theta, p, steps)

    error("not implemented")

    # FIXME repeated
    dy = diff(y)
    y = y[2:end]
    T = length(y)

    # one step at a time
    dys = dy[(end-p):end]
    yhats = y[(end-p):end]
    for s in 1:steps
        curr_t = s + T
        # AirPassengers is an integer number
        dyhat = round(Int, alpha + beta * curr_t + gamma * yhats[end] + theta' * dys[end-p:end-1])
        yhat = round(Int, dyhat + yhats[end])

        push!(dys, dyhat)
        push!(yhats, yhat)
    end
    return yhats[end-steps:end]
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
y_train, y_val = y[1:end-12], y[end-12:end]

d = Dict()
for k in 1:5:25

    if !FIND_BEST_K
        print("FIND_BEST_K was set to $FIND_BEST_K, skipping...\n")
        d = Dict(
            6  => 1384.43,
            16 => 1368.97,
            11 => 1375.92,
            21 => 1373.5,
            36 => 1396.61,
            26 => 1376.61,
            31 => 1386.61,
            1  => 1423.17,
        )
        break
    end

    model = problem(y_train, k, 25)
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

pplots = []
for p in (3, 6, 13, 25)

    model = problem(y_train, best_k, p)
    optimize!(model)

    h = histogram(value.(model[:epsilon]).data, normalize = true)
    plot!(h, x -> pdf(Normal(0, std(value.(model[:epsilon]))), x), label="N(0, σ²)")
    title!(h, "k = $best_k, p = $p")
    push!(pplots, h)

    # # FIXME tried to plot the predictions but it's not working
    # alpha, beta, gamma = value.(model[s] for s in (:alpha, :beta, :gamma))
    # theta = value.(model[:theta])
    # preds = optarima(y_train, alpha, beta, gamma, theta, p)

    # plt = plot(y, label="y")
    # plot!(plt, 1:(length(y) - p - 1), preds, label="preds")
    # push!(pplots, plt)
end

plot(pplots..., ncol = 1)
# savefig("./exercises/optimization/arima/residue.png")

model = problem(y_train, 16, 25)
optimize!(model)

bar(
    [[value(model[n]) for n in [:alpha :beta :gamma]]... value.(model[:theta])...]', 
    xticks = 1:(25 + 3)
)
# savefig("./exercises/optimization/arima/bar.png")

# visualizing the residue, it seems like the better distributed model is the one with `p = 25`,
# but the other ones are not bad either: `p = 3` and `p = 6` have almost a uniform distribution
# with some outliers; while `p = 13` has a more gaussian distribution, but with a long tail (
# seems to be over estimating `epsilon < 0`)