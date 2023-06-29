using JuMP
using GLPK
using CSV: read
using DataFrames: DataFrame, rename!
using GLM: fit, LinearModel, @formula, predict
using MLDataUtils: shuffleobs, splitobs
using Distributions: sample
using Random: seed!

seed!(42)

function newsvendor(d::AbstractArray; x::Union{Number, Nothing} = nothing)

    S = size(d, 1)
    global c, r, q, u
    m = Model(GLPK.Optimizer)

    # first stage
    if x === nothing
          @variable(m, x >= 0)
    else
          @assert 0. <= x <= u
          @expression(m, x, x)
    end

    # second stage
    @variable(m, y[1:S] >= 0)
    @variable(m, z[1:S] >= 0)

    @constraint(m, x <= u)
    @constraint(m, y .<= d)
    @constraint(m, y .+ z .<= x)

    @objective(m, Min, c * x - (1/S) * sum(q .* y .+ r .* z))

    return m
end

function solve(m)
    optimize!(m)
    return objective_value(m), value.(m[:x])
end

# FIXME from `pred2presc.jl`, but augmented `u` as `mean(demand) ~ 220`
c, r, q, u = 10, 5, 25, 300

FOLDER_PATH = "./exercises/optimization/presc2pred"
FILE_PATH = "$FOLDER_PATH/newsvendor_data.csv"

data = read(FILE_PATH, DataFrame)
# FIXME `Dia da Semana` wasn't accepted in `@formula`
rename!(data, ["weekday", "temperature", "demand"])

percs = .7, .2
trainidx, validx, testidx = splitobs(shuffleobs(1:size(data, 1)), percs)
train, val, test = data[trainidx, :], data[validx, :], data[testidx, :]

# a)
# SAA can use more data, as there are no hyperparameters to be chosen
trainval = vcat(train.demand, val.demand)
_, x_SAA = solve(newsvendor(trainval))
obj_SAA, _ = solve(newsvendor(test.demand, x = x_SAA))
# Out: (-3036.0371051935626, 233.43623354303892)

# b)
# FIXME `dom` became `intercept`
# `weekday` is not significant (p-value > 0.05), which suggests 
# that only temperature is relevant
ols = fit(
    LinearModel, 
    @formula(demand ~ weekday + temperature), 
    train
)

# c)
# using the training set, we can train a model and estimate the error distribution
# then, we can use the validation set to estimate the order quantity
# finally, we can use the test set to test how well the order quantity performs
for (f, X_val, X_test) in (
    (@formula(demand ~ temperature), val[!, ["temperature"]], test[!, ["temperature"]]),
    (@formula(demand ~ weekday), val[!, ["weekday"]], test[!, ["weekday"]]),
)
    ols = fit(LinearModel, f, train)
    epsilon = train.demand .- predict(ols, train)
    pred_demand = predict(ols, X_val) .+ sample(epsilon, size(X_val, 1))
    _, x_hat = solve(newsvendor(pred_demand))
    obj_hat, _ = solve(newsvendor(predict(ols, X_test), x = x_hat))
    println("objective value: $obj_hat | order quantity: $x_hat | ($f)")
end
# Out:
# objective value: -3137.8319334480890 | order quantity: 242.76957704342135 | (demand ~ temperature)
# objective value: -3206.7245804533027 | order quantity: 232.58289939014790 | (demand ~ weekday)

# Interestingly enough, the model that uses only temperature performs worse than the one 
# that uses only weekday, but both performed better than the SAA approach
