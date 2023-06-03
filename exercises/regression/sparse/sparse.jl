using JuMP
using Distributions

using Ipopt, Juniper
optimizer = optimizer_with_attributes(
    Juniper.Optimizer, 
    "nl_solver" => Ipopt.Optimizer,
    # "set_silent" => true # doesn't exist
)

mse(beta, (X, y)) = 1 / 2 * sum((y .- X * beta) .^ 2)

n_data = 100
n_feat, n_corr = 3, 10 # number of (features, fake features)
total_feat = n_feat + n_corr
m = n_feat + 1 # number of variables to select

lambda = 1e-2

Xt = rand(Normal(0, 1), n_data, n_feat)
Xf = rand(Uniform(0, 1), n_data, n_corr)
X = hcat(Xt, Xf)

true_coef = [1 2 3]'
y = Xt * true_coef

train_cut = trunc(Int, .7 * n_data)
(X_train, y_train) = (X[1:train_cut, :], y[1:train_cut])
(X_test, y_test) = (X[train_cut:end, :], y[train_cut:end])

function problem(model, (X, y), lambda = 1e-2)
    p = size(X, 2)
    beta = @variable(model, [1:p])
    @objective(model, Min, mse(beta, (X, y)) + 1 / (2 * lambda) * sum(beta .^ 2))
    return beta
end

"""Adds true l0 constraint (`binary == false`) or using 
auxiliary binary variable `b` (`binary == true`)"""
function add_l0(model, beta, K, binary::Bool)
    if !binary
        # true implementation of the problem
        @constraint(model, sum(beta .!= 0) <= K)
    else
        # a binary varible is added to control how many
        # `beta`'s can be not zero
        @variable(model, b[1:p], Bin)
        @constraint(model, [i in 1:p], beta[i] * (1 - b[i]) == 0)
        @constraint(model, sum(b) <= K)
    end
    return
end

"""Adds l0 constraint with fixed big M"""
function add_l0(model, beta, K, M::Float64)
    p = size(beta, 1)
    # a big M restriction is added, as per the class notes
    @variable(model, b[1:p], Bin)
    @constraint(model, [i in 1:p], beta[i] <= M * b[i])
    @constraint(model, [i in 1:p], beta[i] >= - M * b[i])
    @constraint(model, sum(b) <= K)
    return
end

function add_l0(model, beta, K, (X, y)::Tuple, M::Float64)
    p = size(beta, 1)
    # a big M restriction is added, using a previous upper bound
    # and optimizing the value for beta with the objective function
    # as a constraint
    m = Model(optimizer)
    @variable(m, beta_[1:p])
    # FIXME repeated objective function as `objective_function(model)` finishes
    # with "VariableNotOwned" due to `beta` being `model`'s variable
    # @constraint(m, objective_function(model) <= M)
    @constraint(m, mse(beta_, (X, y)) + 1 / (2 * lambda) * sum(beta_ .^ 2) <= M)

    @objective(m, Min, sum(beta_))
    optimize!(m)
    m_minus = value.(beta_)

    @objective(m, Max, sum(beta_))
    optimize!(m)
    m_plus = value.(beta_)

    # add_l0(model, beta, K, max.(m_minus, m_plus), true)
    M = max.(m_minus, m_plus)
    @variable(model, b[1:p], Bin)
    @constraint(model, [i in 1:p], beta[i] <= M[i] * b[i])
    @constraint(model, [i in 1:p], beta[i] >= - M[i] * b[i])
    @constraint(model, sum(b) <= K)
    return
end

"""Adds l0 constraint using Δy/Δx as values for big M[i]"""
function add_l0(model, beta, K, (X, y)::Tuple)
    # a big M for each variable is added
    p = size(beta, 1)
    ydelta = maximum(y) - minimum(y)
    M = [ydelta / (maximum(X[:, i]) - minimum(X[:, i])) for i in 1:p]
    @variable(model, b[1:p], Bin)
    @constraint(model, [i in 1:p], beta[i] <= M[i] * b[i])
    @constraint(model, [i in 1:p], beta[i] >= - M[i] * (1 - b[i]))
    @constraint(model, sum(b) <= K)
    return
end

function lasso(model, (X, y), lambda = 1e-2)

    function ols((X, y))
        model = Model(optimizer)
        set_silent(model)
        p = size(X, 2)
        beta = @variable(model, [1:p])
        @objective(model, Min, mse(beta, (X, y)))
        optimize!(model)
        return value.(beta)
    end

    p = size(X, 2)
    beta = @variable(model, [1:p])
    # from https://github.com/jump-dev/JuMP.jl/issues/48#issuecomment-25575389
    @variable(model, z[1:p])
    @constraint(model, z .>= 0)
    @constraint(model, z .>=  beta)
    @constraint(model, z .>= -beta)
    # # although below `w` is calculated for the implementation of
    # # AdaLasso, the optimization failed as OLS returned most 
    # # `beta`'s near zero (as expected) and thus the inverse of 
    # # it became a very big value for the objective function to handle,
    # # making it zero out all betas. An option here would be to redefine
    # # `w` values greater than a threshold to be e.g. 1e3
    # w = 1 ./ abs.(ols((X, y))) ^ 2
    @objective(model, Min, mse(beta, (X, y)) + 1 / (2 * lambda) * sum(beta))
    optimize!(model)
    return value.(beta)
end

# I kept receiving the error below when trying to implementing the `big M` constraint
# I tried changing the solver, but neither `GLPK`, `Ipopt` nor `HiGHS` worked
# then I used `Juniper`, but the results don't seem promising (most of the `beta`'s were
# almost zeroed out, but the model did not select it correctly); this might be due to
# how well correlated they are with `y`
# ERROR: Constraints of type MathOptInterface.ScalarQuadraticFunction{Float64}-in-
# MathOptInterface.LessThan{Float64} are not supported by the solver.

model = Model(optimizer)
beta = problem(model, (X_train, y_train))
t = @elapsed optimize!(model) # 0.0811454
value.(beta)'
# 0.076668  0.182976  0.101792  0.161939  0.150034  0.108556  0.171414  0.148496  0.0839363  0.146094  0.219129  0.160784  0.173243
# as expected, no values were selected: every regressor was partially selected
beta_naive = [0.076668  0.182976  0.101792  0.161939  0.150034  0.108556  0.171414  0.148496  0.0839363  0.146094  0.219129  0.160784  0.173243]

model = Model(optimizer)
beta = problem(model, (X_train, y_train))
add_l0(model, beta, m, 1e3)
t = @elapsed optimize!(model) # 42.00444543
value.(beta)'
# 1.7757e-9  9.07339e-9  2.16096e-9  9.12766e-9  4.00345e-5  2.69103e-9  0.334328  0.309015  1.75778e-9  9.01683e-9  0.392479  0.312662  9.06632e-9
# adding a fixed big M made the problem take a lot more time, but some variables were 
# selected, even if not the correct ones
beta_M = [1.7757e-9  9.07339e-9  2.16096e-9  9.12766e-9  4.00345e-5  2.69103e-9  0.334328  0.309015  1.75778e-9  9.01683e-9  0.392479  0.312662  9.06632e-9]

model = Model(optimizer)
beta = problem(model, (X_train, y_train))
add_l0(model, beta, m, (X, y), 1e3)
t = @elapsed optimize!(model) # 15.7629185
value.(beta)'
# 9.95953e-9  2.93675e-10  1.13249e-9  0.332748  0.315112  5.72182e-10  0.319452  7.97109e-10  1.55655e-9  7.4909e-9  1.84612e-7  0.274085  7.58956e-9
# adding a variable big M for each variable still made it take a lot of time, but much less 
# than before - even though it had to solve other two optimization problems. Either way, 
# the selected variables were not correctly chosen, most probably due to the variables being
# very correlated
beta_heuristic = [9.95953e-9  2.93675e-10  1.13249e-9  0.332748  0.315112  5.72182e-10  0.319452  7.97109e-10  1.55655e-9  7.4909e-9  1.84612e-7  0.274085  7.58956e-9]

model = Model(optimizer)
beta = problem(model, (X_train, y_train))
add_l0(model, beta, m, (X, y)) 
t = @elapsed optimize!(model) # 12.8920474
value.(beta)'
# -8.8585e-9  -1.00732e-8  -9.13341e-9  0.332748  0.315112  -3.74864e-8  0.319452  -1.73186e-8  -9.54877e-9  2.0009e-7  -2.93374e-8  0.274085  -5.96217e-9
# Δy/Δx as an heuristic seems to have worked fairly well, considering that it chose the same
# regressors as the heuristic suggested by the class notes, while taking a bit less time.
beta_delta = [-8.8585e-9  -1.00732e-8  -9.13341e-9  0.332748  0.315112  -3.74864e-8  0.319452  -1.73186e-8  -9.54877e-9  2.0009e-7  -2.93374e-8  0.274085  -5.96217e-9]

model = Model(optimizer)
beta = problem(model, (X_train, y_train))
beta_lasso = lasso(model, (X_train, y_train))'

for (name, found_beta) in (
    ("naive", beta_naive),
    ("M", beta_M),
    ("heuristic", beta_heuristic),
    ("delta", beta_delta),
    ("lasso", beta_lasso)
)
    test_mse = round(mse(found_beta', (X_test, y_test)), digits = 2)
    println("Experiment $name got $test_mse")
end

# Experiment naive got 1.86
# Experiment M got 6.51
# Experiment heuristic got 2.21
# Experiment delta got 2.21
# Experiment lasso got 54441.42

# interestingly enough, the best result came from the naive approach; this is
# most probably due to having very correlated regressors, making diversifying 
# (not zeroing out any `beta`, but allowing all to take part on the prediction)
# a good strategy
