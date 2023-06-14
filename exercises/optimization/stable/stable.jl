using JuMP
using Ipopt
using Statistics: std
using MLDataUtils: splitobs
using Distributions: Uniform, Normal

n_data = 100
n_feat, n_corr = 3, 10 # number of (features, fake features)
total_feat = n_feat + n_corr

lambda = 1e-2

Xt = rand(Normal(0, 1), n_data, n_feat)
Xf = rand(Uniform(0, 1), n_data, n_corr)
X = hcat(Xt, Xf)

true_coef = [.8 .6 .3]'
y = Xt * true_coef + rand(Normal(0, .49), n_data)

traincut, valcut, testcut = splitobs(1:n_data, at = (.7, .2))
(X_train, y_train) = (X[traincut, :], y[traincut])
(X_val, y_val) = (X[valcut, :], y[valcut])
(X_test, y_test) = (X[testcut, :], y[testcut])

optimizer = Ipopt.Optimizer

mse(beta, (X, y)) = 1 / 2 * sum((y .- X * beta) .^ 2)

function linear_model(n_feat)
    m = Model(optimizer)
    set_silent(m)
    beta = @variable(m, [1:n_feat])
    return (m, beta)
end

function l1_reg(m::Model, beta)
    # add constraint on `theta` to enable l1 restriction
    theta = @variable(m, [1:size(beta)[1]])
    @constraint(m, theta .>=  beta)
    @constraint(m, theta .>= -beta)
    return theta
end

function regression(
    X::AbstractMatrix, 
    y::AbstractArray, 
    n_feat::Integer;
    lambda::Float64 = 0.
)
    m, beta = linear_model(n_feat)

    theta = l1_reg(m, beta)
    @objective(m, Min, mse(beta, (X, y)) + lambda * sum(theta))
    optimize!(m)
    return value.(beta)
end

function regression(
    X::AbstractMatrix, 
    y::AbstractArray, 
    n_feat::Integer,
    K::Float64;
    lambda::Float64 = 0.
)
    n_data = size(X)[1]

    m, beta = linear_model(n_feat)

    @variable(m, delta >= 0)
    @variable(m, u[1:n_data] >= 0)
    @constraint(m, delta .+ u .>= mse(beta, (X, y)))

    reg = 0.
    if lambda != 0.
        theta = l1_reg(m, beta)
        reg = lambda * sum(theta)
    end

    @objective(m, Min, K * delta + sum(u) + reg)
    optimize!(m)
    return value.(beta)
end

function find_best_beta(
    func::Base.Callable,
    X::AbstractMatrix, 
    y::AbstractArray,
    n_feat::Integer,
    (X_val, y_val)::Tuple{AbstractMatrix, AbstractArray},
    lambda::Vector{Float64} = 10 .^ collect(-4.:2);
    func_kwargs...
)
    best_lambda, best_mse, best_beta = 0., Inf, nothing
    for λ in lambda
        # using values here is dangerous: dictionaries don't ensure ordering,
        # so it might be unpacked in an incorrect order. This is kept like this
        # because the only possible argument to be received here is `K`, so
        # there is not risk of using arguments where there were not supposed
        # to be used. Ideally this call would be:
        # `func(X, y, n_feat, func_kwargs...; lambda = λ)`,
        # but this returning a `MethodError`, as the unpacking is giving 
        # `::Pair{Symbol, Float64}` (e.g. `:K = .8`), but it was expecting
        # simply a `Float64` (which is why `values` solves the "issue")
        beta = func(X, y, n_feat, values(func_kwargs)...; lambda = λ)
        cur_mse = mse(beta, (X_val, y_val))
        if cur_mse < best_mse
            best_lambda = λ
            best_mse = cur_mse
            best_beta = beta
        end
    end
    return (best_lambda, best_beta, best_mse)
end

K = .8

naive_beta = regression(X_train, y_train, total_feat)
reg_lambda, reg_beta, _ = find_best_beta(regression, X_train, y_train, total_feat, (X_val, y_val))

k_beta = regression(X_train, y_train, total_feat, K)
kreg_lambda, kreg_beta, _ = find_best_beta(regression, X_train, y_train, total_feat, (X_val, y_val); K = K)

for (name, beta) in (
    (:naive, naive_beta),
    (:reg, reg_beta),
    (:k, k_beta),
    (:kreg, kreg_beta),
)
    best_mse = round(mse(beta, (X_test, y_test)), digits = 2)
    pred_std = round(std(X_test * naive_beta), digits=  2)
    print("\t$name: test_mse=$best_mse ; pred_std = $pred_std\n")
end

# stdout:
#   naive: test_mse=1.04 ; pred_std = 0.83
#   reg: test_mse=0.71 ; pred_std = 0.83
#   k: test_mse=1.04 ; pred_std = 0.83
#   kreg: test_mse=0.68 ; pred_std = 0.83

# although it was expected that the std of the `K` regressions would be lower,
# it is clear that the l1 regularized regressions performed best, specially
# due to the amount of correlated features. However, because the features are 
# highly correlated, the regularized models still weren't able to correctly 
# choose which features to include.
