using Printf
using CSV, DataFrames
using JuMP, Ipopt
using MLDataUtils

function read_data(path::String, file::String)
    # categorical features have been dropped, but should've been considered
    y = Dict(
        "CASP.csv" => 1,
        "mcs_ds_edited_iter_shuffled.csv" => 5,
        "qsar_fish_toxicity.csv" => 7,
        "Metro_Interstate_Traffic_Volume.csv" => 5,
    )
    if haskey(y, file)
        ycol = y[file]
    else
        throw("file = `$file` not recognized")
    end 
    filepath = joinpath(path, file)
    data = CSV.read(filepath, DataFrame)
    return data, ycol
end

function get_X(df::SubDataFrame, ycol::Integer)
    X = Matrix(df[!, Not(ycol)])    # allow vector multiplication
    hcat(ones(size(X)[1]), X)       # add intercept
end

lm = (X, beta) -> X * beta
mse = (yt, yh) -> sum((yt .- yh) .^ 2)
mae = (yt, yh) -> sum(abs.(yt .- yh))

datadir = "./exercises/linearmodels/data/"

lambdas = 10.0 .^ range(-2, 8)
norms = [(i, j) for i in range(1, 2) for j in range(1, 2)]
K = 5

model = JuMP.Model(Ipopt.Optimizer)
set_silent(model)

for file in filter(file -> endswith(file, ".csv"), readdir(datadir))

    data, ycol = read_data(datadir, file)

    @printf("For %s: Predicting `%s` using %s\n", file, names(data)[ycol], names(data)[Not(ycol)])
    trainval, test = splitobs(data, 0.8)

    beta = @variable(model, [1:ncol(trainval)])

    best_mse = Inf
    best_beta = nothing
    best_lambda = nothing
    for (obj_norm, reg_norm) in norms

        for lambda in lambdas

            obj_mse, lambda_betas = [], []
            for (train, val) in kfolds(trainval, K)

                X_train, y_train = get_X(train, ycol), train[!, ycol]

                @objective(
                    model,
                    Min,
                    sum(
                        (y_train .- lm(X_train, beta)) .^ obj_norm
                    ) + lambda * sum(beta .^ reg_norm)
                )
                optimize!(model)
                cur_beta = value.(beta)

                X_val, y_val = get_X(val, ycol), val[!, ycol]
                push!(obj_mse, mse(y_val, lm(X_val, cur_beta)))
                push!(lambda_betas, cur_beta)
            end

            cur_mse = sum(obj_mse) / K
            if cur_mse < best_mse
                best_lambda = lambda
                best_beta = vcat(sum(hcat(lambda_betas...) ./ K, dims = 2))
            end
            # @printf("lambda = %.1E\t%.1E\n", lambda, cur_mse)
        end

        X_test, y_test = get_X(test, ycol), test[!, ycol]
        preds = lm(X_test, best_beta)
        @printf(
            "\t::(obj_norm, reg_norm) = (%d, %d) - chose lambda = %.1E with mae, rmse = %.2E, %.2E\n",
            obj_norm, reg_norm, best_lambda, round(mae(y_test, preds), digits=2), round(mse(y_test, preds), digits=2) ^ .5
        )
    end

    delete(model, beta)
    print("\n")
end