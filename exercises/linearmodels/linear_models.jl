using CSV, JuMP, Ipopt, MLDataUtils, DataFrames, Printf

function read_data(path::String, file::String) #::(DataFrames, Integer)
    # TODO instead of dropping, it should consider categorical
    filepath = joinpath(path, file)
    data = CSV.read(filepath, DataFrame)
    if file == "CASP.csv"
        ycol = 1
    elseif file == "mcs_ds_edited_iter_shuffled.csv"
        ycol = 5
    elseif file == "qsar_fish_toxicity.csv"
        ycol = 7
    elseif file == "Metro_Interstate_Traffic_Volume.csv"
        ycol = 5
    else
        throw("file = `$file` not recognized")
    end
    return data, ycol
end

lm = (X, beta) -> X * beta
mse = (yt, yh) -> sum((yt .- yh) .^ 2)

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

    beta = @variable(model, [1:ncol(trainval)-1])

    best_mse = Inf
    best_beta = nothing
    best_lambda = nothing
    for (obj_norm, reg_norm) in norms

        for lambda in lambdas

            obj_mse = []
            for (train, val) in kfolds(trainval, K)

                X_train, y_train = Matrix(train[!, Not(ycol)]), train[!, ycol]

                @objective(
                    model,
                    Min,
                    sum(
                        (y_train .- lm(X_train, beta)) .^ obj_norm
                    ) + lambda * sum(beta .^ reg_norm)
                )
                optimize!(model)
                cur_beta = value.(beta)

                X_val, y_val = Matrix(val[!, Not(ycol)]), val[!, ycol]
                push!(obj_mse, mse(y_val, lm(X_val, cur_beta)))
            end

            cur_mse = sum(obj_mse) / K
            if cur_mse < best_mse
                best_lambda = lambda
                best_beta = cur_beta
            end
            # @printf("lambda = %.1E\t%.1E\n", lambda, cur_mse)
        end

        X_test, y_test = Matrix(test[!, Not(ycol)]), test[!, ycol]
        preds = lm(X_test, best_beta)
        test_mse = round(mse(y_test, preds), digits=2)
        @printf(
            "\t::(obj_norm, reg_norm) = (%d, %d) - chose lambda = %.1E with mse = %.2E\n",
            obj_norm, reg_norm, best_lambda, test_mse
        )
    end

    delete(model, beta)
    print("\n")
end