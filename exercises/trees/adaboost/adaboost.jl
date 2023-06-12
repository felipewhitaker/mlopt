using DecisionTree: load_data
using MLDataUtils: shuffleobs, splitobs
using DecisionTree: build_stump, apply_tree, _weighted_error
using DecisionTree: DecisionTreeClassifier, fit!, predict
using IterTools: product

using JuMP, Ipopt

function adaboost(
    (X_train, y_train), 
    (X_val, y_val), 
    n_estimators::Integer
)
    # adapted from https://github.com/JuliaAI/DecisionTree.jl/blob/dev/src/classification/main.jl
    # FIXME assumes weak learner is `stump` frmo `build_stump`

    metric_stops = (1, 10, 20, 50)
    train_errs, val_errs, wdist = [], [], []

    n_samples = size(X_train, 1)
    weights = fill(1 / n_samples, n_samples)
    estimators, estimators_weights = [], []
    for i in 1:n_estimators
        new_stump = build_stump(
            y_train, X_train, weights; impurity_importance=false
        )
        pred_train = apply_tree(new_stump, X_train)
        err = _weighted_error(y_train, pred_train, weights)

        alpha = 0.5 * log((1 - err) / err)
        weights = weights .* exp.(-alpha .* y_train .* pred_train)
        weights = weights ./ sum(weights)
        push!(estimators, deepcopy(new_stump))
        push!(estimators_weights, alpha)

        if i in metric_stops
            pred_val = adaboost_predict(estimators, estimators_weights, X_val)
            err_train = sum(pred_train .!= y_train) / size(X_train, 1)
            err_val = sum(pred_val .!= y_val) / size(X_val, 1)
            push!(train_errs, err_train)
            push!(val_errs, err_val)
            push!(wdist, weights)
        end
    end
    return estimators, estimators_weights, (train_errs, val_errs, wdist)
end

function adaboost_predict(clfs, weights, X)
    preds = zeros(size(X, 1))
    for (clf, weight) in zip(clfs, weights)
        y_pred = apply_tree(clf, X)
        preds += weight .* y_pred
    end
    return sign.(preds)
end

features, labels = load_data("adult")

subset = 1:10_000 # FIXME low computing resources
features = features[subset, [1, 3, 5, 11, 12, 13]] # keeping numeric features
labels = (labels[subset] .== " >50K") .* 2 .- 1 # converting to -1, 1

# spliting train, val
trainperc, valperc = .6, .2
trainidx, validx, testidx = splitobs(shuffleobs(1:size(features, 1)), (trainperc, valperc))
X_train, y_train = features[trainidx, :], labels[trainidx]
X_val, y_val = features[validx, :], labels[validx]
X_test, y_test = features[testidx, :], labels[testidx]

# train and evaluate adaboost
estimators, estimators_weights, (train_errs, val_errs, weights_dist) = adaboost((X_train, y_train), (X_val, y_val), 50)
adaboost_pred = adaboost_predict(estimators, estimators_weights, X_test)
adaboost_acc = sum(adaboost_pred .== y_test) / size(X_test, 1)

# train and evaluate a naive tree
# allow tree to go as deep as there are estimators with weights
# greater or equal than 10%
max_depth = sum(estimators_weights .>= .1)
forest_model = DecisionTreeClassifier(max_depth=max_depth)
fit!(forest_model, X_train, y_train)
naive_preds = predict(forest_model, X_test)
naive_acc = sum(naive_preds .== y_test) / size(X_test, 1)

println("(Naive, Adaboost) accuracy: ($naive_acc, $adaboost_acc)")
# (Naive, Adaboost) accuracy: (0.8225, 0.8365)

# # FIXME I adapted a SVM implementation with JuMP to take in weights,
# # but it is very slow and lacks regularization
# function svm((X, y); weights=nothing)
#     # from https://github.com/matbesancon/SimpleSVMs.jl/blob/master/src/SimpleSVMs.jl

#     n, p = size(X)

#     m = JuMP.Model(Ipopt.Optimizer)
#     set_silent(m)
#     @variable(m, W[1:p])
#     @variable(m, b)
#     @variable(m, l[1:n] >= 0)
#     @constraint(m, l .>= 1 .- y .* (X * W .+ b))
#     weights = weights === nothing ? ones(n) : weights
#     @objective(m, Min, weights' * l)
#     optimize!(m)

#     svm_predict(X) = sign.(X * value.(W) .+ value.(b))
#     return svm_predict
# end
# svm_model = svm((X_train, y_train))
# y_pred = svm_model(X_test)
# naive_acc = sum(y_pred .== y_test) / size(X_test, 1)