using DecisionTree: load_data
using MLDataUtils: shuffleobs, splitobs
using DecisionTree: build_stump, apply_tree, _weighted_error
using DecisionTree: DecisionTreeClassifier, fit!, predict
using IterTools: product

using Plots

using JuMP, Ipopt

using Random: seed!
seed!(42)

function adaboost(
    (X_train, y_train), 
    (X_val, y_val), 
    n_estimators::Integer
)
    # adapted from https://github.com/JuliaAI/DecisionTree.jl/blob/dev/src/classification/main.jl
    # FIXME assumes weak learner is `stump`

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
            # # FIXME should train error consider the new 
            # # estimators or shouldn't it?
            # pred_train = adaboost_predict(
            #     estimators[1:(end - 1)], 
            #     estimators_weights[1:(end - 1)], 
            #     X_val
            # )
            err_train = sum(pred_train .!= y_train) / size(X_train, 1)

            pred_val = adaboost_predict(estimators, estimators_weights, X_val)
            err_val = sum(pred_val .!= y_val) / size(X_val, 1)
            push!(train_errs, err_train)
            push!(val_errs, err_val)
            push!(wdist, deepcopy(weights))
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

subset = 1:10_000
features = features[subset, [1, 3, 5, 11, 12, 13]] # keeping only numeric features
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
# (Naive, Adaboost) accuracy: (0.801, 0.8185)

# make visualizations
n_estimators = [1, 10, 20, 50]

p = plot(n_estimators, train_errs, label = "train_error")
plot!(p, n_estimators, val_errs, label = "val_error")
xlabel!(p, "Estimators")
ylabel!(p, "Error")
savefig(p, "./imgs/estimators_error.png")

# as expected, the non-weighted train errors grows (new estimators have to 
# learn less and less), but the validation error keeps lowering (as new 
# learners' weights are smaller)

p = plot(
    histogram.(weights_dist)..., 
    layout = (4, 1), 
    legend = false, 
    sharex = true
)
p[:plot_title] = "Train sample weights"
plot(p)
savefig(p, "./imgs/train_sample_weights.png")

# as expected, the train weights start uniform, and then gets concentrated
# towards zero as most train observations have been learned by the
# weak learners

p = histogram(estimators_weights, xlabel = "learner weight", ylabel = "frequence", label = "learner weight")
savefig(p, "./imgs/learner_weights")

# sum(estimators_weights .>= .2) # Out: 4
# finally, it is also interesting to see that most of the weights for the
# estimators are close to zero, with only 4 greater than `.2`, as they 
# become responsible for learning less and less
