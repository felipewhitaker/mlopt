using DecisionTree: load_data
using MLDataUtils: shuffleobs, splitobs
using DecisionTree: DecisionTreeClassifier, fit!, predict
using IterTools: product

using ScikitLearn: fit!, predict, @sk_import
@sk_import DecisionTree: DecisionTreeClassifier

using JuMP
using Ipopt
optimizer = Ipopt.Optimizer

function svm((X, y); weights=nothing)
    # from https://github.com/matbesancon/SimpleSVMs.jl/blob/master/src/SimpleSVMs.jl

    n, p = size(X)

    m = JuMP.Model(optimizer)
    set_silent(m)
    @variable(m, W[1:p])
    @variable(m, b)
    @variable(m, l[1:n] >= 0)
    @constraint(m, l .>= 1 .- y .* (X * W .+ b))
    weights = weights === nothing ? ones(n) : weights
    @objective(m, Min, weights' * l)
    optimize!(m)

    svm_predict(X) = sign.(X * value.(W) .+ value.(b))
    return svm_predict
end

function adaboost(
    clf, 
    (X, y), 
    (X_val, y_val), 
    n_estimators::Integer
)
    n_samples = size(X, 1)
    weights = fill(1 / n_samples, n_samples)
    metrics = []
    estimators, estimators_weights = [], []
    for i in 1:n_estimators
        svm_model = clf((X, y); weights=weights) # FIXME specific to SVM
        y_pred = svm_model(X) # FIXME specific to SVM
        err = sum(weights .* (y_pred .!= y))

        # # FIXME should break training
        # if err > 0.5
        #     break
        # end

        alpha = 0.5 * log((1 - err) / err)
        weights = weights .* exp.(-alpha .* y .* y_pred)
        weights = weights ./ sum(weights)
        push!(estimators, deepcopy(svm_model))
        push!(estimators_weights, alpha)

        if i in (1, 10, 20, 50)
            pred_val = adaboost_predict(estimators, estimators_weights, X_val)
            err_val = sum(pred_val .!= y_val) / size(X_val, 1)
            push!(metrics, (err, err_val, deepcopy(estimators_weights)))
        end
    end
    return estimators, estimators_weights, metrics
end

function adaboost_predict(clfs, weights, X)
    preds = zeros(size(X, 1))
    for (clf, weight) in zip(clfs, weights)
        y_pred = clf(X)
        preds += weight .* y_pred
    end
    # FIXME where should `sign.` be called, as it is
    # model (SVM) specific?
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

naive_model = svm((X_train, y_train))
y_pred = svm_model(X_test)
naive_acc = sum(y_pred .== y_test) / size(X_test, 1)

estimators, estimators_weights, metrics = adaboost(svm, (X_train, y_train), (X_val, y_val), 50)

y_pred = adaboost_predict(estimators, estimators_weights, X_test)
adaboost_acc = sum(y_pred .== y_test) / size(X_test, 1)

println("(Naive, Adaboost) accuracy: $naive_acc, $adaboost_acc")
# Out: (Naive, Adaboost) accuracy: 0.7585, 0.8095
