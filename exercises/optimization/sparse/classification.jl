using JuMP
using Ipopt, GLPK, HiGHS
using MLDataUtils: shuffleobs, splitobs
using Distributions
using DecisionTree: load_data
using DecisionTree: RandomForestClassifier, fit!, predict

using Ipopt, Juniper
optimizer = optimizer_with_attributes(
    Juniper.Optimizer, 
    "nl_solver" => Ipopt.Optimizer,
    "mip_solver" => HiGHS.Optimizer
)

function problem((X, y); K::Integer = 3, lambda::Float64 = 1e-2)

    global optimizer
    m = Model(optimizer)

    n, p = size(X)
    @variable(m, w[1:p])
    @variable(m, b)
    @variable(m, ϵ[1:n] >= 0)
    @variable(m, z[1:p], Bin)

    @constraint(m, y .* (X * w .+ b) .>= 1 .- ϵ)
    @constraint(m, (1 .- z) .* w .== 0)
    @constraint(m, sum(z) <= K)

    @objective(m, Min, sum(ϵ) + lambda * sum(w .^ 2))

    optimize!(m)
    return value.(w), value(b), objective_value(m)
end

features, labels = load_data("adult")

# preprocessing
subset = 1:10_000
features = features[subset, [1, 3, 5, 11, 12, 13]] # keeping only numeric features
labels = (labels[subset] .== " >50K") .* 2 .- 1 # converting to -1, 1

# spliting train, val
trainperc, valperc = .6, .2
trainidx, validx, testidx = splitobs(shuffleobs(1:size(features, 1)), (trainperc, valperc))
X_train, y_train = features[trainidx, :], labels[trainidx]
X_val, y_val = features[validx, :], labels[validx]
X_test, y_test = features[testidx, :], labels[testidx]

# optimization model
w, b, _ = problem((X_train, y_train))
acc_svm = mean(y_test .* (w' .* X_test .+ b) .> 0)
# Out: 0.751

# RandomForest comparison
# to try to be fair, we use the same number of features per tree
baseline = RandomForestClassifier(n_subfeatures = 3)
fit!(baseline, X_train, y_train)
y_pred = predict(baseline, X_test)
acc_rf = mean(y_pred .== y_test)
# Out: 0.812