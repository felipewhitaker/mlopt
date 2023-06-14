using JuMP
using Ipopt
using Distributions
using DecisionTree: load_data
using DecisionTree: RandomForestClassifier, fit!, predict

optimizer = Ipopt.Optimizer

function problem((X, y), c::Float64, s_prev::Vector{Float64}; gamma::Float64=.0)
    
    N, p = size(X)

    global optimizer
    m = Model(optimizer)

    @variable(m, eta)
    @variable(m, epsilon[1:N] >= .0)
    @variable(m, beta[1:p])
    @variable(m, alpha[1:N])

    # FIXME very similar to subproblem objetive
    # FIXME refactor to a function to be the cut
    @constraint(m, eta >= c - (gamma / 2 * alpha' * (X * X') * alpha)' * (s .- sprev))

    @constraint(m, epsilon .>= 1 .- y .* (X * beta))
    @constraint(m, y .* alpha .>= -1.0)
    @constraint(m, y .* alpha .<= -1.0)

    @objective(m, Min, eta)
    return (model = m)

end

function subproblem(s, (X, y); gamma::Float64=.0)
    global optimizer
    N = size(X, 1)

    m = Model(optimize)

    @variable(m, alpha[1:N])
    @variable(m, e[1:N], Bin)

    @constraint(m, e' * a == 0)
    
    @objective(m, Max, -gamma / 2 .* s * alpha' * (X * X') * alpha - y' * alpha)
    optimize!(m)
    return (obj_value = objective_value(m), alpha = value.(alpha))
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
eta, c = -1., .0            # FIXME check if there are any heuristics
s = zeros(size(X_train, 1)) # FIXME check if there are any heuristics
model = problem((X_train, y_train), 0.0, s)
optimize!(model)
while eta <= c
    c, _ = subproblem(s, (X_train, y_train))
    @constraint(model, 1) # FIXME add cut
    eta = value.(model[:eta])
end

# TODO how to make out of sample predictions?

# RandomForest comparison
baseline = RandomForestClassifier()
fit!(baseline, X_train, y_train)
y_pred = predict(baseline, X_test)