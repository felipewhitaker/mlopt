using DecisionTree: load_data, DecisionTreeClassifier
using MLDataUtils: shuffleobs, splitobs
using MLBase: mode as mlmode

using JuMP
using GLPK

# tree = 
# size = 2^depth - 1
# [
#     st, 
#     nd_left, nd_right, 
#     rd_left_left, rd_left_right, rd_right_left, rd_right_right
# ]
# [
#     1, 
#     2, 3, 
#     4, 5, 6, 7
# ]

# helper functions
parent(i) = div(i, 2)
anc_l(i) = i == 1 ? [] : i % 2 == 0 ? [parent(i); anc_l(parent(i))] : anc_l(parent(i))
anc_r(i) = i == 1 ? [] : i % 2 == 1 ? [parent(i); anc_r(parent(i))] : anc_r(parent(i))

# loading and pre processing data
features, labels = load_data("iris")

# scaling features
min = copy(minimum(eachrow(features))')
max = copy(maximum(eachrow(features))')
scaled_features = (features .- min) ./ max

# transforming labels
possible_labels = unique(labels)
int_labels = indexin(labels, possible_labels)

# spliting train, val
trainperc = .4 # FIXME low computing resources
trainidx, validx = splitobs(shuffleobs(1:size(features)[1]), trainperc)
X_train, y_train = scaled_features[trainidx, :], int_labels[trainidx]
# FIXME how to use the optimization model with the validation set?
X_val, y_val = scaled_features[validx, :], int_labels[validx]

function oct(
    (X, y)::Tuple{AbstractMatrix, AbstractArray},
    possible_labels::Vector,
    depth::Integer,
    Nmin::Integer;
    alpha::Float64 = 1e-1
)

    # Define indexes
    p = 1:size(X)[end]
    n = size(X)[1]
    K = 1:length(possible_labels)
    T = 2^(depth + 1) - 1

    Tb = 1:(div(T, 2))
    Tl = (div(T, 2)+1):T

    # Define constants
    Lhat = sum(y .== mlmode(y))

    X_sort = sort(X, dims = 1)
    epsilon = [
        minimum([
            X_sort[i + 1, j] - X_sort[i, j] 
            for i = 1:(n-1) if X_sort[i + 1, j] != X_sort[i, j]
        ])
        for j in p
    ]
    epsilon_min = minimum(epsilon)
    M1 = 1 + maximum(epsilon)
    M2 = 1

    # Define model
    model = Model(GLPK.Optimizer)

	@variable(model, a[m in p, t in Tb], Bin)
    @variable(model, b[t in Tb] >= 0)
    @variable(model, c[k in K, t in Tl])
	@variable(model, d[t in Tb], Bin)

    @variable(model, L[t in Tl] >= 0)
    @variable(model, l[t in Tl])

    @variable(model, Nt[t in Tl])
    @variable(model, Nkt[k in K, t in Tl])

    @variable(model, z[i in 1:n, t in Tl])

	@constraint(model, [t in Tl, k in K], L[t] >= Nt[t] - Nkt[k, t] - n * (1 - c[k, t]))
	@constraint(model, [t in Tl, k in K], L[t] <= Nt[t] - Nkt[k, t] - n * c[k, t])
	@constraint(model, [t in Tl, k in K], Nkt[k, t] == 1 / 2 * sum((1 + y[i]) * z[i, t] for i in n))
	@constraint(model, [t in Tl], Nt[t] == sum(z[:, t]))
	@constraint(model, [t in Tl], sum(c[:, t]) == l[t])
	@constraint(model, [t in Tb], sum(a[:, t]) == d[t])
	@constraint(model, [t in Tb], b[t] <= d[t])
	@constraint(model, [t in Tl], z[:, t] .<= l[t])
	@constraint(model, [t in Tl], sum(z[:, t]) >= Nmin * l[t])
	@constraint(model, [i in 1:n], sum(z[i, :]) == 1)

	for t in Tl
        @constraint(model, [i in 1:n, j in anc_l(t)], a[:, j]'X[i, :] <= b[j] + M1 * (1 - z[i, t]) .- epsilon_min)
		@constraint(model, [i in 1:n, j in anc_r(t)], a[:, j]'X[i, :] >= b[j] - M2 * (1 - z[i, t]))
	end

    @objective(model, Min, (1 / Lhat) * sum(L) + alpha * sum(d))
    return model
end

depth, Nmin = 2, 1
model = oct((X_train, y_train), possible_labels, depth, Nmin)
optimize!(model)
