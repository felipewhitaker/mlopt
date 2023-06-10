using DecisionTree: load_data
using MLDataUtils: shuffleobs, splitobs
using Base.Iterators: product
using DecisionTree: fit!, predict, accuracy
using DecisionTree: nfoldCV_forest, nfoldCV_tree
using DecisionTree: RandomForestClassifier, DecisionTreeClassifier

mean(y) = sum(y) / length(y)

RUN_FULL_GRID_SEARCH = false

features, labels = load_data("adult")

# spliting train, val
trainperc = .1 # FIXME low computing resources
trainidx, testidx = splitobs(shuffleobs(1:size(features, 1)), trainperc)
X_train, y_train = features[trainidx, :], labels[trainidx]
X_test, y_test = features[testidx, :], labels[testidx]

# hyperparameters
n_folds, seed = 3, 42
arg_names = (
    :n_subfeatures,
    :n_trees,
    :partial_sampling,
    :max_depth,
    :min_samples_leaf,
    :min_samples_split,
    :min_purity_increase,
)
hparams = (
    [2 6 12 14], # :n_subfeatures
    [10 15 20], # :n_trees
    0.7, # :partial_sampling
    [5 10 20], # :max_depth
    [5 10 15], # :min_samples_leaf
    2, # :min_samples_split
    .0  # :min_purity_increase
)

min_acc, best_params = Inf, nothing
# FIXME full grid search is too expensive
for args in product(hparams...)
    if !RUN_FULL_GRID_SEARCH
        previous_best_args = (2, 10, 0.7, 5, 10, 2, 0.0)
        println(
            "`RUN_FULL_GRID_SEARCH` was set to ",
            "$RUN_FULL_GRID_SEARCH. ",
            "Previously found best params: ",
            "$previous_best_args\n"
        )
        best_params = previous_best_args
        break
    end
    acc = nfoldCV_forest(
        y_train, 
        X_train,
        n_folds,
        args...;
        verbose = false,
        rng = seed
    )
    if mean(acc) < min_acc
        min_acc = mean(acc)
        best_params = args
    end
end

# default pruning purity with forest's best params
# without arguments `n_subfeatures` (which makes it 
# a bit invalid, as the tree will have more information), 
# `n_trees`, `partial_sampling` from `best_params`
tree_arg_names = (:pruning_purity_threshold, arg_names[4:end]...)
best_tree_params = (1.0, best_params[4:end]...)
tree_valacc = nfoldCV_tree(
    y_train, 
    X_train, 
    n_folds,
    best_tree_params...;
    verbose = false,
    rng = seed
)
tree_valacc = mean(tree_valacc)

accs = round.((min_acc, tree_valacc), digits=3)
println(
    "(forest_acc, tree_acc)=$accs\n",
    "best_params=$best_params"
)
# Out:
# (forest_acc, tree_acc) = (0.803, 0.849)
# best_params = (2, 10, 0.7, 5, 10, 2, 0.0)

# Interestingly, the tree is better than the forest, probably
# due to the fact that it receives more information (all features)

test_acc = []
for (name, model, pnames, params) in (
    ("RandomForestClassifier", RandomForestClassifier, arg_names, best_params),
    ("DecisionTreeClassifier", DecisionTreeClassifier, tree_arg_names, best_tree_params),
)
    named_params = Dict(k => v for (k, v) in zip(pnames, params))
    clf = model(;named_params...)
    fit!(clf, X_train, y_train)
    y_pred = predict(clf, X_test)
    acc = accuracy(y_pred, y_test)
    push!(test_acc, (name, round(acc, digits = 3)))
end
println("test_accs=$(test_acc)")
# Out: test_accs = Any[("RandomForestClassifier", 0.812), ("DecisionTreeClassifier", 0.849)]

# and this is further confirmed by the test accuracy, as
# the tree performed better than the forest