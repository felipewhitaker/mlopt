using MLDataUtils
@sk_import DecisionTree: DecisionTreeClassifier
using ScikitLearn.GridSearch: GridSearchCV

accuracy(y, yhat) = sum(y .== yhat) / length(y)

features, labels = load_data("adult")

hparams = Dict(
    :n_subfeatures => [2 6 12 14], 
    :n_trees => [10 15 20], 
    :partial_sampling => 0.7, 
    :max_depth => [5 10 20], 
    :min_samples_leaf => [5 10 15], 
    :min_samples_split => 2, 
    :min_purity_increase => 0
)

GridSearchCV

kfolds(trainval, K)