# Exercício

Através do `DecisionTree.jl`, aplicar o algoritmo Random Forest no dataset `adult`. Para tal, realizar um Cross Validation sobre as opções de hiperparâmetros descritas abaixo fazendo o uso de 3 folds. Ao final compare a acurácia da melhor configuração obtida com o uso de uma única árvore de decisão (`nfoldCV_tree()`) com os mesmos hiperparâmetros.

Dataset: [adult](https://archive.ics.uci.edu/ml/datasets/adult) (disponível no `DecisionTree.jl`)

```julia
features, labels = load_data("adult")
```

## Hiperparâmetros

- n_subfeatures: `[2, 6, 12, 14]`
- n_trees: `[10, 15, 20]`
- partial_sampling: `0.7`
- max_depth: `[5, 10, 20]`
- min_samples_leaf: `[5, 10, 15]`
- min_samples_split: `2`
- min_purity_increase: `0`
- Métrica de comparação: `average accuracy`
