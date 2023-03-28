# Execício

Encontre 3 bases de dados de interesse, em formato aplicável para o problema de regressão. Para cada base, separe em treino, validação e teste. Desenvolva e execute o modelo de regressão estudado:

$$
min_{\beta} ||y - X\beta||_p + \lambda ||\beta||_q
$$

Para:

1. p = 1, q = 1
2. p = 1, q = 2
3. p = 2, q = 1
4. p = 2, q = 2

Para cada uma das 4 configurações, defina o parâmetro $\lambda$ via validação cruzada. Ao fim, compare a performance dos 4 modelos finais no conjunto de teste utilizando alguma métrica de erro comum.

## Data

Abaixo estão os conjuntos de dados baixados de [UCI Data Archive](https://archive.ics.uci.edu/ml/index.php). Todos os arquivos foram cortados para ter até 10.000 linhas, dadas limitações de espaço do [Git Large File System](https://docs.github.com/pt/billing/managing-billing-for-git-large-file-storage/viewing-your-git-large-file-storage-usage). Além disso, colunas com variáveis categóricas foram desconsideradas.

* [UCI Machine Learning Repository: Physicochemical Properties of Protein Tertiary Structure Data Set](https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure)

Data is available at [CASP](data/CASP.csv).

```bash
For CASP.csv: Predicting `RMSD` using ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"]
    ::(obj_norm, reg_norm) = (1, 1) - chose lambda = 1.0E+08 with mae, rmse = 2.82E+29, 6.82E+27
    ::(obj_norm, reg_norm) = (1, 2) - chose lambda = 1.0E+08 with mae, rmse = 1.20E+11, 2.89E+09
    ::(obj_norm, reg_norm) = (2, 1) - chose lambda = 1.0E+08 with mae, rmse = 3.88E+08, 1.19E+07
    ::(obj_norm, reg_norm) = (2, 2) - chose lambda = 1.0E+08 with mae, rmse = 1.05E+04, 2.83E+02
```

* [UCI Machine Learning Repository: Average Localization Error (ALE) in sensor node localization process in WSNs Data Set](https://archive.ics.uci.edu/ml/datasets/Average+Localization+Error+%28ALE%29+in+sensor+node+localization+process+in+WSNs)

Data file is available at [MCS](data/mcs_ds_edited_iter_shuffled.csv)

```bash
For mcs_ds_edited_iter_shuffled.csv: Predicting `ale` using ["anchor_ratio", "trans_range", "node_density", "iterations"]
    ::(obj_norm, reg_norm) = (1, 1) - chose lambda = 1.0E+08 with mae, rmse = 5.48E+23, 1.24E+23
    ::(obj_norm, reg_norm) = (1, 2) - chose lambda = 1.0E+08 with mae, rmse = 1.99E+01, 4.64E+00
    ::(obj_norm, reg_norm) = (2, 1) - chose lambda = 1.0E+08 with mae, rmse = 9.63E+07, 3.04E+07
    ::(obj_norm, reg_norm) = (2, 2) - chose lambda = 1.0E+08 with mae, rmse = 1.98E+01, 4.61E+00
```

* [UCI Machine Learning Repository: QSAR fish toxicity Data Set](https://archive.ics.uci.edu/ml/datasets/QSAR+fish+toxicity)

Data file is available at [Fish Toxicity](data/qsar_fish_toxicity.csv). The column header was added: `CIC0`, ` SM1_Dz(Z)`, `GATS1i`, `NdsCH`, `NdssC`, `MLOGP`, `LC50[-LOG(mol/L)]`.

```bash
For qsar_fish_toxicity.csv: Predicting `LC50[-LOG(mol/L)]` using ["CIC0", "SM1_Dz(Z)", "GATS1i", "NdsCH", "NdssC", "MLOGP"]
    ::(obj_norm, reg_norm) = (1, 1) - chose lambda = 1.0E+08 with mae, rmse = 1.67E+23, 1.28E+22
    ::(obj_norm, reg_norm) = (1, 2) - chose lambda = 1.0E+08 with mae, rmse = 7.59E+02, 5.90E+01
    ::(obj_norm, reg_norm) = (2, 1) - chose lambda = 1.0E+08 with mae, rmse = 4.39E+07, 4.01E+06
    ::(obj_norm, reg_norm) = (2, 2) - chose lambda = 1.0E+08 with mae, rmse = 7.59E+02, 5.90E+01
```

* [UCI Machine Learning Repository: Metro Interstate Traffic Volume Data Set](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume)

Data file is available at [Metro Traffic Volume](data/Metro_Interstate_Traffic_Volume.csv).

```bash
For Metro_Interstate_Traffic_Volume.csv: Predicting `traffic_volume` using ["temp", "rain_1h", "snow_1h", "clouds_all"]
    ::(obj_norm, reg_norm) = (1, 1) - chose lambda = 1.0E+08 with mae, rmse = 6.68E+25, 1.50E+24
    ::(obj_norm, reg_norm) = (1, 2) - chose lambda = 1.0E+08 with mae, rmse = 6.96E+06, 1.81E+05
    ::(obj_norm, reg_norm) = (2, 1) - chose lambda = 1.0E+08 with mae, rmse = 1.29E+22, 2.49E+21
    ::(obj_norm, reg_norm) = (2, 2) - chose lambda = 1.0E+08 with mae, rmse = 3.82E+06, 9.72E+04
```

## TODO

[ ] Add data description
