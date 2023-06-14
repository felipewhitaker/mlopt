# Exercício

Implemente o algoritmo de detecção de ϵ-multicolinearidade discutido em sala, seguindo o algoritmo 5.1 e a equação do problema de otimização 5.3 no livro. O objetivo é realizar um simples teste da capacidade de detectar colunas redundantes numa matriz de regressores X. Para isso, siga os passos:

1. Simule uma matriz X com dimensões n por p a partir de uma normal multivariada;
2. Acrescente uma/algumas coluna(s) a essa matriz X, onde X[p+k] = b'X + u, u ∼ N(0, σ2), k ≥ 1 e b um vetor de coeficientes;
3. Considere que X[l] identifica a coluna l da matriz X. Isto é, as colunas adicionadas serão combinações lineares das colunas originais acrescidas de um ruído;
4. Execute pelo menos 2 iterações do algoritmo definido;
5. Avalie o resultado.

O objetivo do exercício é identificar os vetores esparsos que reproduzam os autovetores associados aos autovalores com valor menor do que ϵ em X.
