# Exercício

A atividade requer a implementação e reoslução de um modelo de regressão esparsa utilizando as mesmas bases de dados da aula anterior. É necessário implementar a minimização da soma dos quadrados dos resíduos, sujeito a uma restrição de norma L$_0$ que limita o número de variáveis explicativas selecionadas e um termo de regularização de norma L$_2$ para controlar o *trade-off* entre ajuste e complexidade do modelo.

$$
min ||y - X\beta||_2^2 + \frac{1}{2\lambda} ||\beta||_2^2 \\
s.t. ||\beta||_0 \le K
$$

Além disso, deve-se implementar uma modificação que utilize restrições de big M para substituir a restrição de norma L$_0$. Crie uma heurística para calcular os valores de big M para cada explicativa.

Ao final, deve-se comparar o tempo de solução dos dois problemas e verificar se houve alguma diferença significativa. Para a seleção das variáveis explicativas, é permitido fixar um número ótimo com base no conhecimento prévio sobre as bases de dados utilizadas.

Finalmente implemente uma métrica de decisão do número ótimo de explicativas (K) tal como AIC, BIC, AICC ou Cross Validation e compare os resultados com lasso e com ada lasso.