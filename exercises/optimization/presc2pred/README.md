# Exercício

Considere o Problema do Jornaleiro, um problema de decisão em dois estágios: toda manhã o jornaleiro precisa decidir qual é a quantidade de jornais `x` a ser comprada por um preço `c` de uma editora local. Como não possui poder de compra infinito, o Jornaleiro pode comprar até `u` jornais por dia. Após sua decisão da quantidade de jornais `x`, ocorre a realização da demanda incerta `D`. Cada jornal pode então ser vendido por `q`, e há a opção de devolver os jornais à editora por um preço residual `r`, inferior a `q` e a `c`.

Seja sua formulação:

$$
\begin{align}
    \underset{0 \le x \le u}{\text{min}} \textbf{E}(Q(z, \tilde{d})&) \\
    s.t \quad Q(z, d) &= \underset{y, z}{\text{min}} -qy -rz \\
    y &\le d \\
    y + z &\le x \\
    y \ge 0 &, z \ge 0
\end{align}
$$

Seja:

```julia
c = 10
r = 5
q = 25
u = 150
```

Simule 1.000 cenários para X e Y, de forma que não sejam independentes e identicamente distribuídos (iid). O arquivo para geral uma simulação se encontra no arquivo `exercício.jl`, sendo:

```julia
using Distributions 

not_iid = phi' * X # size(phi) = (2, 3)
unif_error = [1 theta...]' * rand(MvNormal(Mu, Sigma), 3)
X = not_iid + unif_error

Y = A + (X * rand(Uniform(0, 1), 3) / 4) + B * X .* rand(Uniform(0, 1), 1)
scenarios = 50 + Y * 100 # [50, 150]
```

Assim:

1. Resolva o modelo de dois estágios para encontrar os valores ótimos. 
2. Gere 100 novos 𝑋 e um novo 𝑌. 
3. Dado  que  o  ótimo  é  conhecido  (letra  b),  encontre  o  custo  levando  em consideração o novo cenário e o 𝑥𝑆𝐴𝐴. 
4. Resolva o problema com o kNN para 𝑘 =5. 
5. Varie 𝑘 entre 3 e 15. Resolva o problema com o kNN para esses valores de 𝑘. Faça  o  gráfico  da  evolução  dos  custos  e  dos  𝑘  𝑌  mais  próximos  conforme aumentamos 𝑘. 
6. Calcule P para uma nova amostra (200 cenários) e para 𝑘 = 13. 
