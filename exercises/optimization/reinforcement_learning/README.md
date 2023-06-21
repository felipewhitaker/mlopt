# Exercício

Do livro [Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf), implemente o exemplo 3.3, página 52, do robô de reciclagem. Então apresente a política ótima para os estados e ações.

## Solução

Considerando o enunciado do livro, podemos implementar o problema no Julia da seguinte forma:

```julia
using Match
using POMDPs, QuickPOMDPs, POMDPModels, QMDP, POMDPModelTools, POMDPSimulators
using DiscreteValueIteration
using Statistics: mean, std

STATES = ["high", "low"]
INITIAL_STATE = Deterministic("high")

alpha_, beta_ = 0.8, 0.7
r_search, r_wait = 3., 1.       # FROM EXERCISE
r_recharge, r_rescue = 0., -3.  # FROM BOOK

m = QuickMDP(
    states = STATES,
    initialstate = INITIAL_STATE,
    discount = 0.95,
	actions = function (s = nothing)
        return @match (s) begin
            ("high") => ["wait", "search"]
            _ => ["wait", "search", "recharge"]
        end
    end,
	transition = function (s, a)
        return @match (s, a) begin
            ("high", "search") => SparseCat(
                ["high", "low"], [alpha_, 1 - alpha_]
            )
            ("low", "search") => SparseCat(
                ["low", "high"], [beta_, 1 - beta_]
            )
            ("high", "wait") => Deterministic("high")
            ("low", "wait") => Deterministic("low")
            ("low", "recharge") => Deterministic("high")
            _ => error("Invalid $s-$a pair")
        end
    end,
	reward = function (s, a, sp)
        return @match (s, a, sp) begin
            ("high", "search", _) => r_search
            ("low", "search", "high") => r_rescue
            ("low", "search", "low") => r_search
            ("low", "recharge", "high") => r_recharge
            (_, "wait", _), if s == sp end => r_wait
            _ => error("Invalid $s-$a-$sp triple")
        end
    end,
)

solver = SparseValueIterationSolver(max_iterations = 1000)
policy = solve(solver, m)
```

Resultando em uma política ótima que atribui os seguintes valores para cada estado e ação:

```julia
(a, s) = ("wait", "high")
action(policy, s) = "search"
value(policy, s) = 50.40
value(policy, s, a) = 48.88

(a, s) = ("wait", "low")
action(policy, s) = "recharge"
value(policy, s) = 47.88
value(policy, s, a) = 46.48

(a, s) = ("search", "high")
action(policy, s) = "search"
value(policy, s) = 50.40
value(policy, s, a) = 50.40

(a, s) = ("search", "low")
action(policy, s) = "recharge"
value(policy, s) = 47.88
value(policy, s, a) = 47.40

(a, s) = ("recharge", "high")
action(policy, s) = "search"
value(policy, s) = 50.40
value(policy, s, a) = -Inf

(a, s) = ("recharge", "low")
action(policy, s) = "recharge"
value(policy, s) = 47.88
value(policy, s, a) = 47.88
```

A partir dessa política, podemos simular o problema e guardar a recompensa por simulação, permitindo a visualizar sua distribuição.

![reward histogram](./exercises/optimization/reinforcement_learning/histogram.png)
