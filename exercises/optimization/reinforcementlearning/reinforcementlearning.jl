using Match
using POMDPs, QuickPOMDPs, POMDPModels, QMDP, POMDPModelTools, POMDPSimulators
using DiscreteValueIteration
using Statistics: mean, std

STATES = ["high", "low"]
ACTIONS = ["wait", "search", "recharge"]
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
            _ => ACTIONS
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

for a in ACTIONS, s in STATES
    @show a, s
    @show action(policy, s)
    @show value(policy, s)
    @show value(policy, s, a)
    println()
end

nsims, max_steps = 1_000, 100
sims = [sum(r for r in stepthrough(m, policy, "r", max_steps = max_steps)) for nsim in 1:nsims]

using Plots
using Distributions, StatsPlots

mu_, sigma_ = mean(sims), std(sims)
p = histogram(sims, normalize = true)
dist = Normal(mu_, sigma_)
plot!(x -> pdf(dist, x))
savefig(p, "./exercises/optimization/reinforcement_learning/histogram.png")

