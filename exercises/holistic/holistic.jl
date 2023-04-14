using JuMP
using HiGHS
using Distributions
using LinearAlgebra

n_data = 100
n_feat, n_corr = 3, 2 # number of (features, fake features)
total_feat = n_feat + n_corr
m = 2 # number of variables to select

Xt = rand(Normal(0, 1), n_data, n_feat)
# coef = [1 1 0]' # rand(Uniform(0, 1), size(Xt, 2), n_corr)
Xf = hcat(Xt * [1 0 0]', Xt * [0 1 0]') .+ rand(Normal(0, 0.04), n_data, 2)
X = hcat(Xt, Xf)

# B = vcat(rand(Uniform(0, 1), n_feat), zeros(n_corr))
# y = X * B + rand(Normal(0, 1), size(Xt, 1))
ev = eigvals(X'X)
V = eigvecs(X'*X)[:, sortperm(abs.(ev))][:, 1:m] # size = (p, m) = (total_feat, m)

M = 1e3
p = total_feat
delta = 1e-6

model = Model(HiGHS.Optimizer)
function scalar_problem(model)
    @variable(model, z[1:p], Bin)
    @variable(model, a[1:p])
    @variable(model, theta[1:m])

    @constraint(model, [j in 1:p], a[j] == sum(theta[i] * V[j, i] for i in 1:m))

    @constraint(model, sum(theta[i] for i in 1:m) >= delta)
    @constraint(model, sum(theta[i] for i in 1:m) <= - delta)

    @constraint(model, [j in 1:p], a[j] <= M * z[j])
    @constraint(model, [j in 1:p], a[j] >= -M * z[j])

    @objective(model, Min, sum(z))
    optimize!(model)
    value.(z)
end
scalar_problem(model)


model = Model(HiGHS.Optimizer)
function problem(model)

    global p, m
    global M, delta

    @variable(model, theta[1:m])
    @variable(model, z[1:p], Bin)
    @variable(model, a[1:p])

    @constraint(model, a .== V * theta)
    @constraint(model, sum(theta) >= delta)
    @constraint(model, sum(theta) <= -delta)
    @constraint(model, a .>= - M .* z)
    @constraint(model, a .<= M .* z)

    # @constraint(model, theta .>= delta)
    # @constraint(model, theta .<= -delta)

    @objective(model, Min, sum(z))
    optimize!(model)

    if JuMP.termination_status(model) == JuMP.OPTIMAL
        return (ob = objective_value(model), a = value.(a), z = value.(z), theta = value.(theta))
    end
    return (ob = nothing, a = nothing, z = nothing, theta = nothing)
end

ob, a, z, theta = problem(model)
