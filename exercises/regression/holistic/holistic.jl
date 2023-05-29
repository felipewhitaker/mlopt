using JuMP
using HiGHS
using Distributions
using LinearAlgebra

n_data = 100
n_feat, n_corr = 3, 2 # number of (features, fake features)
total_feat = n_feat + n_corr
m = n_corr # number of variables to select

Xt = rand(Normal(0, 1), n_data, n_feat)
coef = rand(Uniform(0, 1), size(Xt, 2), n_corr)
Xf = Xt * coef .+ rand(Normal(0, 0.04), n_data, 2)
X = hcat(Xt, Xf)

ev = eigvals(X'X)
V = eigvecs(X'*X)[:, sortperm(abs.(ev))][:, 1:m] # size = (p, m) = (total_feat, m)

M = 1e1
p = total_feat
delta = 1e-1

model = Model(HiGHS.Optimizer)
set_silent(model)
function problem(model)

    global p, m
    global M, delta

    @variable(model, theta[1:m])
    @variable(model, z[1:p], Bin)
    @variable(model, a[1:p])
    @variable(model, y)

    @constraint(model, a .== V * theta)
    # following given tip
    @constraint(model, y >= delta)
    @constraint(model, y >= sum(theta))
    @constraint(model, y <= -sum(theta))
    # big M strategy for absolute value constraints
    @constraint(model, a .>= - M .* z)
    @constraint(model, a .<= M .* z)

    @objective(model, Min, sum(z))
end

problem(model)

for i in 1:2
    println("Iteration $i")
    optimize!(model)

    if termination_status(model) == MOI.INFEASIBLE
        println("Infeasible")
        break
    end

    z = value.(model[:z])
    println("z = $z")
    
    not_zero = (z .!= 0)
    @constraint(model, z' * not_zero <= sum(not_zero) - 1)
end

# although there are two correlated features in the data, the algorithm 
# was not able to identify that the last two features should have been
# zero; reaching infeability on the second iteration. 
