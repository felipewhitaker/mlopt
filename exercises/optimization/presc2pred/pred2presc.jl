# Packages to the two-stage problem
using JuMP
using GLPK
using Distributions
# Packages to the kNN
using NearestNeighbors
using Plots
using StatsPlots

module Exercise

      # module given in class

      #############################################################################################################
      ############################ Defining the coefficients and making the simulation ############################
      #############################################################################################################

      using Random
      using Distributions

      Random.seed!(9)

      dy = 1 # locations
      dx = 3 # dimension of X
      # X is not iid. Instead, it follows an ARMA(2,2) model -> X = Φ1*X1 + Φ2*X2 + U + Θ1*U1 + Θ2*U2

      # Matrixes with the coefficients of the ARMA(2,2) model
      # Coefficients from AR part
      Φ1 = [0.5 -0.9 0;
            1.1 -0.7 0;
            0    0   0.5
      ]

      Φ2 = [ 0   -0.5 0;
            -0.5  0   0;
            0    0   0
      ]

      # Coefficients from MA part
      Θ1 = [0.4  0.8 0;
            1.1 -0.3 0;
            0    0   0
      ]

      Θ2 = [ 0   -0.8 0;
            -1.1 -0.3 0;
            0    0   0
      ]

      # Coefficient for the multivariated normal - the multivariate normal generates erros
      ΣU = [1   0.5 0;
            0.5 1.2 0.5;
            0   0.5 0.8
      ]

      # Coefficient for the multivariated normal
      μ = [0; 0; 0]

      # The last two observations (for the AR part)
      X1 = [1; 1; 1]
      X2 = [1; 1; 1]

      # Demand D -> generated accoding to a factor model -> D = A * (X + δ/4) + (B*X) .* ϵ
      A = 2.5 * [ 0.8 0.1 0.1;
                  0.1 0.8 0.1;
                  0.1 0.1 0.8;
                  0.8 0.1 0.1;
                  0.1 0.8 0.1;
                  0.1 0.1 0.8;
                  0.8 0.1 0.1;
                  0.1 0.8 0.1;
                  0.1 0.1 0.8;
                  0.8 0.1 0.1;
                  0.1 0.8 0.1;
                  0.1 0.1 0.8
      ]

      B = 7.5 * [  0 -1 -1;
                  -1  0 -1;
                  -1 -1  0;
                  0 -1  1;
                  -1  0  1;
                  -1  1  0;
                  0  1 -1;
                  1  0 -1;
                  1 -1  0;
                  0  1  1;
                  1  0  1;
                  1  1  0
      ]

      # Function to simulate Ω scenarios according to the supplementary material from Bertsimas
      function sim_scenarios_Bert(Ω)
            
            X_scenarios = Array{Float64}(undef, Ω, dx)
            Y_scenarios = Array{Float64}(undef, Ω, dy)
            for ω in 1:Ω
                  U = rand(MvNormal(μ, ΣU))
                  U1 = rand(MvNormal(μ, ΣU))
                  U2 = rand(MvNormal(μ, ΣU))
                  X = Φ1*X1 + Φ2*X2 + U + Θ1*U1 + Θ2*U2
                  for i in 1:dx
                        X_scenarios[ω,i] = X[i]
                  end

                  δ = rand(3)
                  ϵ = rand(1)
                  Y = A * (X + δ/4) + (B*X) .* ϵ
                  for i in 1:dy
                        # FIXME `50 + 100 * maximum([0; Y[i]]) >> u`
                        Y_scenarios[ω, i] = 50 + 4 * maximum([0; Y[i]]) 
                  end
            end
            return X_scenarios, Y_scenarios
      end
end

function newsvendor(d::AbstractArray; x::Union{Number, Nothing} = nothing)

      S = size(d, 1)
      global c, r, q, u
      m = Model(GLPK.Optimizer)

      # first stage
      if x === nothing
            @variable(m, x >= 0)
      else
            @assert 0. <= x <= u
            @expression(m, x, x)
      end

      # second stage
      @variable(m, y[1:S] >= 0)
      @variable(m, z[1:S] >= 0)

      @constraint(m, x <= u)
      @constraint(m, y .<= d)
      @constraint(m, y .+ z .<= x)

      @objective(m, Min, c * x - (1/S) * sum(q .* y .+ r .* z))

      return m
end

function newsvendor(d::AbstractArray, X::AbstractMatrix, kdtree::KDTree; x::Union{Number, Nothing} = nothing, k::Integer = 5)

      # update `d` to consider kNN
      idxs, _ = knn(kdtree, X', k, true)
      idxs = [i for idx in idxs for i in idx]
      knn_d = d[idxs]

      m = newsvendor(knn_d, x = x)
      return m
end

function solve(m)
      optimize!(m)
      return objective_value(m), value.(m[:x])
end

function prescr_efficacy(obj_star, obj_hat, obj_SAA)
      return (obj_SAA - obj_hat) / (obj_SAA - obj_star)
end

c = 10
r = 5
q = 25
u = 150

S = 1000 # Total number of scenarios
X, demand = Exercise.Exercise.sim_scenarios_Bert(S)

# a)
obj_true, x_true = solve(newsvendor(demand))
# Out: (-758.792755043592, 58.46957783444397)

# b)
new_X, new_demand = Exercise.Exercise.sim_scenarios_Bert(100)

# c)
obj_SAA, x_SAA = solve(newsvendor(new_demand, x = x_true))
# Out: (-752.351390440655, 58.46957783444397)

# it worsened a bit (-758.80 to -752.35), but it's still a good solution

# d)
# # FIXME copilot got creative
# # KNN(x, y, k) = mean(y[sortperm(vec(sum(abs.(x .- x'), dims=2)), dims=2)[1:k]])

m = newsvendor(demand, new_X, KDTree(X'); k = 5)
obj_knn, x_knn = solve(m)
# Out: (-754.9422442976366, 56.148190447618475)

# using kNN improved the result a bit (-752.35 to -754.94)

# e)
objs_knn, ys_knn = [], []
for k_ in 3:15
      m = newsvendor(demand, new_X, KDTree(X'); k = k_)
      obj_knn, x_knn = solve(m)
      push!(objs_knn, obj_knn)
      push!(ys_knn, x_knn)
end

plot(3:15, objs_knn, label = "kNN", xlabel = "k", ylabel = "Objective value", title = "Objective value vs k")
# `argmin(objs_knn) = best_k = 8`, but not by much (< 1%)

# f)
last_X, last_demand = Exercise.Exercise.sim_scenarios_Bert(200)

obj_SAA, _ = solve(newsvendor(new_demand, x = x_true))
# Out: (-752.351390440655, 58.46957783444397)
obj_hat, _ = solve(newsvendor(demand, new_X, KDTree(X'); k = 13))
# Out: (-754.8526194295482, 55.93017391059969)

prescr_efficacy(obj_true, obj_hat, obj_SAA)
# Out: 0.38830731422232295

# using a kNN improved the result, but just a bit, resulting in a
# a prescriptive efficacy of 38.83%