using JuMP
using Ipopt
using GLPK

# Define some constants
n_input = 2 # number of input nodes
n_hidden = 3 # number of hidden nodes
n_output = 1 # number of output nodes
n_data = 4 # number of data samples

# Define some sample data (XOR problem)
X = [0 0; 0 1; 1 0; 1 1] # input matrix
y = [0; 1; 1; 0] # output vector

# Define a model
model = Model(GLPK.Optimizer)

# Define variables
@variable(model, W1[i=1:n_input, j=1:n_hidden]) # weight matrix from input to hidden layer
@variable(model, b1[j=1:n_hidden]) # bias vector for hidden layer
@variable(model, W2[j=1:n_hidden, k=1:n_output]) # weight matrix from hidden to output layer
@variable(model, b2[k=1:n_output]) # bias vector for output layer
@variable(model, z[i=1:n_data], Bin)

@NLobjective(
    model,
    Min,
    sum(
        (y .- (((1 .- z) .* (X * W1 .+ b1')) * W2 .+ b2'))
    )
)

optimize!(model)

using JuMP, GLPK
using Distributions

n_input = 3 # number of input units
n_hidden = 10 # number of hidden units
n_output = 2 # number of output units
n_data = 1000 # number of data points

X = rand(Normal(0, 1), n_data, n_input) # input data matrix of size n_data x n_input
Y = X * rand(Uniform(0, 1), n_input, n_output) .> .5 # target data matrix of size n_data x n_output

model = Model(GLPK.Optimizer)

@variable(model, W1[1:n_hidden, 1:n_input]) # weights of the first layer
@variable(model, b1[1:n_hidden]) # biases of the first layer
@variable(model, W2[1:n_output, 1:n_hidden]) # weights of the second layer
@variable(model, b2[1:n_output]) # biases of the second layer
@variable(model, z[1:n_data, 1:n_hidden], Bin) # binary variable for ReLU

@variable(model, h[1:n_data, 1:n_hidden]) # hidden layer output before ReLU
@variable(model, a[1:n_data, 1:n_hidden]) # hidden layer output after ReLU
@variable(model, y[1:n_data, 1:n_output]) # network output before softmax

# additional variables for linearizing softmax
@variable(model, t[1:n_data]) # auxiliary variable for log-sum-exp trick
@variable(model, w[1:n_data, 1:n_output]) # auxiliary variable for McCormick envelopes

# bounds on t and a (can be chosen arbitrarily)
tmin = -10.0
tmax = 10.0
amin = 0 # minimum(y)
amax = 1 # maximum(y)

@constraint(model, h .== X * W1' .+ b1') # hidden layer output before ReLU
@constraint(model, a .== (1 .- z) .* h) # hidden layer output after ReLU
@constraint(model, y .== a * W2' .+ b2') # network output before softmax

# additional constraints for linearizing ReLU
M = 1000 # a large constant
@constraint(model, h .<= M * z)
@constraint(model, h .>= 0)
@constraint(model, z .>= 0)

# additional constraints for linearizing softmax
@constraint(model, tmin .<= t .<= tmax) # bounds on t

# lower and upper bound constraints on t using log-sum-exp trick
for i in 1:n_data
    @NLconstraint(model,
        t[i] + log(sum(exp(y[i, j] - amin) for j in 1:n_output)) >= maximum(y[i, :] - amax)
    )
    @NLconstraint(model,
        t[i] + log(sum(exp(y[i, j] - amax) for j in 1:n_output)) <= minimum(y[i, :] - amin)
    )
end

# linear constraints on w using McCormick envelopes
for i in 1:n_data
    for j in 1:n_output
        @constraint(model,
            w[i, j] >= exp(tmin + log(sum(exp(amin[j] - amax) for k in 1:n_output)) - xmax[j])
        )
        @constraint(model,
            w[i, j] >= exp(tmax + log(sum(exp(amax[j] - amin) for k in 1:n_output)) - xmin[j])
        )
        @constraint(model,
            w[i, j] <= exp(tmin + log(sum(exp(amax[j] - amin) for k in 1:n_output)) - xmin[j])
        )
        @constraint(model,
            w[i, j] <= exp(tmax + log(sum(exp(amin[j] - amax) for k in 1:n_output)) - xmax[j])
        )
    end
end

# summation constraints on w using McCormick envelopes
for i in 1:n_data
    @constraint(model,
        sum(w[i, j] for j in 1:n_output) <= 1.0
    )
    @constraint(model,
        sum(w[i, j] for j in 1:n_output) >= 1.0
    )
end

# objective function (cross-entropy loss)
@objective(model,
    Min,
    sum(-Y[i, j] * log(w[i, j]) for i in 1:n_data, j in 1:n_output)
)