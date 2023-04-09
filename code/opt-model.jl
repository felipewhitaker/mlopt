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