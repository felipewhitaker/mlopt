using JuMP
using Ipopt # solver for nonlinear optimization

# Define activation functions (sigmoid)
function sigmoid(x)
    return 1 / (1 + exp(-x))
end

function sigmoid_derivative(x)
    return sigmoid(x) * (1 - sigmoid(x))
end

# Define objective function (mean squared error)
function mse(y_true,y_pred)
    return sum((y_true - y_pred).^2) / length(y_true)
end

# Define some constants
n_input = 2 # number of input nodes
n_hidden = 3 # number of hidden nodes
n_output = 1 # number of output nodes
n_data = 4 # number of data samples

# Define some sample data (XOR problem)
X = [0 0; 0 1; 1 0; 1 1] # input matrix
y = [0; 1; 1; 0] # output vector

# Define a model
model = Model(Ipopt.Optimizer)

# Define variables
@variable(model, W1[i=1:n_input,j=1:n_hidden]) # weight matrix from input to hidden layer
@variable(model, b1[j=1:n_hidden]) # bias vector for hidden layer
@variable(model, W2[j=1:n_hidden,k=1:n_output]) # weight matrix from hidden to output layer
@variable(model, b2[k=1:n_output]) # bias vector for output layer

# Define constraints (forward propagation)

# FIXME how to implement a neural network with Ipopt?
# FIXME nonlinear constraints: how to create yhat to compare it to y? shouldn't there be a tolerance?

@NLexpression(model, yhat == sigmoid.(sigmoid.(X * W1 .+ b1') * W2 .+ b2'))

@NLconstraint(
    model, 
    [i=1:n_data],
    Y == yhat
    # Y[i] == sigmoid.(
    #     sum(
    #         W2[:,k] .* sigmoid.(sum(W1[:,j].* X[i,:] .+ b1[j]) for j in 1:n_hidden) .+ b2[k]
    #     ) for k in n_output
    # )
)

# Minimize objective function
@NLobjective(model, :Min, mse(y, yhat))

# Solve model
optimize!(model)

# Print optimal values
println("Optimal W_11: ", value.(W_11))