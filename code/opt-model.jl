using JuMP
using Ipopt # solver for nonlinear optimization

# Define activation functions (sigmoid)
function sigmoid(x)
    return 1 / (1 + exp(-x))
end

function linear(x)
    return x
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
@variable(model, out, ...?) # FIXME idea: create out variable for each layer to apply non-linearity

@variable(model, yhat[i in 1:n_data])

@constraint(
    model,
    yhat[i] == linear(
        sum(
            linear.(
                sum(
                    W1[:, j] .* X[i,:] .+ b1[j]
                    for j in 1:n_hidden
                )
            ) * W2[:, k] .+ b2[k]
            for k in 1:n_output,
            axis = 2
        )
    )
)

# Minimize objective function
@NLobjective(model, Min, mse(y, yhat))

# Solve model
optimize!(model)

# Print optimal values
println("Optimal W_11: ", value.(W_11))