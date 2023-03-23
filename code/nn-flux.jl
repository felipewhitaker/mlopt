using Flux

# Define the independent variable x and dependent variable y
x = collect(0:0.05:2π)
y = sin.(x)

# Reshape x into a matrix with one row
X = reshape(x, 1, :)

# Define the model as a chain of layers with a ReLU activation function in the hidden layer
hidden_size = 16
model = Chain(
    Dense(1, hidden_size, relu),
    Dense(hidden_size, 1)
)

# Define the loss function as the mean squared error
loss(X, y) = Flux.mse(model(X), y)

# Define the optimizer as stochastic gradient descent with a learning rate of 0.01
opt = Descent(0.01)

# Train the model for 100 epochs
for i in 1:1000
    Flux.train!(loss, model, [(X,y)], opt)
end

# Make predictions using the trained model
ŷ = model(X)

# println("The fitted equation is: ŷ = $ŷ")

using Plots
plot(x, [y, ŷ'])
