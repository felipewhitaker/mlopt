using LinearAlgebra
using Distributions

# Define the independent variable x and dependent variable y
x = 0:0.1:2π
y = sin.(x) + rand(Truncated(Normal(0, 1), 0, 1), size(x, 1))

# Define the number of hidden units
H = 8

# Initialize the weights (W1 and W2) with random values
W1 = rand(H, size(x,2))
W2 = rand(1, H)

# Define the learning rate
α = 0.01

# Define the number of iterations
N = 10

for i in 1:N
    # Forward pass
    
    # Calculate the hidden layer activations using the equation h = max(0, x * W1')
    h = max.(0, x * W1')
    
    # Calculate the predictions using the equation ŷ = h * W2'
    ŷ = h * W2'
    
    # Backward pass
    
    # Calculate the gradient of the cost function with respect to W2
    ∇J_W2 = (ŷ - y)' * h
    
    # Calculate the gradient of the cost function with respect to W1
    ∇J_W1 = (((ŷ - y) * W2) .* (h .> 0))' * x
    
    # Update W1 and W2 using gradient descent
    W1 -= α * ∇J_W1
    W2 -= α * ∇J_W2

end

# Make predictions using forward propagation

# Calculate the hidden layer activations using the equation h = max(0, x * W1')
h = max.(0, x * W1')

# Calculate the predictions using the equation ŷ = h * W2'
ŷ = h * W2'

println("The neural network with one hidden layer and ReLU activation function is: ŷ = $ŷ")

plot(x, sin.(x))
plot!(x, [y, ŷ])