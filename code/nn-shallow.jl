using LinearAlgebra

# Define the independent variables X and dependent variable y
X = [1 2; 2 3; 3 4]
y = [2, 4, 5]

# Add a column of ones to X to represent the constant term
X = hcat(ones(size(X,1)), X)

# Initialize the weights (W) with random values
W = rand(size(X,2))

# Define the learning rate
α = 0.01

# Define the number of iterations
N = 1000

for i in 1:N
    # Calculate the predictions using the equation ŷ = XW
    ŷ = X * W
    
    # Calculate the gradient of the cost function with respect to W
    ∇J = X' * (ŷ - y)
    
    # Update W using gradient descent
    W -= α * ∇J
end

# Make predictions using the equation ŷ = XW
ŷ = X * W

println("The simple neural network equation using gradient descent is: ŷ = $W")