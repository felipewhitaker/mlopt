using LinearAlgebra

# Define the independent variables X and dependent variable y
X = [1 2; 2 3; 3 4]
y = [2, 4, 5]

# Add a column of ones to X to represent the constant term
X = hcat(ones(size(X,1)), X)

# Define the ridge parameter
λ = 0.1

# Initialize the coefficients (β) with random values
β = rand(size(X,2))

# Define the learning rate
α = 0.01

# Define the number of iterations
N = 1000

for i in 1:N
    # Calculate the gradient of the cost function with respect to β
    ∇J = X' * (X * β - y) + λ * β
    
    # Update β using gradient descent
    β -= α * ∇J
end

# Make predictions using the equation ŷ = Xβ
ŷ = X * β

println("The multiple linear regression equation with ridge regularization using gradient descent is: ŷ = $β")