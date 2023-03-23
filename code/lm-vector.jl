using LinearAlgebra

# Define the independent variables X and dependent variable y
X = [1 2; 2 3; 3 4]
y = [2, 4, 5]

# Add a column of ones to X to represent the constant term
X = hcat(ones(size(X,1)), X)

# Define the ridge parameter
λ = 1e-2

# Calculate the coefficients (β) using the formula β = (X'X + λI)^-1X'y
β = inv(X' * X + λ * I) * X' * y # FIXME ridge regularization is used to allow the matrix to be inversible

# Make predictions using the equation ŷ = Xβ
ŷ = X * β

println("The multiple linear regression equation with ridge regularization is: ŷ = $β")