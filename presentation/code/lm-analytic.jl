using Statistics

# Define the independent variable x and dependent variable y
x = [1, 2, 3]
y = [2, 4, 5]

# Calculate the mean of x and y
x̄ = mean(x)
ȳ = mean(y)

# Calculate the slope (β1) using the formula β1 = Σ[(xi - x̄)(yi - ȳ)] / Σ[(xi - x̄)^2]
β1 = sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄).^2)

# Calculate the y-intercept (β0) using the formula β0 = ȳ - β1 * x̄
β0 = ȳ - β1 * x̄

# Make predictions using the equation ŷi = β0 + β1 * xi
ŷ = β0 .+ β1 .* x

println("The linear regression equation is: ŷ = $β0 + $β1 * x")