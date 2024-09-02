import torch
import matplotlib.pyplot as plt
import numpy as np

# Observed/training input and output for NAND operator
x_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
y_train = torch.tensor([[1.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

class NANDOperatorModel:
    def __init__(self):
        # Model variables
        self.W = torch.rand((2, 1), requires_grad=True)  # Initialize weights for 2 inputs
        self.b = torch.rand((1,1), requires_grad=True)  # Initialize bias

    # Predictor
    def f(self, x):
        return torch.sigmoid(x @ self.W + self.b)  # Apply sigmoid for binary classification

    # Uses Binary Cross Entropy Loss
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy(self.f(x), y)

model = NANDOperatorModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.Adam([model.W, model.b], lr=0.1)
for epoch in range(10000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b
    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
fig = plt.figure("The logical NAND operator")
plot1 = fig.add_subplot(111, projection='3d')

# Plot the original training points
plot1.scatter(x_train[:, 0].numpy(), x_train[:, 1].numpy(), y_train[:, 0].numpy(), color="green", label="$(x_1^{(i)}, x_2^{(i)},y^{(i)})$")

# Generate a grid of points
x1_grid, x2_grid = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
y_grid = np.empty([10, 10])

# Calculate the model's output for each point in the grid
for i in range(x1_grid.shape[0]):
    for j in range(x1_grid.shape[1]):
        y_grid[i, j] = model.f(torch.tensor([[x1_grid[i, j], x2_grid[i, j]]], dtype=torch.float)).item()

# Plot the wireframe representing the model's predictions
plot1.plot_wireframe(x1_grid, x2_grid, y_grid, color="blue", label="$\\hat y=f(\\mathbf{x})=\\sigma(\\mathbf{xW}+b)$")

# Set labels and legend
plot1.set_xlabel("$x_1$")
plot1.set_ylabel("$x_2$")
plot1.set_zlabel("$y$")
plot1.legend()
# Display the plot
plt.show()
